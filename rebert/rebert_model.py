import copy
from typing import Optional, Union, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, BaseModelOutput
from transformers.utils import DUMMY_INPUTS, DUMMY_MASK, is_torch_fx_proxy

from transformers.utils import logging

logger = logging.get_logger(__name__)


class ReBertConfig(PretrainedConfig):
    model_type = "rebert"

    def __init__(
            self,
            vocab_size=28996,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-6,
            initializer_range=0.02,
            is_encoder_decoder=True,
            use_cache=True,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            init_pos=512,
            classifier_dropout=0.0,
            **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id,
                         bos_token_id=bos_token_id,
                         eos_token_id=eos_token_id,
                         is_encoder_decoder=is_encoder_decoder,
                         use_cache=use_cache,
                         **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout = classifier_dropout
        self.initializer_range = initializer_range
        self.init_pos = init_pos


class ROPEEmbedding(nn.Module):

    def __init__(self, d_model: int, max_seq: int, theta=10000, device=None):
        super().__init__()
        self.theta = theta
        self.d_model = d_model
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.d_model, 2).float().to(device) / self.d_model))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq = max_seq
        cos, sin = self.calculate_rope(self.max_seq, device=device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def calculate_rope(self, max_length, device=None, dtype=None):
        if dtype is None:
            dtype = torch.get_default_dtype()
        t = torch.arange(max_length, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freq)
        # pairs dimension i with dimension i + d // 2
        # Applies a permutation over the original paper without changing the resulting inner product
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def resize_rope(self, length, device=None, dtype=None):
        if device is None:
            device = self.cos.device
        if dtype is None:
            dtype = self.cos.dtype
        cos, sin = self.calculate_rope(length, device=device, dtype=dtype)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.max_seq = length

    def forward(self, seq: torch.Tensor, length):
        if length > self.max_seq:
            self.resize_rope(length, seq.device, seq.dtype)
        return self.cos[:length], self.sin[:length]

    @staticmethod
    def apply_embedding(seq, cos, sin, position_ids, unsqueeze_dim=1):
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        n_rotations = seq.shape[-1] // 2
        seq_split = torch.cat([-seq[..., n_rotations:], seq[..., :n_rotations]], dim=-1)
        roped = seq * cos + seq_split * sin
        return roped


def multihead_view(proj: torch.Tensor, heads, head_size, transpose=False):
    proj_view = proj.view(proj.shape[0], proj.shape[1], heads, head_size)
    if not transpose:
        return proj_view.permute(0, 2, 1, 3)
    else:
        return proj_view.permute(0, 2, 3, 1)


class ReBertAttention(nn.Module):

    def __init__(self, d_model: int, attention_head: int,
                 num_key_value_heads: int, rope: ROPEEmbedding,
                 attn_dropout: float = 0.1, hidden_dropout: float = 0.1,
                 is_decoder=False):
        super().__init__()
        assert d_model % attention_head == 0

        self.heads = attention_head
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.heads // self.num_key_value_heads
        self.size_per_head = d_model // self.heads
        self.d_model = d_model
        self.is_decoder = is_decoder
        self.gradient_checkpointing = False

        self.q_proj = nn.Linear(in_features=self.d_model,
                                out_features=self.heads * self.size_per_head,
                                bias=False)
        self.k_proj = nn.Linear(in_features=self.d_model,
                                out_features=self.num_key_value_heads * self.size_per_head,
                                bias=False)
        self.v_proj = nn.Linear(in_features=self.d_model,
                                out_features=self.num_key_value_heads * self.size_per_head,
                                bias=False)
        self.rope = rope
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.o_proj = nn.Linear(in_features=self.size_per_head * self.heads,
                                out_features=self.d_model,
                                bias=False)
        self.output_dropout = nn.Dropout(hidden_dropout)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, position_ids: torch.LongTensor,
                key_value_states: torch.Tensor = None,
                past_key_value: torch.Tensor = None,
                cross_position_ids: torch.LongTensor = None,
                use_cache=False,
                output_attentions=False):
        # Input is (batch_size, seq_length, dim)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = seq.shape[:2]
        real_seq_length = seq_length
        if cross_position_ids is None:
            cross_position_ids = position_ids

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
                )
            real_seq_length = int(position_ids[0, -1]) + 1
        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        k_proj, q_proj, v_proj = self.map_kqv(seq, position_ids, cross_position_ids, key_length, real_seq_length,
                                              key_value_states, past_key_value)
        k_proj = torch.repeat_interleave(k_proj, self.num_key_value_groups, dim=1)
        v_proj = torch.repeat_interleave(v_proj, self.num_key_value_groups, dim=1)
        k_proj = k_proj.transpose(2, 3)

        raw_scores = torch.matmul(q_proj, k_proj) / np.sqrt(q_proj.shape[-1])
        if mask is not None:
            masked_scores = raw_scores + mask
        else:
            masked_scores = raw_scores
        scaled_scores = torch.nn.functional.softmax(masked_scores, dim=-1, dtype=torch.float32).to(q_proj.dtype)
        scaled_scores = self.attn_dropout(scaled_scores)
        results = torch.matmul(scaled_scores, v_proj).permute(0, 2, 1, 3)
        combined = results.reshape(seq.shape[0], seq.shape[1], -1)
        o_proj = self.map_output(combined)

        present_key_value_state = (k_proj.transpose(2, 3), v_proj) if (self.is_decoder and use_cache) else None
        outputs = (o_proj,) + (present_key_value_state,)

        if output_attentions:
            outputs = outputs + (scaled_scores,)
        return outputs

    def map_kqv(self, seq: torch.Tensor, position_ids: torch.LongTensor,
                cross_position_ids: torch.LongTensor, k_len: int, q_len: int,
                key_value_states: torch.Tensor, past_key_value: torch.Tensor):
        q_state = multihead_view(self.q_proj(seq), self.heads, self.size_per_head)
        q_cos, q_sin = self.rope(q_state, length=q_len)
        q_state = self.rope.apply_embedding(q_state, q_cos, q_sin, position_ids)

        key_states = self.project(
            seq, self.k_proj, key_value_states,
            past_key_value[0] if past_key_value is not None else None
        )
        value_states = self.project(
            seq, self.v_proj, key_value_states,
            past_key_value[1] if past_key_value is not None else None
        )
        k_cos, k_sin = self.rope(key_states, length=k_len)
        key_states = self.rope.apply_embedding(key_states, k_cos, k_sin, cross_position_ids)

        return key_states, q_state, value_states

    def map_output(self, combined: torch.Tensor):
        return self.output_dropout(self.o_proj(combined))

    def project(self, hidden_states, proj_func, key_value_states, past_key_value):
        """projects hidden states correctly to key/query states"""
        if key_value_states is None:
            # self-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = proj_func(hidden_states)
            hidden_states = multihead_view(hidden_states, self.num_key_value_heads, self.size_per_head)
        elif past_key_value is None:
            # cross-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = proj_func(key_value_states)
            hidden_states = multihead_view(hidden_states, self.num_key_value_heads, self.size_per_head)

        if past_key_value is not None:
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, key_length, dim_per_head)
                hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
            else:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = proj_func(key_value_states)
                hidden_states = multihead_view(hidden_states, self.num_key_value_heads, self.size_per_head)
        return hidden_states


class ReBertSelfAttention(nn.Module):

    def __init__(self, config: ReBertConfig, rope: ROPEEmbedding):
        super().__init__()
        self.self = ReBertAttention(d_model=config.hidden_size, attention_head=config.num_attention_heads,
                                    num_key_value_heads=config.num_key_value_heads,
                                    rope=rope,
                                    attn_dropout=config.attention_probs_dropout_prob,
                                    hidden_dropout=config.hidden_dropout_prob, is_decoder=config.is_decoder)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.self(
            normed_hidden_states,
            mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]
        return outputs


class ReBertCrossAttention(nn.Module):
    def __init__(self, config: ReBertConfig, rope: ROPEEmbedding):
        super().__init__()
        self.cross = ReBertAttention(d_model=config.hidden_size, attention_head=config.num_attention_heads,
                                     num_key_value_heads=config.num_key_value_heads,
                                     rope=rope,
                                     attn_dropout=config.attention_probs_dropout_prob,
                                     hidden_dropout=config.hidden_dropout_prob, is_decoder=config.is_decoder)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            hidden_states,
            key_value_states,
            attention_mask=None,
            position_ids=None,
            cross_position_ids=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.cross(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_ids=position_ids,
            cross_position_ids=cross_position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]
        return outputs


class ReBertMLP(nn.Module):

    def __init__(self, config: ReBertConfig):
        super().__init__()
        self.d_model = config.hidden_size
        self.intermediate = config.intermediate_size

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate_proj = nn.Linear(in_features=self.d_model, out_features=self.intermediate)
        self.intermediate_act = getattr(nn, config.hidden_act.upper())()
        self.intermediate_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(in_features=self.intermediate, out_features=self.d_model)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, seq: torch.Tensor):
        normed_out = self.layer_norm(seq)
        inter_out = self.intermediate_act(self.intermediate_proj(normed_out))
        inter_out = self.intermediate_dropout(inter_out)
        layer_out = seq + self.out_dropout(self.out_proj(inter_out))
        return layer_out


class ReBertLayer(nn.Module):

    def __init__(self, config: ReBertConfig, rope: ROPEEmbedding):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(ReBertSelfAttention(config, rope))
        if self.is_decoder:
            self.layer.append(ReBertCrossAttention(config, rope))
        self.layer.append(ReBertMLP(config))

    def forward(self, hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                position_ids=None,
                cross_position_ids=None,
                past_key_value=None,
                use_cache=False,
                output_attentions=False):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_ids=position_ids,
                cross_position_ids=cross_position_ids,
                past_key_value=cross_attn_past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs


class ReBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        last_token = hidden_states[:, -1]
        pooled_output = self.dense(last_token)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ReBertPreTrainedModel(PreTrainedModel):
    config_class = ReBertConfig
    base_model_prefix = "rebert"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ReBertLayer"]

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined."
            )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear,)):
            # Slightly different from Mesh Transformer JAX which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class ReBertStack(ReBertPreTrainedModel):

    def __init__(self, config: ReBertConfig, embedding: nn.Embedding, rope: ROPEEmbedding):
        super().__init__(config)
        self.embedding = embedding
        self.rope = rope
        self.is_decoder = config.is_decoder
        self.gradient_checkpointing = False

        self.block = nn.ModuleList(
            [ReBertLayer(config, rope) for _ in range(config.num_hidden_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, new_embeddings):
        self.embedding = new_embeddings

    def forward(
            self,
            input_ids,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = self.embedding(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
        mask_seq_length = past_key_values_length + seq_length

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # Construct position ids for ROPE
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=inputs_embeds.device
        )
        position_ids = position_ids.unsqueeze(0)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, 1, encoder_sequence_length, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            cross_position_ids = torch.arange(0, encoder_sequence_length, dtype=torch.long, device=inputs_embeds.device)
            cross_position_ids = cross_position_ids.unsqueeze(0)
        else:
            encoder_extended_attention_mask = None
            cross_position_ids = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None

        hidden_states = self.dropout(inputs_embeds)
        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    extended_attention_mask,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    position_ids,
                    cross_position_ids,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    position_ids=position_ids,
                    cross_position_ids=cross_position_ids,
                    past_key_value=past_key_value,  # past_key_value is always None with gradient checkpointing
                    use_cache=use_cache,
                    output_attentions=output_attentions
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[3],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class ReBertForConditionalGeneration(ReBertPreTrainedModel):
    _tied_weights_keys = ["encoder.embedding.weight", "decoder.embedding.weight", "lm_head.weight"]

    def __init__(self, config: ReBertConfig):
        super().__init__(config)
        self.model_dim = config.hidden_size
        self.shared = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=self.model_dim,
                                   padding_idx=config.pad_token_id)
        self.rope = ROPEEmbedding(config.hidden_size // config.num_attention_heads, max_seq=config.init_pos)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = True
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = ReBertStack(encoder_config, self.shared, self.rope)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        self.decoder = ReBertStack(decoder_config, self.shared, self.rope)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embedding, self.shared)
            self._tie_or_clone_weights(self.decoder.embedding, self.shared)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            decoder_attention_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past