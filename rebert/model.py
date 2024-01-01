from abc import ABC
from typing import Optional, Union, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, MaskedLMOutput, \
    SequenceClassifierOutput


class ReBertConfig(PretrainedConfig):
    model_type = "rebert"

    def __init__(
            self,
            vocab_size=28996,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=4,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            classifier_dropout=None,
            **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
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


def multihead_view(proj: torch.Tensor, heads, head_size, transpose=False):
    proj_view = proj.view(proj.shape[0], proj.shape[1], heads, head_size)
    if not transpose:
        return proj_view.permute(0, 2, 1, 3)
    else:
        return proj_view.permute(0, 2, 3, 1)


class ReBertBaseSelfAttention(nn.Module):

    def __init__(self, d_model: int, attention_head: int, num_key_value_heads: int, attn_dropout: float = 0.1):
        super().__init__()
        assert d_model % attention_head == 0

        self.heads = attention_head
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.heads // self.num_key_value_heads
        self.size_per_head = d_model // self.heads
        self.d_model = d_model

        self.q_proj = nn.Linear(in_features=self.d_model, out_features=self.heads * self.size_per_head)
        self.k_proj = nn.Linear(in_features=self.d_model, out_features=self.num_key_value_heads * self.size_per_head)
        self.v_proj = nn.Linear(in_features=self.d_model, out_features=self.num_key_value_heads * self.size_per_head)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, position_ids: torch.LongTensor = None,
                output_attentions=False):
        k_proj, q_proj, v_proj = self.map_kqv(seq, position_ids)
        k_proj = torch.repeat_interleave(k_proj, self.num_key_value_groups, dim=1)
        v_proj = torch.repeat_interleave(v_proj, self.num_key_value_groups, dim=1)
        raw_scores = torch.matmul(q_proj, k_proj) / np.sqrt(q_proj.shape[-1])
        masked_scores = raw_scores + mask
        scaled_scores = torch.nn.functional.softmax(masked_scores, dim=-1)
        scaled_scores = self.attn_dropout(scaled_scores)
        results = torch.matmul(scaled_scores, v_proj).permute(0, 2, 1, 3)
        combined = results.reshape(seq.shape[0], seq.shape[1], -1)
        if not output_attentions:
            return combined,
        else:
            return combined, scaled_scores

    def map_kqv(self, seq: torch.Tensor, position_ids: torch.LongTensor):
        k_proj = self.map_key(seq, position_ids)
        q_proj = self.map_query(seq, position_ids)
        v_proj = self.map_value(seq, position_ids)

        return k_proj, q_proj, v_proj

    def map_key(self, seq: torch.Tensor, position_ids: torch.LongTensor):
        return multihead_view(self.k_proj(seq), self.num_key_value_heads, self.size_per_head, transpose=True)

    def map_query(self, seq: torch.Tensor, position_ids: torch.LongTensor):
        return multihead_view(self.q_proj(seq), self.heads, self.size_per_head)

    def map_value(self, seq: torch.Tensor, position_ids: torch.LongTensor):
        return multihead_view(self.v_proj(seq), self.num_key_value_heads, self.size_per_head)


class ReBertBaseMultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, attention_head: int, num_key_value_heads: int,
                 attn_dropout: float = 0.1,
                 hidden_dropout: float = 0.1, layer_norm_eps: float = 1e-12):
        super().__init__()
        assert d_model % attention_head == 0

        self.heads = attention_head
        self.size_per_head = d_model // self.heads
        self.d_model = d_model

        self.self_attention = ReBertBaseSelfAttention(d_model, attention_head, num_key_value_heads, attn_dropout)
        self.o_proj = nn.Linear(in_features=self.size_per_head * self.heads, out_features=self.d_model)
        self.output_dropout = nn.Dropout(hidden_dropout)
        self.prelayer_norm = nn.LayerNorm(self.d_model, eps=layer_norm_eps)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, position_ids: torch.LongTensor = None,
                output_attentions=False):
        seq = self.prelayer_norm(seq)
        attention_out = self.self_attention(seq, mask, position_ids, output_attentions=output_attentions)
        o_proj = self.map_output(attention_out[0])
        new_embedding = o_proj + seq
        if not output_attentions:
            return new_embedding,
        else:
            return new_embedding, attention_out[1]

    def map_output(self, combined: torch.Tensor):
        return self.output_dropout(self.o_proj(combined))


class FirstTokenPooler(nn.Module):

    def __init__(self, config: ReBertConfig):
        super().__init__()
        self.pool_proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.pool_act = nn.Tanh()

    def forward(self, seq: torch.Tensor):
        return self.pool_act(self.pool_proj(seq[:, 0, :]))


class ReBertEmbedding(nn.Module):

    def __init__(self, vocab_size: int, d_model: int, vocab_pad: int = 0, layer_norm_eps=1e-12, dropout=0.1):
        super().__init__()
        self.word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model,
                                           padding_idx=vocab_pad)
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, seq):
        return self.dropout(self.layer_norm(self.word_embedding(seq)))


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
        # Applies a permutation over the original paper without changing the inner product between k and q
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


class ReBertSelfAttention(ReBertBaseSelfAttention):

    def __init__(self, d_model: int, attention_head: int, num_key_value_heads: int,
                 rope: ROPEEmbedding, attn_dropout: float = 0.1):
        super().__init__(d_model=d_model, attention_head=attention_head,
                         num_key_value_heads=num_key_value_heads,
                         attn_dropout=attn_dropout)
        self.rope = rope

    def map_key(self, seq: torch.Tensor, position_ids: torch.LongTensor):
        k_proj = multihead_view(self.k_proj(seq), self.num_key_value_heads, self.size_per_head)
        cos, sin = self.rope(k_proj, length=seq.shape[-2])
        k_proj = self.rope.apply_embedding(k_proj, cos, sin, position_ids)
        return k_proj.transpose(2, 3)

    def map_query(self, seq: torch.Tensor, position_ids: torch.LongTensor):
        q_proj = multihead_view(self.q_proj(seq), self.heads, self.size_per_head)
        cos, sin = self.rope(q_proj, length=seq.shape[-2])
        q_proj = self.rope.apply_embedding(q_proj, cos, sin, position_ids)
        return q_proj


class ReBertMultiHeadAttention(ReBertBaseMultiHeadAttention):

    def __init__(self, d_model: int, attention_head: int, num_key_value_heads: int,
                 rope: ROPEEmbedding, attn_dropout: float = 0.1,
                 hidden_dropout: float = 0.1, layer_norm_eps: float = 1e-12):
        super().__init__(d_model, attention_head, num_key_value_heads,
                         attn_dropout, hidden_dropout, layer_norm_eps)
        self.self_attention = ReBertSelfAttention(d_model, attention_head,
                                                  num_key_value_heads, rope, attn_dropout)


class ReBertEncoderLayer(nn.Module):

    def __init__(self, config: ReBertConfig, rope: ROPEEmbedding):
        super().__init__()
        self.d_model = config.hidden_size
        self.intermediate = config.intermediate_size
        self.attention = ReBertMultiHeadAttention(d_model=config.hidden_size, attention_head=config.num_attention_heads,
                                                  num_key_value_heads=config.num_key_value_heads,
                                                  rope=rope,
                                                  attn_dropout=config.attention_probs_dropout_prob,
                                                  hidden_dropout=config.hidden_dropout_prob,
                                                  layer_norm_eps=config.layer_norm_eps)
        self.intermediate_proj = nn.Linear(in_features=self.d_model, out_features=self.intermediate)
        self.intermediate_act = getattr(nn, config.hidden_act.upper())()
        self.out_proj = nn.Linear(in_features=self.intermediate, out_features=self.d_model)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_norm = nn.LayerNorm(self.d_model, eps=config.layer_norm_eps)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, position_ids: torch.LongTensor = None,
                output_attentions: bool = False):
        attn_out = self.attention(seq, mask, position_ids, output_attentions=output_attentions)
        inter_out = self.intermediate_act(self.intermediate_proj(attn_out[0]))
        layer_out = self.out_dropout(self.out_proj(inter_out))
        if output_attentions:
            return self.out_norm(layer_out + attn_out[0]), attn_out[1]
        return self.out_norm(layer_out + attn_out[0]),


class ReBertEncoder(nn.Module):

    def __init__(self, config: ReBertConfig):
        super().__init__()
        self.rope = ROPEEmbedding(config.hidden_size // config.num_attention_heads, config.max_length)
        # self.rope = LlamaRotaryEmbedding(dim=config.hidden_size // config.num_attention_heads)
        self.encoder_layers = nn.ModuleList(
            [ReBertEncoderLayer(config, rope=self.rope) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, position_ids: torch.LongTensor = None,
                output_attentions: bool = False,
                output_hidden_states: bool = False,
                return_dict: bool = True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        final_embeddings = seq
        for layer in self.encoder_layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (final_embeddings,)
            if self.gradient_checkpointing and self.training:
                layer_out = self._gradient_checkpointing_func(
                    layer.__call__,
                    final_embeddings,
                    mask,
                    position_ids,
                    output_attentions
                )
            else:
                layer_out = layer(final_embeddings, mask, position_ids, output_attentions)
            final_embeddings = layer_out[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_out[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (final_embeddings,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    final_embeddings,
                    None,
                    all_hidden_states,
                    all_self_attentions
                ]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=final_embeddings,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ReBertPreTrainedModel(PreTrainedModel):
    config_class = ReBertConfig
    base_model_prefix = "rebert"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ReBertEmbedding", "ReBertSelfAttention"]


class ReBertModel(ReBertPreTrainedModel):

    def __init__(self, config: ReBertConfig, add_pooling_layer=True):
        super(ReBertModel, self).__init__(config)
        self.embedding = ReBertEmbedding(config.vocab_size, config.hidden_size, config.pad_token_id,
                                         config.layer_norm_eps, config.hidden_dropout_prob)
        self.encoder = ReBertEncoder(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = FirstTokenPooler(config) if add_pooling_layer else None
        self.post_init()

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        batch_size, seq_length = input_shape = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        embedding_output = self.embedding(input_ids)
        encoder_outputs = self.encoder(
            embedding_output,
            mask=extended_attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = self.layer_norm(encoder_outputs[0])
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ReBertLMHead(nn.Module):

    def __init__(self, config: ReBertConfig):
        super().__init__()
        self.transform = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.transform_act = getattr(nn, config.hidden_act.upper())()
        self.layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, seq):
        transformed_embeddings = self.layer_norm(self.transform_act(self.transform(seq)))
        scores = self.decoder(transformed_embeddings)
        return scores


class ReBertForMaskedLM(ReBertPreTrainedModel):

    def __init__(self, config: ReBertConfig):
        super().__init__(config)
        self.rebert = ReBertModel(config, add_pooling_layer=False)
        self.mlm_head = ReBertLMHead(config)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.rebert(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.mlm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(prediction_scores.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ReBertForSequenceClassification(ReBertPreTrainedModel):

    def __init__(self, config: ReBertConfig):
        super().__init__(config)
        self.rebert = ReBertModel(config)
        self.num_labels = config.num_labels

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.rebert(
            input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
