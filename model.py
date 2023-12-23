from typing import List, Iterable, Dict, Any

import torch
from torch import nn
import numpy as np
import random
import tensorflow as tf
from torch.nn import ModuleList
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings


class BertConfig:

    def __init__(self, vocab_size: int, vocab_pad: int = 0, d_model: int = 768, inter_size: int = 3072,
                 inter_activation: str = "GELU", seq_len: int = 512, attention_heads=12,
                 encoder_layers=12, layer_norm_eps=1e-12, hidden_dropout=0.1,
                 attn_dropout=0.1, attention_layer: str = "BertMultiHeadAttention",
                 embedding_layer: str = "BertEmbedding"):
        self.vocab_size = vocab_size
        self.vocab_pad = vocab_pad
        self.d_model = d_model
        self.seq_len = seq_len
        self.layer_norm_eps = layer_norm_eps
        self.attention_head = attention_heads
        self.hidden_dropout = hidden_dropout
        self.attn_dropout = attn_dropout
        self.inter_size = inter_size
        self.inter_activation = inter_activation
        self.encoder_layers = encoder_layers
        self.attention_layer = attention_layer
        self.embedding_layer = embedding_layer


class BertEmbedding(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.d_model,
                                           padding_idx=config.vocab_pad)
        self.segment_embedding = nn.Embedding(num_embeddings=2, embedding_dim=config.d_model)
        self.pos_embedding = nn.Embedding(num_embeddings=config.seq_len, embedding_dim=config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout)

    def forward(self, seq, seq_seg):
        seq_positions = torch.IntTensor(list(range(seq.shape[1])))
        embedding = self.word_embedding(seq) + self.pos_embedding(seq_positions) + self.segment_embedding(seq_seg)
        return self.dropout(self.layer_norm(embedding))


class BertMultiHeadAttention(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()
        assert config.d_model % config.attention_head == 0

        self.heads = config.attention_head
        self.size_per_head = config.d_model // self.heads
        self.d_model = config.d_model

        self.q_proj = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.k_proj = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.v_proj = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.o_proj = nn.Linear(in_features=self.size_per_head * self.heads, out_features=self.d_model)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.output_dropout = nn.Dropout(config.hidden_dropout)
        self.output_norm = nn.LayerNorm(self.d_model, eps=config.layer_norm_eps)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, output_attentions=False):
        k_proj = self.map_key(seq)
        q_proj = self.map_query(seq)
        v_proj = self.map_value(seq)

        raw_scores = torch.matmul(q_proj, k_proj) / np.sqrt(q_proj.shape[-1])
        masked_scores = raw_scores.masked_fill(mask == 0, -10000)
        scaled_scores = torch.nn.functional.softmax(masked_scores, dim=-1)
        scaled_scores = self.attn_dropout(scaled_scores)
        results = torch.matmul(scaled_scores, v_proj).permute(0, 2, 1, 3)
        combined = results.reshape(seq.shape[0], seq.shape[1], -1)

        o_proj = self.map_output(combined)
        new_embedding = self.output_norm(o_proj + seq)
        if not output_attentions:
            return new_embedding
        else:
            return new_embedding, scaled_scores

    def multihead_view(self, proj: torch.Tensor, transpose=False):
        proj_view = proj.view(proj.shape[0], proj.shape[1], self.heads, self.size_per_head)
        if not transpose:
            return proj_view.permute(0, 2, 1, 3)
        else:
            return proj_view.permute(0, 2, 3, 1)

    def map_key(self, seq: torch.Tensor):
        return self.multihead_view(self.k_proj(seq), transpose=True)

    def map_query(self, seq: torch.Tensor):
        return self.multihead_view(self.q_proj(seq))

    def map_value(self, seq: torch.Tensor):
        return self.multihead_view(self.v_proj(seq))

    def map_output(self, combined: torch.Tensor):
        return self.output_dropout(self.o_proj(combined))


class BertEncoderLayer(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()
        self.d_model = config.d_model
        self.intermediate = config.inter_size
        self.attention = None
        if config.attention_layer == "BertMultiHeadAttention":
            self.attention = BertMultiHeadAttention(config)
        else:
            raise ValueError(config.attention_layer)
        self.intermediate_proj = nn.Linear(in_features=self.d_model, out_features=self.intermediate)
        self.intermediate_act = getattr(nn, config.inter_activation)()
        self.out_proj = nn.Linear(in_features=self.intermediate, out_features=self.d_model)
        self.out_dropout = nn.Dropout(config.hidden_dropout)
        self.out_norm = nn.LayerNorm(self.d_model, eps=config.layer_norm_eps)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor):
        attn_out = self.attention(seq, mask)
        inter_out = self.intermediate_act(self.intermediate_proj(attn_out))
        layer_out = self.out_dropout(self.out_proj(inter_out))
        return self.out_norm(layer_out + attn_out)


class BertEncoder(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()
        if config.embedding_layer == "BertEmbedding":
            self.embeddings = BertEmbedding(config)
        else:
            raise ValueError(config.embedding_layer)
        self.encoder_layers = nn.ModuleList([BertEncoderLayer(config) for _ in range(config.encoder_layers)])

    def forward(self, seq: torch.Tensor, mask: torch.Tensor, seq_seg: torch.Tensor):
        word_embeddings = self.embeddings(seq, seq_seg)
        final_embeddings = word_embeddings
        for layer in self.encoder_layers:
            final_embeddings = layer(final_embeddings, mask)
        return final_embeddings


class FirstTokenPooler(nn.Module):

    def __init__(self, config: BertConfig):
        super().__init__()
        self.pool_proj = nn.Linear(in_features=config.d_model, out_features=config.d_model)
        self.pool_act = nn.Tanh()

    def forward(self, seq: torch.Tensor):
        return self.pool_act(self.pool_proj(seq[:, 0, :]))


class BertMLMHead(nn.Module):

    def __init__(self, config: BertConfig, word_embedding: nn.Embedding):
        super().__init__()
        self.transform = nn.Linear(in_features=config.d_model, out_features=config.d_model)
        self.transform_act = getattr(nn, config.inter_activation)()
        self.layer_norm = nn.LayerNorm(normalized_shape=config.d_model)
        self.word_embedding = word_embedding
        self.bias = nn.Parameter(
            data=torch.rand(config.vocab_size) * 2 * np.sqrt(config.d_model) - np.sqrt(config.d_model),
            requires_grad=True)

    def forward(self, seq):
        transformed_embeddings = self.layer_norm(self.transform_act(self.transform(seq)))
        scores = torch.matmul(transformed_embeddings, self.word_embedding.weight.transpose(0, 1))
        return scores + self.bias


def load_tf_var(chpt: str, src_var: str, target: [torch.Tensor | nn.Parameter], processor=lambda x: x):
    src_val = tf.train.load_variable(chpt, src_var)
    src_val = processor(src_val)
    src_val = torch.from_numpy(src_val).float()
    target.copy_(src_val)
    assert torch.sum(target - src_val) <= 0.0001


def load_bert_embeddings(tf_chk, embedding: BertEmbedding):
    with torch.no_grad():
        # layer norm
        load_tf_var(tf_chk, "bert/embeddings/LayerNorm/gamma", embedding.layer_norm.weight)
        load_tf_var(tf_chk, "bert/embeddings/LayerNorm/beta", embedding.layer_norm.bias)

        # token embeddings
        load_tf_var(tf_chk, "bert/embeddings/word_embeddings", embedding.word_embedding.weight)
        load_tf_var(tf_chk, "bert/embeddings/position_embeddings", embedding.pos_embedding.weight)
        load_tf_var(tf_chk, "bert/embeddings/token_type_embeddings", embedding.segment_embedding.weight)


def load_bert_mha(tf_chk: str, layer: int, attention: BertMultiHeadAttention):
    with torch.no_grad():
        # K
        load_tf_var(tf_chk, f"bert/encoder/layer_{layer}/attention/self/key/bias", attention.k_proj.bias)
        load_tf_var(tf_chk, f"bert/encoder/layer_{layer}/attention/self/key/kernel", attention.k_proj.weight,
                    processor=lambda x: np.transpose(x))

        # Q
        load_tf_var(tf_chk, f"bert/encoder/layer_{layer}/attention/self/query/bias", attention.q_proj.bias)
        load_tf_var(tf_chk, f"bert/encoder/layer_{layer}/attention/self/query/kernel", attention.q_proj.weight,
                    processor=lambda x: np.transpose(x))

        # Q
        load_tf_var(tf_chk, f"bert/encoder/layer_{layer}/attention/self/value/bias", attention.v_proj.bias)
        load_tf_var(tf_chk, f"bert/encoder/layer_{layer}/attention/self/value/kernel", attention.v_proj.weight,
                    processor=lambda x: np.transpose(x))

        # Output
        load_tf_var(tf_chk, f"bert/encoder/layer_{layer}/attention/output/dense/bias", attention.o_proj.bias)
        load_tf_var(tf_chk, f"bert/encoder/layer_{layer}/attention/output/dense/kernel", attention.o_proj.weight,
                    processor=lambda x: np.transpose(x))
        load_tf_var(tf_chk, f"bert/encoder/layer_{layer}/attention/output/LayerNorm/beta", attention.output_norm.bias)
        load_tf_var(tf_chk, f"bert/encoder/layer_{layer}/attention/output/LayerNorm/gamma",
                    attention.output_norm.weight)


def load_bert_encoder(tf_chk: str, layer: int, encoder: BertEncoderLayer):
    load_bert_mha(tf_chk, layer, encoder.attention)
    with torch.no_grad():
        # Intermediate
        load_tf_var(tf_chk, f"bert/encoder/layer_{layer}/intermediate/dense/bias", encoder.intermediate_proj.bias)
        load_tf_var(tf_chk, f"bert/encoder/layer_{layer}/intermediate/dense/kernel", encoder.intermediate_proj.weight,
                    processor=lambda x: np.transpose(x))

        # Output
        load_tf_var(tf_chk, f"bert/encoder/layer_{layer}/output/dense/bias", encoder.out_proj.bias)
        load_tf_var(tf_chk, f"bert/encoder/layer_{layer}/output/dense/kernel", encoder.out_proj.weight,
                    processor=lambda x: np.transpose(x))
        load_tf_var(tf_chk, f"bert/encoder/layer_{layer}/output/LayerNorm/beta", encoder.out_norm.bias)
        load_tf_var(tf_chk, f"bert/encoder/layer_{layer}/output/LayerNorm/gamma", encoder.out_norm.weight)


def convert_tf_bert(tf_check: str, encoder: BertEncoder):
    load_bert_embeddings(tf_check, encoder.embeddings)
    for i, layer in enumerate(encoder.encoder_layers):
        load_bert_encoder(tf_check, i, layer)


def load_mlm_head(tf_check: str, head: BertMLMHead):
    with torch.no_grad():
        # Transform
        load_tf_var(tf_check, "cls/predictions/transform/LayerNorm/beta", head.layer_norm.bias)
        load_tf_var(tf_check, "cls/predictions/transform/LayerNorm/gamma", head.layer_norm.weight)
        load_tf_var(tf_check, "cls/predictions/transform/dense/bias", head.transform.bias)
        load_tf_var(tf_check, "cls/predictions/transform/dense/kernel", head.transform.weight,
                    processor=lambda x: np.transpose(x))

        # Bias
        load_tf_var(tf_check, "cls/predictions/output_bias", head.bias)


def load_transformers_embeddings(tf_embedding: BertEmbeddings, embeddings: BertEmbedding):
    with torch.no_grad():
        embeddings.word_embedding.weight.copy_(tf_embedding.word_embeddings.weight)
        embeddings.pos_embedding.weight.copy_(tf_embedding.position_embeddings.weight)
        embeddings.segment_embedding.weight.copy_(tf_embedding.token_type_embeddings.weight)


def load_transformers_encoders(tf_layers: ModuleList, layers: ModuleList[BertEncoderLayer]):
    with torch.no_grad():
        for tf_l, l in zip(tf_layers, layers):
            # Linear Weights
            l.attention.k_proj.weight.copy_(tf_l.attention.self.key.weight)
            l.attention.q_proj.weight.copy_(tf_l.attention.self.query.weight)
            l.attention.v_proj.weight.copy_(tf_l.attention.self.value.weight)
            l.attention.o_proj.weight.copy_(tf_l.attention.output.dense.weight)

            # Linear Bias
            l.attention.k_proj.bias.copy_(tf_l.attention.self.key.bias)
            l.attention.q_proj.bias.copy_(tf_l.attention.self.query.bias)
            l.attention.v_proj.bias.copy_(tf_l.attention.self.value.bias)
            l.attention.o_proj.bias.copy_(tf_l.attention.output.dense.bias)

            # Attention Norm
            l.attention.output_norm.weight.copy_(tf_l.attention.output.LayerNorm.weight)
            l.attention.output_norm.bias.copy_(tf_l.attention.output.LayerNorm.bias)

            # Intermediate Linear
            l.intermediate_proj.weight.copy_(tf_l.intermediate.dense.weight)
            l.intermediate_proj.bias.copy_(tf_l.intermediate.dense.bias)

            # Output Linear + Norm
            l.out_proj.weight.copy_(tf_l.output.dense.weight)
            l.out_proj.bias.copy_(tf_l.output.dense.bias)
            l.out_norm.weight.copy_(tf_l.output.LayerNorm.weight)
            l.out_norm.bias.copy_(tf_l.output.LayerNorm.bias)


def load_transformers_base_bert(tf_bert: BertModel, bert_base: BertEncoder):
    load_transformers_embeddings(tf_bert.embeddings, bert_base.embeddings)
    load_transformers_encoders(tf_bert.encoder.layer, bert_base.encoder_layers)
