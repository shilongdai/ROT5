import torch
from torch.nn import ModuleList
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings, RobertaModel, RobertaLMHead

from .model import ReBertEmbedding, ReBertModel, ReBertLMHead, ReBertConfig


def load_transformers_embeddings(tf_embedding: RobertaEmbeddings, embeddings: ReBertEmbedding):
    with torch.no_grad():
        embeddings.word_embedding.weight.copy_(
            tf_embedding.word_embeddings.weight + tf_embedding.token_type_embeddings.weight[0])


def average_attention_head_weights(src_weight: torch.Tensor, size_per_head, num_heads, kv_head_group_size):
    if kv_head_group_size == 1:
        return src_weight

    input_size = size_per_head * num_heads
    kv_group = num_heads // kv_head_group_size
    src_weight = src_weight.reshape(num_heads, size_per_head, input_size)
    output_weight = torch.zeros((kv_group, size_per_head, input_size))

    for i in range(kv_group):
        start = i * kv_head_group_size
        end = start + kv_head_group_size
        head_buffer = src_weight[start:end, :, :]
        output_weight[i, :, :] = torch.mean(head_buffer, dim=0, keepdim=False)

    return output_weight.reshape(-1, input_size)


def average_attention_head_biases(src_weight: torch.Tensor, size_per_head, num_heads, kv_head_group_size):
    if kv_head_group_size == 1:
        return src_weight

    kv_group = num_heads // kv_head_group_size
    src_weight = src_weight.reshape(num_heads, size_per_head)
    output_weight = torch.zeros(kv_group, size_per_head)

    for i in range(kv_group):
        start = i * kv_head_group_size
        end = start + kv_head_group_size
        head_buffer = src_weight[start:end, :]
        output_weight[i, :] = torch.mean(head_buffer, dim=0, keepdim=False)

    return output_weight.reshape(kv_group * size_per_head)


def load_grouped_attention(tf_attention, rebert_attention, config: ReBertConfig):
    size_per_head = config.hidden_size // config.num_attention_heads
    kv_head_group_size = config.num_attention_heads // config.num_key_value_heads

    # Linear Weights
    self = rebert_attention.self_attention
    src_self = tf_attention.self
    self.k_proj.weight.copy_(
        average_attention_head_weights(src_self.key.weight, size_per_head, config.num_attention_heads,
                                       kv_head_group_size))
    self.q_proj.weight.copy_(src_self.query.weight)
    self.v_proj.weight.copy_(
        average_attention_head_weights(src_self.value.weight, size_per_head, config.num_attention_heads,
                                       kv_head_group_size))
    rebert_attention.o_proj.weight.copy_(tf_attention.output.dense.weight)

    # Linear Bias
    self.k_proj.bias.copy_(
        average_attention_head_biases(src_self.key.bias, size_per_head, config.num_attention_heads, kv_head_group_size))
    self.q_proj.bias.copy_(src_self.query.bias)
    self.v_proj.bias.copy_(average_attention_head_biases(src_self.value.bias, size_per_head, config.num_attention_heads,
                                                         kv_head_group_size))
    rebert_attention.o_proj.bias.copy_(tf_attention.output.dense.bias)


def load_transformers_encoders(tf_layers: ModuleList, layers: ModuleList, config: ReBertConfig):
    with torch.no_grad():
        for tf_l, l in zip(tf_layers, layers):
            # Attention
            load_grouped_attention(tf_l.attention, l.attention, config)

            # Intermediate Linear
            l.intermediate_proj.weight.copy_(tf_l.intermediate.dense.weight)
            l.intermediate_proj.bias.copy_(tf_l.intermediate.dense.bias)

            # Output Linear
            l.out_proj.weight.copy_(tf_l.output.dense.weight)
            l.out_proj.bias.copy_(tf_l.output.dense.bias)


def load_transformers_base_bert(tf_bert: RobertaModel, bert_base: ReBertModel, config: ReBertConfig):
    load_transformers_embeddings(tf_bert.embeddings, bert_base.embedding)
    load_transformers_encoders(tf_bert.encoder.layer, bert_base.encoder.encoder_layers, config)


def load_transformers_base_mlm(tf_mlm: RobertaLMHead, mlm: ReBertLMHead):
    with torch.no_grad():
        mlm.transform.weight.copy_(tf_mlm.dense.weight)
        mlm.transform.bias.copy_(tf_mlm.dense.bias)
        mlm.decoder.weight.copy_(tf_mlm.decoder.weight)
