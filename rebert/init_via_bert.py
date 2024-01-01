import torch
from torch.nn import ModuleList
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings, RobertaModel, RobertaLMHead

from .model import ReBertEmbedding, ReBertModel, ReBertLMHead


def load_transformers_embeddings(tf_embedding: RobertaEmbeddings, embeddings: ReBertEmbedding):
    with torch.no_grad():
        embeddings.word_embedding.weight.copy_(
            tf_embedding.word_embeddings.weight + tf_embedding.token_type_embeddings.weight[0])


def load_transformers_encoders(tf_layers: ModuleList, layers: ModuleList):
    with torch.no_grad():
        for tf_l, l in zip(tf_layers, layers):
            # Linear Weights
            l.attention.self_attention.k_proj.weight.copy_(tf_l.attention.self.key.weight)
            l.attention.self_attention.q_proj.weight.copy_(tf_l.attention.self.query.weight)
            l.attention.self_attention.v_proj.weight.copy_(tf_l.attention.self.value.weight)
            l.attention.o_proj.weight.copy_(tf_l.attention.output.dense.weight)

            # Linear Bias
            l.attention.self_attention.k_proj.bias.copy_(tf_l.attention.self.key.bias)
            l.attention.self_attention.q_proj.bias.copy_(tf_l.attention.self.query.bias)
            l.attention.self_attention.v_proj.bias.copy_(tf_l.attention.self.value.bias)
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


def load_transformers_base_bert(tf_bert: RobertaModel, bert_base: ReBertModel):
    load_transformers_embeddings(tf_bert.embeddings, bert_base.embedding)
    load_transformers_encoders(tf_bert.encoder.layer, bert_base.encoder.encoder_layers)


def load_transformers_base_mlm(tf_mlm: RobertaLMHead, mlm: ReBertLMHead):
    with torch.no_grad():
        mlm.transform.weight.copy_(tf_mlm.dense.weight)
        mlm.transform.bias.copy_(tf_mlm.dense.bias)
        mlm.layer_norm.weight.copy_(tf_mlm.layer_norm.weight)
        mlm.layer_norm.bias.copy_(tf_mlm.layer_norm.bias)
        mlm.decoder.weight.copy_(tf_mlm.decoder.weight)
        mlm.decoder.bias.copy_(tf_mlm.decoder.bias)