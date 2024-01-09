from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification, \
    AutoModelForSeq2SeqLM

from rebert.model import ReBertConfig, ReBertForConditionalGeneration

AutoConfig.register("rebert", ReBertConfig)
# AutoModel.register(ReBertConfig, ReBertModel)
AutoModelForSeq2SeqLM.register(ReBertConfig, ReBertForConditionalGeneration)
