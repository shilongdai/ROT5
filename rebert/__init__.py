from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification

from rebert.model import ReBertConfig, ReBertModel, ReBertForMaskedLM, ReBertForSequenceClassification

AutoConfig.register("rebert", ReBertConfig)
AutoModel.register(ReBertConfig, ReBertModel)
AutoModelForMaskedLM.register(ReBertConfig, ReBertForMaskedLM)
AutoModelForSequenceClassification.register(ReBertConfig, ReBertForSequenceClassification)