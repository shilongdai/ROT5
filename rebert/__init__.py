from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

from rebert.model import ReBertConfig, ReBertModel, ReBertForMaskedLM

AutoConfig.register("rebert", ReBertConfig)
AutoModel.register(ReBertConfig, ReBertModel)
AutoModelForMaskedLM.register(ReBertConfig, ReBertForMaskedLM)
