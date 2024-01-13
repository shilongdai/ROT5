from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification, \
    AutoModelForSeq2SeqLM

from .rot5_model import ROT5Config, ROT5ForConditionalGeneration

AutoConfig.register("rebert", ROT5Config)
AutoModel.register(ROT5Config, ROT5ForConditionalGeneration)
AutoModelForSeq2SeqLM.register(ROT5Config, ROT5ForConditionalGeneration)
