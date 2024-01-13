from transformers import AutoConfig, AutoModelForSeq2SeqLM

from .rebert_model import ReBertConfig, ReBertForConditionalGeneration

AutoConfig.register("rebert", ReBertConfig)
AutoModelForSeq2SeqLM.register(ReBertConfig, ReBertForConditionalGeneration)