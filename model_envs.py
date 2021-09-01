from transformers import ElectraConfig, ElectraModel, ElectraTokenizer
from transformers import LongformerConfig, LongformerModel, LongformerTokenizer

############################################################
# Model Related Global Varialbes
############################################################
MODEL_CLASSES = {
    'electra': (ElectraConfig, ElectraModel, ElectraTokenizer),
    'longformer': (LongformerConfig, LongformerModel, LongformerTokenizer)
}