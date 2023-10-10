from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_LLM(model_name):
    if model_name == 'bert-multilingual':
        LLM = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    elif model_name == 'KoAlpaca-Polyglot-5.8B':
        LLM = AutoModelForCausalLM.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B", output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")
    return LLM, tokenizer
    
