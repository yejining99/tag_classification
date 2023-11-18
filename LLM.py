from transformers import BertModel, BertTokenizer,  AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_LLM(model_name):
    if model_name == 'bert-multilingual':
        LLM = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # elif model_name == 'KoAlpaca-Polyglot-5.8B':
    #     LLM = AutoModelForCausalLM.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B", output_hidden_states=True)
    #     tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")
        
    elif model_name == 'Kobert':
        LLM = AutoModel.from_pretrained("monologg/kobert", output_hidden_states=True)  
        tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")

    elif model_name == 'KR-Finbert':
        LLM = AutoModel.from_pretrained("snunlp/KR-FinBert-SC", output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")
        
    return LLM, tokenizer
    
