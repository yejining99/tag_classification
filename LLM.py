from transformers import BertModel, BertTokenizer

def get_LLM(model_name):
    if model_name == 'bert-multilingual':
        LLM = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        
    return LLM, tokenizer
    
