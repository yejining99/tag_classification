import torch
import torch.nn as nn
import loralib as lora

class ContrastiveModel(nn.Module):
    def __init__(self, LLM, tokenizer, use_lora=True, max_length=512, K=None, lora_layer=None, finetuning=False):  # Default value for lora_layer set to None
        super(ContrastiveModel, self).__init__()
        self.LLM = LLM
        if finetuning == False: # If we're not finetuning, freeze the weights
            for param in self.LLM.parameters():
                param.requires_grad = False
    
        else: # If we're finetuning, freeze the first 3 layers
            for i, layer in enumerate(self.LLM.encoder.layer):
                if i < 20:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    break

        self.tokenizer = tokenizer 
        self.max_length = max_length
        self.use_lora = use_lora
        self.K = K
        self.lora_layer = lora_layer if lora_layer is not None else -1  # If None, use last hidden state
        d_model = self.LLM.config.hidden_size  # BERT hidden size

        # If we're using LoRA
        if self.use_lora:
            self.lora = lora.Linear(d_model, d_model, r=self.K)


    def forward(self, text_tokens, keyword_tokens=None):
        outputs = self.LLM(input_ids=text_tokens["input_ids"], attention_mask=text_tokens["attention_mask"])
        text_embedding = outputs.hidden_states[self.lora_layer].mean(dim=1)
        
        if keyword_tokens is None:
            return text_embedding, None
        
        outputs = self.LLM(input_ids=keyword_tokens["input_ids"], attention_mask=keyword_tokens["attention_mask"])
        keyword_embedding = outputs.hidden_states[self.lora_layer].mean(dim=1)

        if self.use_lora:
            text_embedding = self.lora(text_embedding)
            keyword_embedding = self.lora(keyword_embedding)

        return text_embedding, keyword_embedding
    
    def summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        print(f"Total Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")
        print(f"Non-trainable Parameters: {non_trainable_params}")
        