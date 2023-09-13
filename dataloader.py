import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, unique_list, max_length, device):
        self.text = dataframe['title'] + " " + dataframe['body']
        self.index = dataframe['index']
        self.len = len(self.text)
        self.tokenizer = tokenizer
        self.unique_list = unique_list
        self.device = device
        self.max_length = max_length

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        text = self.text.iloc[idx]
        keyword = self.index.iloc[idx]
        negative_sample = self.generate_negative_samples(keyword, self.unique_list)
        device = self.device
        max_length = self.max_length

         # Tokenization
        tokenized_text = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length).to(device)
        tokenized_pos_keyword = self.tokenizer(keyword, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length).to(device)
        tokenized_neg_keyword = self.tokenizer(negative_sample, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length).to(device)

        # Squeeze the tensors to remove the unnecessary dimension
        for key in tokenized_text:
            tokenized_text[key] = tokenized_text[key].squeeze().to(device)
        for key in tokenized_pos_keyword:
            tokenized_pos_keyword[key] = tokenized_pos_keyword[key].squeeze().to(device)
        for key in tokenized_neg_keyword:
            tokenized_neg_keyword[key] = tokenized_neg_keyword[key].squeeze().to(device)

        return tokenized_text, tokenized_pos_keyword, tokenized_neg_keyword

    def generate_negative_samples(self, keyword, unique_list):
        return np.random.choice(list(set(unique_list) - set(keyword)), 1)[0]
