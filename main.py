import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import pickle
import os

from dataloader import CustomDataset
from loss import loss_and_distance
from LLM import get_LLM
from model import ContrastiveModel
from evaluation import recall_at_k, precision_at_k, NDCG_at_k, mean_reciprocal_rank_at_k, hit_rate_at_k, F1_at_k

import argparse

################## 0. Setting ##################

def parse_args():
    parser = argparse.ArgumentParser(description="Training a Contrastive Model")

    parser.add_argument("--max_length", default=512, type=int, help="Maximum token length")
    parser.add_argument("--batch_size", default=64, type=int, help="Training batch size")
    parser.add_argument("--use_lora", default=True, type=lambda x: (str(x).lower() == 'true'), help="Whether to use LoRA")
    parser.add_argument("--lora_layer", default=None, type=str, help="LoRA layer")
    parser.add_argument("--finetuning", default=False, type=lambda x: (str(x).lower() == 'true'), help="Fine-tune LLM parameters")
    parser.add_argument("--K", default=32, type=int, help="Rank K for adaptation")
    parser.add_argument("--epochs", default=2, type=int, help="Number of training epochs")
    parser.add_argument("--topk", default=10, type=int, help="Value for Recall@k")
    parser.add_argument("--LLM_name", default="bert-multilingual", type=str, help="Name of the LLM")
    parser.add_argument("--loss", default="contrastive", type=str, help="Loss function to use")
    parser.add_argument("--distance", default="pairwise_distance", type=str, help="Distance function to use")
    parser.add_argument("--device", default="cuda:1", type=str, help="Device to run the model on. E.g., 'cuda:1' or 'cpu'")
    parser.add_argument("--save_dir", default="saved_models", type=str, help="Directory to save the model")
    parser.add_argument("--result_dir", default="results", type=str, help="Directory to save the results")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # Use the parsed arguments
    max_length = args.max_length
    batch_size = args.batch_size
    use_lora = args.use_lora
    lora_layer = args.lora_layer
    finetuning = args.finetuning
    K = args.K
    epochs = args.epochs
    topk = args.topk
    LLM_name = args.LLM_name
    loss_name = args.loss
    distance_name = args.distance
    
    LLM, tokenizer = get_LLM(LLM_name)
    loss_function, distance_function = loss_and_distance(loss_name, distance_name)
    
    save_dir = args.save_dir
    result_dir = args.result_dir
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")


    ################## 1.Data Loading ##################
    df_temp = pd.read_csv('data/df_short.csv', encoding='utf-8-sig')
    df = pd.read_csv('data/total.csv', encoding='utf-8-sig')

    max_length_token = df_temp['body'].map(lambda x: len(x)).max()
    df['body'] = df['body'].map(lambda x: x[:max_length_token])

    df = df[:100] # 실험을 위해 cut

    # train, valid, test set으로 나누기
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # train set만 keyword 나눠주기
    train_df['index'] = train_df['index'].apply(lambda x: x.split(','))
    train_df['index'] = train_df['index'].apply(lambda x: [i.strip() for i in x])
    train_df = train_df.explode('index')

    print("The data is loaded")
    print("Train {}, valid {}, Test {}".format(train_df.shape, valid_df.shape, test_df.shape))

    # 나올 수 있는 모든 index 정의
    unique_list = df['index'].apply(lambda x: x.split(',')).explode().unique()
    unique_list = [x.strip() for x in unique_list]

    train_dataset = CustomDataset(train_df, tokenizer, unique_list, max_length, device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ################## 2. Training ##################
    print(f"Starting training for LLM_name: {LLM_name}, use_lora: {use_lora}, LoRA layer: {lora_layer}, K: {K}, finetuning: {finetuning}, epochs: {epochs}")

    # 모델 초기화
    model = ContrastiveModel(LLM, tokenizer, use_lora=use_lora, max_length=max_length, K=K, lora_layer=lora_layer, finetuning=finetuning)
    model.to(device)

    # recall을 기준으로 모델 고르기
    best_val_recall = float('inf')

    # Determine which parameters to optimize
    if use_lora:
        optimizer = optim.Adam([{"params": model.lora.parameters()}])
    else:
        optimizer = optim.Adam(model.parameters())

    total_train_loss_list = []
    total_val_loss_list = []
    total_recall_list = []
    total_precision_list = []
    total_ndcg_list = []

    for epoch in range(epochs):
        # Training loop
        model.train()
        total_loss = 0.0
        for tokenized_text, pos_keyword, neg_keyword in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            optimizer.zero_grad()

            tokenized_text = {k: v.to(device) for k, v in tokenized_text.items()}
            pos_keyword = {k: v.to(device) for k, v in pos_keyword.items()}
            neg_keyword = {k: v.to(device) for k, v in neg_keyword.items()}
            
            tokenized_text = {k: torch.cat([v, v]) for k, v in tokenized_text.items()}
            keyword = {k: torch.cat([pos_v, neg_v]) for k, pos_v, neg_v in zip(pos_keyword.keys(), pos_keyword.values(), neg_keyword.values())}

            text_embedding, keyword_embedding = model(tokenized_text, keyword)
            label = torch.cat([torch.zeros(len(keyword_embedding)//2), torch.ones(len(keyword_embedding)//2)]).to(device)
            loss = loss_function(text_embedding, keyword_embedding, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader)}")
        training_loss = total_loss/len(train_loader)
        total_train_loss_list.append(training_loss)

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        total_recall = 0.0
        total_precision = 0.0
        total_ndcg = 0.0
        total_mrr = 0.0
        total_hit = 0.0
        total_f1 = 0.0
        
        with torch.no_grad():
            for index, row in tqdm(valid_df.iterrows(), desc=f"Epoch {epoch+1} Validation", total=len(valid_df)):
                total_val_loss = 0.0
                tokenized_text = tokenizer(row['title'] + " " + row['body'], return_tensors="pt", truncation=True, padding="max_length", max_length=max_length).to(device)
                
                text_embedding, _ = model(tokenized_text)  # Assuming the model can process single instances and return the text embedding
                pos_keywords = [x.strip() for x in row['index'].split(',')]
                neg_keywords = list(np.random.choice(list(set(unique_list) - set(pos_keywords)), 100))
                
                all_keywords = pos_keywords + neg_keywords
                keyword_embeddings = []
                ### pos, neg embedding과 loss 구하기
                for keyword in pos_keywords:
                    tokenized_keyword = tokenizer(keyword, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length).to(device)
                    _, keyword_embedding = model(tokenized_text, tokenized_keyword)
                    keyword_embeddings.append(keyword_embedding)
                    loss = loss_function(text_embedding, keyword_embedding, 0)
                    total_val_loss += loss.item()
                    
                for keyword in neg_keywords:
                    tokenized_keyword = tokenizer(keyword, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length).to(device)
                    _, keyword_embedding = model(tokenized_text, tokenized_keyword)
                    keyword_embeddings.append(keyword_embedding)
                    loss = loss_function(text_embedding, keyword_embedding, 1)
                    total_val_loss += loss.item()
                    
                ### distance 구하기
                distance_list = []
                for keyword_embedding in keyword_embeddings:
                    distance = distance_function(text_embedding, keyword_embedding)
                    distance_list.append(distance.item())
                
                if distance_name in ['pairwise_distance', 'euclidean_distance']:
                    sorted_indices = np.argsort(np.array(distance_list))
                elif distance_name in ['cosine_similarity']:
                    sorted_indices = np.argsort(np.array(distance_list))
                    sorted_indices = sorted_indices[::-1] 
                ranked_keywords = [all_keywords[i] for i in sorted_indices][:topk]         
                
                ### recall, precision, ndcg 구하기
                recall = recall_at_k(ranked_keywords, pos_keywords, topk)
                precision = precision_at_k(ranked_keywords, pos_keywords, topk)
                ndcg = NDCG_at_k(ranked_keywords, pos_keywords, topk)
                #mrr = mean_reciprocal_rank_at_k(ranked_keywords, pos_keywords, topk)
                #hit = hit_rate_at_k(ranked_keywords, pos_keywords, topk)
                #f1 = F1_at_k(ranked_keywords, pos_keywords, topk)
                
                total_recall += recall
                total_precision += precision
                total_ndcg += ndcg
                #total_mrr += mrr
                #total_hit += hit
                #total_f1 += f1
                
            avg_val_loss = total_val_loss/len(valid_df)
            avg_val_recall = total_recall/len(valid_df)
            avg_val_precision = total_precision/len(valid_df)
            avg_val_ndcg = total_ndcg/len(valid_df)
            #avg_val_mrr = total_mrr/len(valid_df)
            #avg_val_hit = total_hit/len(valid_df)
            #avg_val_f1 = total_f1/len(valid_df)
            
            print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}, Recall@{topk}: {avg_val_recall}, Precision@{topk}: {avg_val_precision}, NDCG@{topk}: {avg_val_ndcg}")

            total_val_loss_list.append(avg_val_loss)
            total_recall_list.append(avg_val_recall)
            total_precision_list.append(avg_val_precision)
            total_ndcg_list.append(avg_val_ndcg)
        
            if avg_val_recall < avg_val_recall:
                best_val_recall = avg_val_recall
                model_path = os.path.join(save_dir, f'best_model_lora_{lora_layer}_K_{K}.pth')
                torch.save(model.state_dict(), model_path)
                print(f"Model saved at {model_path}")


        # 실험 결과 저장
        experiment_results = {
            'lora_layer': lora_layer,
            'K': K,
            'training_loss': total_train_loss_list,
            'validation_loss': total_val_loss_list,
            'validation_recall': total_recall_list,
            'validation_precision': total_precision_list,
            'validation_ndcg': total_ndcg_list}


        with open(result_dir+'/use_lora: {use_lora}, finetuning: {finetuning}.pkl'.format(use_lora=use_lora, finetuning=finetuning), 'wb') as f:
                pickle.dump(experiment_results, f)

        print(f"Experiment results saved..")

