import pandas as pd
import torch
from LLM import get_LLM
from model import ContrastiveModel
from loss import loss_and_distance
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="case study for tag prediction")
    parser.add_argument("--model_dir", default="./data/best_model_KR-Finbert_4_1e-05_16_pairwise_distance.pth", type=str, help="Directory of saved model")
    parser.add_argument("--news_dir", default="./data/news.txt", type=str, help="news content")
    parser.add_argument("--distance", default="pairwise_distance", type=str, help="Distance function to use")
    parser.add_argument("--k", default=3, type=int, help="Top k tags")
    parser.add_argument("--device", default="cuda:0", type=str, help="cuda device")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # 1. Load the data and model
    args = parse_args()
    model_dir = args.model_dir
    news_dir = args.news_dir
    device = args.device
    
    loss_function, distance_function = loss_and_distance('contrastive', args.distance)
    if 'bert-multilingual' in args.model_dir:
        LLM_name = 'bert-multilingual'
    elif 'KR-Finbert' in args.model_dir:
        LLM_name = 'KR-Finbert'
    elif 'Kobert' in args.model_dir:
        LLM_name = 'Kobert'
    LLM, tokenizer = get_LLM(LLM_name)
    model = ContrastiveModel(LLM, tokenizer, use_lora=False, finetuning=True)
    model.load_state_dict(torch.load(model_dir, map_location=device))
    print("Model loaded")
    
    # 2. set all the tag that we want to predict
    df = pd.read_csv('data/total.csv', encoding='utf-8-sig')
    unique_list = df['index'].apply(lambda x: x.split(',')).explode().unique()
    unique_list = [x.strip() for x in unique_list]
    
    # 3. get the embedding of all the tags
    tag_embedding = {}
    for tag in unique_list:
        tokenized_tag = tokenizer(tag, return_tensors='pt', truncation=True, max_length=512)
        tag_embedding[tag] = model.embedding(tokenized_tag)
    print("Tag embedding loaded")
    
    # 4. get the embedding of all the text
    news_content = open(news_dir, 'r', encoding='utf-8-sig').read()
    tokenized_news = tokenizer(news_content, return_tensors='pt', truncation=True, max_length=512)
    news_embedding = model.embedding(tokenized_news)
    print("News embedding loaded")
    
    # 5. calculate the distance between the text and all the tags
    distance = {}
    for tag in unique_list:
        distance[tag] = distance_function(news_embedding, tag_embedding[tag]).item()
    if args.distance in ['pairwise_distance', 'euclidean_distance']:
        sorted_tag = sorted(distance.items(), key=lambda x: x[1], reverse=False)
    elif args.distance in ['cosine_distance']:
        sorted_tag = sorted(distance.items(), key=lambda x: x[1], reverse=True)
    
    ranked_tag = sorted_tag[:args.k]
    
    print("The top {} tags are: ".format(args.k))
    for tag in ranked_tag:
        print(tag[0])
        