# tag_classification
This code provides an implementation of training a contrastive model using a Language Model (LM) with a LoRA (Local Representational Adaptation) layer. The main objective is to retrieve tag keywords from given texts by training the model. The user can choose to employ the LoRA technique to enhance performance.

## Description
- main.py : This is the primary script for the entire workflow. It orchestrates data loading, model initialization, training, and evaluation processes.
- dataloader.py : Contains functions and classes related to loading and preprocessing data.
- model.py : Defines the architecture of the model.
- LLM.py : Contains utility functions (tokenizer and LLM pre-trained model) related to the Language Model (LLM).
- loss.py : Contains the loss and distance function used for training.
- evaluation.py : Contains various evaluation metrics used to gauge the performance of the model on validation or test data. Function like recall_at_k, precision_at_k, NDCG_at_k etc.

## Usage
1. Using LoRA without Fine-tuning:
If you want to train the model with LoRA (Layer-wise Relevance Analysis) but without fine-tuning the base LLM (Language Model), you can use the following command
```ruby
python main.py --use_lora True --finetuning False --batch_size 64
```

3. Using Fine-tuning without LoRA:
If you want to fine-tune the LLM on your dataset without using LoRA, execute the following command
```ruby
python main.py --use_lora False --finetuning True --batch_size 8
```

## Arguments
- LLM_name: Specifies the name of the pre-trained language model to use. Default is 'bert-multilingual'.
- use_lora: Whether to use the LoRA layer. Set to True or False.
- lora_layer: The depth of the LoRA layer. Use None for the last layer.
- K: Specifies the rank for rank-adaptation in LoRA.
- finetuning: Whether to fine-tune the LLM's parameters.
- epochs: Number of epochs for training.
- topk: The value of k in Recall@k during evaluation.
- device: Specifies the device to run the model on, e.g., 'cuda:1' or 'cpu'.
