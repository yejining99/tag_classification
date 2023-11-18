# tag_classification
This code provides an implementation of training a contrastive model using a Language Model (LM). The main objective is to retrieve tag keywords from given texts by training the model. The user can choose the bert model that want to use.

* The task was conducted as an industry-academic project at Hugraph.*

## Description
- main.py : This is the primary script for the entire workflow. It orchestrates data loading, model initialization, training, and evaluation processes.
- dataloader.py : Contains functions and classes related to loading and preprocessing data.
- model.py : Defines the architecture of the model.
- LLM.py : Contains utility functions (tokenizer and LLM pre-trained model) related to the Language Model (LLM).
- loss.py : Contains the loss and distance function used for training.
- evaluation.py : Contains various evaluation metrics used to gauge the performance of the model on validation or test data. Function like recall_at_k, precision_at_k, NDCG_at_k etc.

## Usage
Example :
If you want to train using the bert-multilingual model with Euclidean distance, you can use the following command
```ruby
python main.py --LLM_name bert-multilingual --distance euclidean_distance
```

Extend:
You can change the model between bert-multilingual, Kobert, and KR-Finbert.
Also you can change the distance between pairwise, cosine and euclidean.

## Arguments
- max_length: Maximum token length
- batch_size: Training batch size
- LLM_name: Specifies the name of the pre-trained language model to use. Default is 'bert-multilingual'.
- distance: Distance function to use. Default is pairwise_distance.
- epochs: Number of epochs for training.
- lr: Learning rate
- loss: Loss function to use
- topk: The value of k in Recall@k during evaluation.
- device: Specifies the device to run the model on, e.g., 'cuda:1' or 'cpu'.
