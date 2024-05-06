import pandas as pd
from torch.utils.data import Dataset
import torch
import os
import random
import numpy as np
from torch import nn
from typing import Dict, Optional, Tuple, List
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, random_split
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_score, f1_score, recall_score, classification_report

def load_and_split(train_path, test_path):
    train_dev_path = train_path
    test_path = test_path
    
    train_dev_df = pd.read_csv(train_dev_path, sep='\t')
    test_df = pd.read_csv(test_path, sep='\t')

    train_dev_questions = list(train_dev_df["question"].unique())
    test_questions = list(test_df["question"].unique())
    num_train_dev_questions = len(train_dev_questions)
    random.shuffle(train_dev_questions)
    
    train_ratio = 0.9
    
    num_train_questions = int(num_train_dev_questions * train_ratio)
    train_questions = set(train_dev_questions[:num_train_questions])
    dev_questions = set(train_dev_questions[num_train_questions:])
    
    print(f"Questions: train - {len(train_questions)}, dev - {len(dev_questions)}, test - {len(test_questions)}")
    
    train_df = train_dev_df[train_dev_df["question"].isin(train_questions)]
    dev_df = train_dev_df[train_dev_df["question"].isin(dev_questions)]
    
    print(f"Train: {train_df.shape}")
    print(f"Dev: {dev_df.shape}")
    print(f"Test: {test_df.shape}")

    train_df["label"] = train_df["correct"].astype(np.float32)
    dev_df["label"] = dev_df["correct"].astype(np.float32)
    test_df["label"] = np.zeros(shape=test_df.shape[0], dtype=np.float32)

    train_df["graph"] = train_df["graph"].apply(eval)
    dev_df["graph"] = dev_df["graph"].apply(eval)
    test_df["graph"] = test_df["graph"].apply(eval)

    return train_df, dev_df, test_df