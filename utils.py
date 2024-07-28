# Author: Munagala Giridhar
# Utils used in the process of Essay scoring.
# Majorly towards data modelling

import torch
from torch.utils.data import Dataset
from typing import List, Dict
from transformers import AutoTokenizer
import re
import evaluate
import numpy as np

# Loading Accuracy Eval metric
accuracy = evaluate.load("accuracy")

def compute_accuracy_metric(eval_pred):
    """
        Computes Accuracy based on the Labels & Predictions
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def preprocess_text(text: str) -> str:
    """
        Removes special characters & lower cases text.
    """
    if text:
        return re.sub('[^a-z0-9,.\s]','', text.lower())
    return ''



class TextClassifDs(Dataset):
    """
        Text dataset for Classification.
        Outputs tokenized inforation.
        Args:
            text: List of texts
            labels: List of labels
            tokenizer: Transformers tokenizer that follows Autotokenizer structures
            transformations: TBD
        returns:
            Dictionary with keys
                input_ids : Tokenized ids
                attention_mask: attention mask of the text
                label: Labels provided
    """
    def __init__(self, text:List, labels:List, tokenizer:AutoTokenizer, transformation=None) -> Dict:
        super().__init__()
        self.text = text
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index) -> Dict:
        """
            Tokenizes with the given tokenizer & returns the data in a dictionary with the labels.
        """
        tokenized_text = self.tokenizer(self.text[index], return_tensors='pt', max_length=self.tokenizer.model_max_length, truncation=True)

        return {
            "input_ids" : tokenized_text['input_ids'].squeeze(),
            "attention_mask": tokenized_text['attention_mask'].squeeze(),
            "labels" : torch.tensor(self.labels[index],dtype=torch.long)
        }

