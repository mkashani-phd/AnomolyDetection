import sys, pymongo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import sklearn.preprocessing 

import src.VAE_LSTM_CNN  as VAE_LSTM_CNN
import src.IQ as IQ


onbody = pd.read_pickle('dataset/onBody.pkl')
onbody_val = pd.read_pickle('dataset/onBody_Val.pkl')
anomoly = pd.read_pickle('dataset/offBody.pkl')


class RFSignalTripletDataset(Dataset):
    def __init__(self, normal_df, anomaly_df):
        self.normal_samples = normal_df
        self.anomaly_samples = anomaly_df

        self.class_labels = normal_df['dvc'].unique()

        self.data_by_class = {}
        for class_label in self.class_labels:
            class_samples = normal_df[normal_df['dvc'] == class_label]
            self.data_by_class[class_label] = np.array(class_samples['freq_dev'])

        self.anchor_indices = []
        for class_label, samples in self.data_by_class.items():
            n = len(samples)
            self.anchor_indices.extend([(class_label, i) for i in range(n)])

        # # Split the normal samples into two halves for anchors and positives
        # x = train_test_split(self.normal_samples, test_size=0.5, random_state=42)
        # self.anchor_samples =  x[0].reset_index(drop=True)
        # self.positive_samples = x[1].reset_index(drop=True)
        
    def __len__(self):
        # The dataset length will be the number of normal samples divided by 2, 
        # since we're using half for anchors and half for positives
        return len(self.anchor_indices)
        # return len(self.anchor_samples)

    def __getitem__(self, idx):
        class_label, anchor_idx = self.anchor_indices[idx]
        n = len(self.data_by_class[class_label]) 
        anchor = self.data_by_class[class_label][anchor_idx]
        positive_idx = (anchor_idx + np.random.randint(1, n)) % n 
        positive = self.data_by_class[class_label][positive_idx]
        

        # choose the other class_labels randomly
        other_class_label = class_label
        while other_class_label == class_label:
            other_class_label = self.class_labels[np.random.randint(len(self.class_labels))]
        
        # Randomly select a negative sample from the other class
        negative1 = self.data_by_class[other_class_label][np.random.randint(len(self.data_by_class[other_class_label]))]
        
        # Randomly select a negative sample from the anomaly samples
        negative2 = self.anomaly_samples[np.random.randint(len(self.anomaly_samples))]

        #randomly select the negative between negative1 and negative2
        negative = negative1 if np.random.random() > 0.5 else negative2

        # negative = negative2

        # anchor = self.anchor_samples.iloc[idx]['freq_dev']
        # positive = self.positive_samples.iloc[idx]['freq_dev']
        # negative = self.anomaly_samples[np.random.randint(len(self.anomaly_samples))]


        # Convert to PyTorch tensors
        anchor = torch.tensor(anchor, dtype=torch.float).float().unsqueeze(0)
        positive = torch.tensor(positive, dtype=torch.float).float().unsqueeze(0)
        negative = torch.tensor(negative, dtype=torch.float).unsqueeze(0)
        
        return anchor, positive, negative