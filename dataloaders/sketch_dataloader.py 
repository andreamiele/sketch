#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: transformers.py
# Created: Tuesday, 12th December 2023 2:12:34 pm
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# GitHub: https://www.github.com/andreamiele
# -----
# Last Modified: Tuesday, 12th December 2023 2:18:09 pm
# Modified By: Andrea Miele (andrea.miele.pro@gmail.com)
# -----
# 
# -----
# Copyright (c) 2023 Your Company
# 
#  ==============================================================================
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader

class SketchDataloader(Dataset):
    def __init__(self, data, k=1000, max_seq_len=200):
        self.data = data
        self.k = k
        self.max_seq_len = max_seq_len
        self.tokenizer = self.create_tokenizer(data, k)

    def create_tokenizer(self, data, k):
        # Flatten the data and sample for K-means
        all_points = np.concatenate([sketch[:, :2] for sketch in data])
        sample_indices = np.random.choice(len(all_points), size=100000, replace=False)
        sample = all_points[sample_indices]

        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=0).fit(sample)
        return kmeans

    def tokenize_sketch(self, sketch):
        tokens = []
        for point in sketch:
            delta_x, delta_y, pen_state = point
            cluster_id = self.tokenizer.predict([[delta_x, delta_y]])[0]
            tokens.append(cluster_id)

            if pen_state == 1:  # Pen lift
                tokens.append(self.k + 1)  # SEP token

        # Add SOS and EOS tokens
        tokens = [self.k + 2] + tokens + [self.k + 3]  # SOS and EOS

        # Padding
        if len(tokens) < self.max_seq_len:
            tokens.extend([self.k + 4] * (self.max_seq_len - len(tokens)))  # PAD token
        else:
            tokens = tokens[:self.max_seq_len]

        return np.array(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sketch = self.data[idx]
        tokenized_sketch = self.tokenize_sketch(sketch)
        return tokenized_sketch