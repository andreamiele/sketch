#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#  ==============================================================================
# File: transformers.py
# Created: Tuesday, 12th December 2023 2:12:34 pm
# Author: Andrea Miele (andrea.miele.pro@gmail.com, https://www.andreamiele.fr)
# GitHub: https://www.github.com/andreamiele
# -----
# Last Modified: Tuesday, 12th December 2023 4:09:32 pm
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
from tokenizers.sketch_tokenizer import Tokenizer




def get_bounds(data, factor=1):
    """Return bounds of data."""
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)



class SketchDataloader(Dataset):
    def __init__(self, data, k=1000, max_seq_len=200):
        self.data = data
        self.k = k
        self.max_seq_len = max_seq_len
        self.tokenizer = Tokenizer('../token_dict.pkl',
                                             max_seq_len=0)
        self.train_data = self.load_data('train')
        self.valid_data = self.load_data('valid')
        self.test_data = self.load_data('test')
        self.limit = 1000

    def load_data(self, data_type):
        loaded_data = []

        # Construct the path to the specific data type
        data_path = os.path.join(self.data_dir, f'/content/quickdraw/The Eiffel Tower.npz')

        if os.path.exists(data_path):
            data = np.load(data_path)
            # Access the sketch data corresponding to the key (e.g., 'train', 'valid', 'test')
            sketch = data[data_type]
            loaded_data.append(sketch)

        return loaded_data
    def create_tokenizer(self, data, k):
        # Flatten the data and sample for K-means
        all_points = np.concatenate([sketch[:, :2] for sketch in data])
        sample_indices = np.random.choice(len(all_points), size=100000, replace=False)
        sample = all_points[sample_indices]

        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=0).fit(sample)
        return kmeans

    def preprocess(self, data):
        preprocessed = []
        for sketch in data:        
            # removes large gaps from the data
            sketch = np.minimum(sketch, self.limit)
            sketch = np.maximum(sketch, -self.limit)
            sketch = np.array(sketch, dtype=np.float32)
            
            min_x, max_x, min_y, max_y = get_bounds(sketch)
            max_dim = max([max_x - min_x, max_y - min_y, 1])
            sketch[:, :2] /= max_dim
            
            sketch = self.tokenizer.encode(sketch)
            if len(sketch) > self.max_seq_len:
                sketch = sketch[:self.max_seq_len]
            sketch = self._cap_pad_and_convert_sketch(sketch)
            sketch = np.squeeze(sketch)
            preprocessed.append(sketch)
        return np.array(preprocessed)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sketch = self.data[idx]
        tokenized_sketch = self.tokenize_sketch(sketch)
        return tokenized_sketch
    
    def _cap_pad_and_convert_sketch(self, sketch):
        desired_length = self.max_seq_len
        skt_len = len(sketch)

        
        converted_sketch = np.ones((desired_length, 1), dtype=int) * self.tokenizer.PAD
        converted_sketch[:skt_len, 0] = sketch
        return converted_sketch