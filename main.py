import numpy as np
import os
import sys
import csv
import numpy as np
import json
import pickle  # python3.x
import time
from datetime import timedelta, datetime
import subprocess
import struct
import errno
from pprint import pprint
import glob
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
import math
import json

print("Script started")
print(time.time)
def check_ndarray_range(ndarray):
    min_val = np.min(ndarray)
    max_val = np.max(ndarray)
    return min_val, max_val

def load_sketch_data(npz_file_path, mode='train',nb=4):
    """
    Load sketches from a .npz file.
    :param npz_file_path: Path to the .npz file containing sketch data.
    :return: Numpy array of sketches.
    """

    data = np.load(npz_file_path, encoding='latin1', allow_pickle=True)
    if mode=='train':
      total_samples = len(data[mode])
      quarter_samples = total_samples // nb  # Integer division to get a quarter of the total

      # Take the first quarter of the samples
      sketches = data[mode][:quarter_samples]  # Assuming the .npz file has a 'train' key for training sketches
    else:
      sketches = data[mode]
    return sketches

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

def normalize_sketch(sketch):
    # removes large gaps from the data
    sketch = np.minimum(sketch, 1000)
    sketch = np.maximum(sketch, -1000)

    # get bounds of sketch and use them to normalise
    min_x, max_x, min_y, max_y = get_bounds(sketch)
    max_dim = max([max_x - min_x, max_y - min_y, 1])
    sketch = sketch.astype(np.float32)
    sketch[:, :2] /= max_dim

    return sketch
# Function to check the range of values within a tensor
def check_tensor_range(tensor):
    min_val = torch.min(tensor).item()
    max_val = torch.max(tensor).item()
    return min_val, max_val

def _cap_pad_and_convert_sketch(sketch, tokenizer):
        desired_length = 200
        skt_len = len(sketch)
        converted_sketch = np.ones((desired_length, 1), dtype=int) * tokenizer.PAD
        converted_sketch[:skt_len, 0] = sketch
        return converted_sketch
    
def preprocess_sketches(sketches, tokenizer):
    """
    Preprocess sketches by normalizing their coordinates and ensuring consistent data types.
    :param sketches: List of sketches to preprocess.
    :return: List of preprocessed sketches.
    """
    processed_sketches = []
    for sketch in sketches:
        # Normalize the sketch here as per your requirements
        # Convert to 'double' if your clustering model expects that
        processed_sketch = normalize_sketch(sketch)
        processed_sketches.append(processed_sketch)
    return processed_sketches  # Return as a list

def lines_to_strokes(lines, omit_first_point=True):
    """
    Convert polyline format to stroke-3 format.
    lines: list of strokes, each stroke has format Nx2
    """
    strokes = []
    for line in lines:
        linelen = len(line)
        for i in range(linelen):
            eos = 0 if i < linelen - 1 else 1
            strokes.append([line[i][0], line[i][1], eos])
    strokes = np.array(strokes)
    strokes[1:, 0:2] -= strokes[:-1, 0:2]
    return strokes[1:, :] if omit_first_point else strokes

def strokes_to_lines(strokes, scale=1.0, start_from_origin=False):
    """
    convert strokes3 to polyline format ie. absolute x-y coordinates
    note: the sketch can be negative
    :param strokes: stroke3, Nx3
    :param scale: scale factor applied on stroke3
    :param start_from_origin: sketch starts from [0,0] if True
    :return: list of strokes, each stroke has format Nx2
    """
    x = 0
    y = 0
    lines = []
    line = [[0, 0]] if start_from_origin else []
    for i in range(len(strokes)):
        x_, y_ = strokes[i, :2] * scale
        x += x_
        y += y_
        line.append([x, y])
        if strokes[i, 2] == 1:
            line_array = np.array(line) + np.zeros((1, 2), dtype=np.uint8)
            lines.append(line_array)
            line = []
    if lines == []:
        line_array = np.array(line) + np.zeros((1, 2), dtype=np.uint8)
        lines.append(line_array)
    return lines

def load_pickle(path):
    """
    load a pickled object
    :param path: .pkl path
    :return: the pickled object
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


class Tokenizer(object):
    """
    tokenize sketches in stroke3 using clustering
    """

    def __init__(self, dict_path, max_seq_len=0):
        """
        initialize dictionary (a sklearn cluster object)
        :param dict_path: path to pickle file
        :param max_seq_len: 0 if variable length sketch
        """
        self.max_seq_len = 200
        self.dict = load_pickle(dict_path)
        self.PAD = 0
        self.SEP = self.dict.n_clusters +1  # whole dictionary needs to be shifted by 1
        self.SOS = self.dict.n_clusters + 2
        self.EOS = self.dict.n_clusters +3
        self.VOCAB_SIZE = self.dict.n_clusters +4

    def encode(self, stroke3, seq_len=0):
        """
        encode stroke3 into tokens
        :param stroke3: stroke3 data as numpy array (nx3)
        :param seq_len: if positive, the output is padded with PAD
        :return: sequence of integers as list
        """
        stroke3 += np.zeros((1, 3))
        stroke3 = stroke3.astype(np.float32)
        out = self.dict.predict(stroke3[:, :2])

        # shift by 1 due to PAD token
        out = out + 1
        out = list(out)
        # insert SEP token

        positions = np.where(stroke3[:, 2] == 1)[0]
        offset = 1
        for i in positions:
            out.insert(i + offset, self.SEP)
            offset += 1
        # insert SOS and EOS
        out = [self.SOS] + out + [self.EOS]
        if self.max_seq_len:  # pad
            npad = self.max_seq_len - len(out)
            if npad > 0:
                out += [self.PAD] * npad
            else:
                out = out[:self.max_seq_len]
                out[-2:] = [self.SEP, self.EOS]
        if len(out) < seq_len:
            out += [self.PAD] * (seq_len-len(out))
        return np.array(out)

    def decode(self, seqs):
        if len(seqs) > 0 and isinstance(seqs[0], (list, tuple, np.ndarray)):
            return self.decode_list(seqs)
        else:
            return self.decode_single(seqs)

    def decode_single(self, seq):
        """
        decode a sequence of token id to stroke3
        :param seq: list of integer
        :return: stroke3 array (nx3)
        """
        cluster_ids = []
        pen_states = []
        for i in seq:
            if i not in [self.SOS, self.EOS, self.SEP, self.PAD]:
                cluster_ids.append(i)
                pen_states.append(0)
            elif i == self.SEP and len(pen_states) > 0:
                pen_states[-1] = 1
            elif i == self.EOS:
                break
        if len(cluster_ids) > 0:
            cluster_ids = np.array(cluster_ids)
            cluster_ids = cluster_ids - 1
            dxy = self.dict.cluster_centers_[cluster_ids]
            out = np.c_[dxy, np.array(pen_states)]
            return np.array(out)
        else:
            return np.zeros((1, 3), dtype=np.float32)  #empty sketch

    def decode_list(self, sketches):
        decoded = []
        for s in sketches:
            decoded.append(self.decode_single(np.squeeze(s)))
        return decoded
    
dict_path = "/home/ml4science2023/GAN/sketch/token_dict.pkl"

sketch=load_sketch_data("/home/ml4science2023/GAN/sketch/quickdraw/The Eiffel Tower.npz", 'train')
sketch2=load_sketch_data("/home/ml4science2023/GAN/sketch/quickdraw/church.npz", 'train')
sketch3=load_sketch_data("/home/ml4science2023/GAN/sketch/quickdraw/skull.npz", 'train')
sketch4=load_sketch_data("/home/ml4science2023/GAN/sketch/quickdraw/sun.npz", 'train')
result1 = np.concatenate((sketch, sketch2), axis=0)
result2 = np.concatenate((result1, sketch3), axis=0)
sketches = np.concatenate((result2, sketch3), axis=0)


tokenizer = Tokenizer(dict_path)
preprocessed_sketches = preprocess_sketches(sketches,tokenizer)


# Tokenize using your Tokenizer class
tokenized_sketches = [tokenizer.encode(sketch) for sketch in preprocessed_sketches]
tokenized_sketches = [_cap_pad_and_convert_sketch(sketch, tokenizer) for sketch in tokenized_sketches]

train_sketches, test_sketches = train_test_split(
    tokenized_sketches, test_size=0.2, random_state=42)

# For reconstruction, the input features and labels are the same
train_labels = train_sketches
test_labels = test_sketches

class SketchDataset(Dataset):
    def __init__(self, sketches, max_length=0, padding_value=0):
        # Convert sketches to padded tensors
        self.sketches = sketches
    def __len__(self):
        return len(self.sketches)

    def __getitem__(self, idx):
        sketch = self.sketches[idx]
        return sketch, sketch
    
dataset = SketchDataset(train_sketches, train_labels)
dataset_test = SketchDataset(test_sketches, test_labels)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=True)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.size(1) should match the sequence length dimension
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SketchEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_length):
        super(SketchEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

    def forward(self, sketch_tokens):
        # Convert sketch_tokens to long tensor if not already
        sketch_tokens = sketch_tokens.long()  # Ensure input is of type Long
        token_embeddings = self.token_embedding(sketch_tokens)  # (batch_size, seq_length, d_model)
        return self.positional_encoding(token_embeddings)


class SelfAttentionBottleneck(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttentionBottleneck, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)

    def forward(self, src):
        # Assuming src is of shape [sequence_length, batch_size, embedding_dimension]
        attn_output, _ = self.self_attn(src, src, src)
        return attn_output


class Sketchformer(nn.Module):
    def __init__(self, config):
        super(Sketchformer, self).__init__()
        self.config = config
        self.vocab_size = 1004  # Adjust based on your tokenizer
        self.d_model = config["d_model"]
        self.nhead = config["nhead"]
        self.dim_feedforward = config["dim_feedforward"]
        self.dropout = config["dropout"]

        # Embedding layer
        self.sketch_embedding = nn.Embedding(self.vocab_size, self.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)

        # Transformer encoder and decoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.nhead,
            dim_feedforward=self.dim_feedforward, dropout=self.dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config["num_layers"])

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=self.nhead,
            dim_feedforward=self.dim_feedforward, dropout=self.dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config["num_layers"])

        # Output linear layer
        self.output_layer = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
            src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
      # Convert src and tgt to long type if they are not already
      src = src.long()
      tgt = tgt.long()

      # Create masks for padding tokens
      PAD_IDX = 0
      src_key_padding_mask = (src == PAD_IDX)
      tgt_key_padding_mask = (tgt == PAD_IDX)

      # Create a causal mask for the target, preventing attention to future tokens
      tgt_mask = torch.triu(torch.ones((tgt.size(1), tgt.size(1)), device=tgt.device).bool(), diagonal=1)

      # Pass through the sketch embedding layer
      src_emb = self.sketch_embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
      tgt_emb = self.sketch_embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

      # Apply positional encoding
      src_emb = self.pos_encoder(src_emb.transpose(0, 1))
      tgt_emb = self.pos_encoder(tgt_emb.transpose(0, 1))

      # Pass through the encoder and decoder
      memory = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
      output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)

      # Apply output layer
      output = self.output_layer(output)

      return output


    def print_config(self):
        print("Model Configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
            
if torch.cuda.is_available():
    print("CUDA (GPU support) is available in PyTorch!")
    device = torch.device("cuda")
else:
    print("CUDA (GPU support) is not available. Using CPU...")
    device = torch.device("cpu")
    


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

config = load_config("/home/ml4science2023/GAN/sketch/config.json")
model = Sketchformer(config)
model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr= 0.001)

from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler2 = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_name')
# Training Loop
num_epochs = 30  # Number of epochs
SOS_token = 1002


for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0.0

    # Training loop
    for i, (sketches, _) in enumerate(dataloader):
        sketches = sketches.to(device)
        sos_tensor = torch.full((sketches.size(0), 1, sketches.size(2)), SOS_token, dtype=torch.long, device=device)
        tgt_input = torch.cat([sos_tensor, sketches[:, :-1, :]], dim=1)

        # Create masks
        src_mask = (sketches != PAD_IDX).unsqueeze(-2)
        tgt_mask = (tgt_input != PAD_IDX).unsqueeze(-2)
        tgt_mask = tgt_mask & torch.triu(torch.ones((tgt_input.size(0), tgt_input.size(1), tgt_input.size(1)), device=device), diagonal=1).bool()

        # Forward pass with masks
        outputs = model(sketches, tgt_input, src_key_padding_mask=~src_mask, tgt_key_padding_mask=~tgt_mask)
        outputs = outputs.view(-1, outputs.size(-1))
        targets = sketches.view(-1)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        print(f"{i/len(dataloader)*100} % | Loss: {loss.item():.4f}")
        for name, param in model.named_parameters():
              writer.add_histogram('gradients/' + name, param.grad, epoch * len(dataloader) + i)

        optimizer.step()

        train_loss += loss.item()
        # Log training information
        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + i)

    # Calculate average training loss
    avg_train_loss = train_loss / len(dataloader)

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for sketches, _ in dataloader_test:
            sketches = sketches.to(device)
            sos_tensor = torch.full((sketches.size(0), 1, sketches.size(2)), SOS_token, dtype=torch.long, device=device)
            tgt_input = torch.cat([sos_tensor, sketches[:, :-1, :]], dim=1)

            # Forward pass
            outputs = model(sketches, tgt_input)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = sketches.view(-1)
            loss = criterion(outputs, targets)

            val_loss += loss.item()

    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader_test)
    writer.add_scalar('Loss/validation', avg_val_loss, epoch)

    # Update learning rate scheduler
    scheduler2.step(avg_val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')