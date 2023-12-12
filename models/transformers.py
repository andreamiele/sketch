import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        # Create a long enough positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
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
    def __init__(self, d_model, nhead, dropout=0.1):
        super(SelfAttentionBottleneck, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)

    def forward(self, src):
        attn_output, _ = self.self_attn(src, src, src)
        return attn_output


class Sketchformer(nn.Module):
    def __init__(self, config):
        super(Sketchformer, self).__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.num_layers = config["num_layers"]
        self.d_model = config["d_model"]
        self.nhead = config["nhead"]
        self.dim_feedforward = config["dim_feedforward"]
        self.max_seq_length = config["max_seq_length"]
        self.dropout = config["dropout"]
        
        self.sketch_embedding = SketchEmbedding(self.vocab_size, self.d_model, self.max_seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, 
                                                   dim_feedforward=self.dim_feedforward, 
                                                   dropout=self.dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.nhead, 
                                                   dim_feedforward=self.dim_feedforward, 
                                                   dropout=self.dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)
        self.self_attention_bottleneck = SelfAttentionBottleneck(self.d_model, self.nhead)
        self.output_layer = nn.Linear(self.d_model, self.vocab_size)
        
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, 
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # src: source sequence, tgt: target sequence for reconstruction

        src_emb = self.sketch_embedding(src)
        tgt_emb = self.sketch_embedding(tgt)

        # Pass through the encoder
        memory = self.encoder(src_emb, src_mask, src_key_padding_mask)

        # Apply self-attention bottleneck
        bottleneck_output = self.self_attention_bottleneck(memory)

        # Decode the output
        output = self.decoder(tgt_emb, bottleneck_output, tgt_mask, memory_mask, 
                              tgt_key_padding_mask, memory_key_padding_mask)

        return self.output_layer(output)
    
    def print_config(self):
        print("Model Configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")