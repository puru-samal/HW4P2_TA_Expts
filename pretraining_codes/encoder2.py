import torch.nn as nn
from embeddings2 import * 
from modules2 import *
from typing import Literal
from masks2 import *

'''
in  : B x T x d_model
out : B x T x d_model
'''


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()

       
        self.pre_norm = torch.nn.LayerNorm(d_model)
        self.self_attn = torch.nn.MultiheadAttention(embed_dim=d_model,
                                                        num_heads=num_heads,
                                                        dropout=dropout,
                                                        batch_first=True)
                                                        

        # Two macaron-style feedforward networks
        self.ffn1 = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        

        
        self.dropout = nn.Dropout(dropout)
        # LayerNorms and Dropouts
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        

        

    def forward(self, x, pad_mask):
        # Self-attention with residual connection and normalization
        # attn_output = self.self_attn(x, x, x,  pos_emb, mask=None)

        residual = x
        
        x = self.pre_norm(x)
        att_out, att_weights = self.self_attn(query=x, key=x, value=x, need_weights=True, key_padding_mask=pad_mask, is_causal=False)
        
        x = residual +  self.dropout(att_out)
        
        x = self.norm1(x)

        residual = x
        # First FFN with residual connection and normalization
        x = residual + self.dropout(self.ffn1(x))

        x = self.norm2(x)

        return x , pad_mask


class Encoder(nn.Module):
    def __init__(self, input_dim, num_layers, d_model, num_heads, d_ff, max_len, embed_type:Literal['Conv1DMLP', 'ResBlockMLP', 'BiLSTM', 'Conv2d']='Conv1DMLP', dropout=0.1, pad_token=None):
        super(Encoder, self).__init__()

        # Conv1D + MLP embedding layer
        self.embed_type = embed_type
        self.pad_idx = pad_token
        if self.embed_type == 'Conv1DMLP':
            self.embed = Conv1DMLPEmbedding(input_dim, output_dim=d_model, dropout=dropout)
        elif self.embed_type == 'ResBlockMLP':
            self.embed = ResBlockMLPEmbedding(input_dim, output_dim=d_model, dropout=dropout)
        elif self.embed_type == 'BiLSTM':
            self.embed = BiLSTMEmbedding(input_dim, output_dim=d_model, dropout=dropout)
        elif self.embed_type == "Conv2d":
            self.embed = Conv2dSubsampling(input_dim, d_model, dropout)
        # self.pos_encoding = RelPositionalEncoding(d_model, dropout_rate=dropout)

       
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)

        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Final LayerNorm
        self.after_norm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, x, x_len):
        # Apply embedding

        # x, x_len = self.specaug(x, x_len)
        
        pad_mask = None
        
       
        if self.embed is None:
            x = x
        elif (
            isinstance(self.embed, Conv2dSubsampling)
           
        ):
            x, masks = self.embed(x, masks)
    
          
        if self.embed_type == 'BiLSTM':
            x = self.embed(x, x_len)
        elif self.embed_type == 'Conv2d':
            x, _ = self.embed(x,None)
        else:
            x = self.embed(x)
        # print("embed", x[0])
        # print("embed", x.shape)

        residual = x
        x = self.pos_encoding(x)
        # print("pos enc", x[0])
        ## Apply Dropout
        x = self.dropout(x)
        # Pass through encoder layers
        x = x + residual
        for layer in self.enc_layers:
            # x = layer(x, pos_emb)
            x,pad_mask = layer(x, pad_mask)
            # print("layer", x[0])
        # Apply final normalization
        
        return self.after_norm(x),  x_len
