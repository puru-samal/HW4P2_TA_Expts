import torch.nn as nn
from embeddings import * 
from modules import *
from typing import Literal

'''
in  : B x T x d_model
out : B x T x d_model
'''


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        # Multi-head self-attention with relative positional encoding
        # self.self_attn = RelPositionMultiHeadedAttention(num_heads, d_model, dropout)
        self.self_attn = MultiHeadedAttention(n_head=num_heads, n_feat=d_model, dropout_rate=dropout)

        # Two macaron-style feedforward networks
        self.ffn1 = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.ffn2 = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Depthwise convolution for local context
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)

        # LayerNorms and Dropouts
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.norm4 = LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb):
        # Self-attention with residual connection and normalization
        # attn_output = self.self_attn(x, x, x,  pos_emb, mask=None)
        attn_output = self.self_attn(x, x, x, mask=None)
        x = self.norm1(x + self.dropout(attn_output))

        # First FFN with residual connection and normalization
        ffn_output = self.ffn1(x)
        x = self.norm2(x + self.dropout(ffn_output))

        # Depthwise convolution with residual connection and normalization
        conv_output = self.depthwise_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm3(x + self.dropout(conv_output))

        # Second FFN with residual connection and normalization
        ffn_output = self.ffn2(x)
        x = self.norm4(x + self.dropout(ffn_output))

        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, num_layers, d_model, num_heads, d_ff, max_len, embed_type:Literal['Conv1DMLP', 'ResBlockMLP', 'BiLSTM']='Conv1DMLP', dropout=0.1):
        super(Encoder, self).__init__()

        # Conv1D + MLP embedding layer
        self.embed_type = embed_type
        
        if self.embed_type == 'Conv1DMLP':
            self.embed = Conv1DMLPEmbedding(input_dim, output_dim=d_model, dropout=dropout)
        elif self.embed_type == 'ResBlockMLP':
            self.embed = ResBlockMLPEmbedding(input_dim, output_dim=d_model, dropout=dropout)
        elif self.embed_type == 'BiLSTM':
            self.embed = BiLSTMEmbedding(input_dim, output_dim=d_model, dropout=dropout)

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

        if self.embed_type == 'BiLSTM':
            x = self.embed(x, x_len)
        else:
            x = self.embed(x)
        # print("embed", x[0])
        # print("embed", x.shape)

        # x, pos_emb = self.pos_encoding.forward(x)
        x = self.pos_encoding(x)
        # print("pos enc", x[0])
        ## Apply Dropout
        x = self.dropout(x)
        # Pass through encoder layers
        for layer in self.enc_layers:
            # x = layer(x, pos_emb)
            x = layer(x, None)
            # print("layer", x[0])
        # Apply final normalization
        return self.after_norm(x)
