from torch.nn import nn
from embeddings import * 
from modules import *
from typing import Literal


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.mha        = MultiHeadAttention(num_heads, d_model)
        self.ffn        = FeedForward(d_model, d_ff, dropout)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)


    def forward(self, inp):
        # Multi-Head Attention
        #   (1) perform Multi-Head Attention on x
        inp = self.layernorm1(inp)
        attn_output, self_attention_weights = self.mha(inp, inp, inp)

        # Skip (Residual) Connection
        #   (1) perform dropout
        #   (2) add the input as a skip connection
        res = inp + self.dropout1(attn_output)

        # Layer Normalization
        #   (1) call layernorm on this resulting value
        res = self.layernorm2(res)

        # Feed Forward Network
        #   (1) apply feed forward layer
        #   (2) apply the padding mask  to the output
        ffn_output = self.ffn(res)
        ffn_output = self.dropout2(ffn_output)

        # Skip (Residual) Connection
        #   (1) add the result before LayerNorm and FFN as skip connection
        x = res + ffn_output

        # Layer Normalization
        #   (1) call layernorm on this resulting value
        block_output = self.layernorm3(x)
        return block_output, self_attention_weights
    


class Encoder(nn.Module):
    def __init__(self, input_dim, num_layers, d_model, num_heads, d_ff, max_len, embed_type:Literal['ConvIDMLP', 'ResBlockBLP', 'BiLSTM']='ConvIDMLP', dropout=0.1):
        super(Encoder, self).__init__()

        # Conv1D + MLP embedding layer
        self.embed_type = embed_type
        
        if self.embed_type == 'ConvIDMLP':
            self.embed = Conv1DMLPEmbedding(input_dim, output_dim=d_model, dropout=dropout)
        elif self.embed_type == 'ResBlockBLP':
            self.embed = ResBlockMLPEmbedding(input_dim, output_dim=d_model, dropout=dropout)
        elif self.embed_type == 'BiLSTM':
            self.embed = BiLSTMEmbedding(input_dim, output_dim=d_model, dropout=dropout)

        # self.pos_encoding = RelPositionalEncoding(d_model, dropout_rate=dropout)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)

        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Final LayerNorm
        self.after_norm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, x):
        # Apply embedding
        x = self.embed(x)
        # print("embed", x[0])
        # print("embed", x.shape)

        # x, pos_emb = self.pos_encoding.forward(x)
        x = self.pos_encoding(x)
        # print("pos enc", x[0])

        # Pass through encoder layers
        for layer in self.enc_layers:
            # x = layer(x, pos_emb)
            x = layer(x, None)
            # print("layer", x[0])
        # Apply final normalization
        return self.after_norm(x)
