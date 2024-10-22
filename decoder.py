
from modules import *
from masks import *
import torch

'''
in  : B x T x d_model
out : B x d_model x vocab_size
'''

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # @TODO: fill in the blanks appropriately (given the modules above)
        self.mha1       = MultiHeadAttention(n_head=num_heads, d_model=d_model, dropout=dropout)
        self.mha2       = MultiHeadAttention(n_head=num_heads, d_model=d_model, dropout=dropout)
        self.ffn        = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        self.layernorm1 = torch.nn.LayerNorm(d_model)
        self.layernorm2 = torch.nn.LayerNorm(d_model)
        self.layernorm3 = torch.nn.LayerNorm(d_model)

        self.dropout1   = torch.nn.Dropout(dropout)
        self.dropout2   = torch.nn.Dropout(dropout)
        self.dropout3   = torch.nn.Dropout(dropout)


    def forward(self, padded_targets, enc_output, enc_input_lengths, dec_enc_attn_mask, pad_mask, slf_attn_mask):

        # Masked Multi-Head Attention
        #   (1) apply MHA with the lookahead mask
        ''' TODO '''
        mha1_output, mha1_attn_weights = self.mha1(q=padded_targets, k=padded_targets, v=padded_targets, mask=slf_attn_mask)

        # Skip (Residual) Connections
        #   (1) perform dropout on padded attention output
        #   (2) add the true outputs (padded_targets) as a skip connection
        ''' TODO '''
        mha1_output = self.dropout1(mha1_output)
        mha1_output = mha1_output + padded_targets

        # Layer Normalization
        #   (1) call layernorm on this resulting value
        ''' TODO '''
        mha1_output = self.layernorm1(mha1_output)

        # Masked Multi-Head Attention on Encoder Outputs and Targets
        #   (1) apply MHA with the self-attention mask
        ''' TODO '''
        mha2_output, mha2_attn_weights = self.mha2(q=mha1_output, k=enc_output, v=enc_output, mask=dec_enc_attn_mask)

        # Skip (Residual) Connections
        #   (1) perform dropout on this second padded attention output
        #   (2) add the output of first MHA block as a skip connection
        ''' TODO '''
        mha2_output = self.dropout2(mha2_output)
        mha2_output = mha2_output + mha1_output

        # Layer Normalization
        #   (1) call layernorm on this resulting value
        ''' TODO '''
        mha2_output = self.layernorm2(mha2_output)

        # Feed Forward Network
        #   (1) pass through the FFN
        ''' TODO '''
        ffn_output = self.ffn(mha2_output)

        # Skip (Residual) Connections
        #   (1) perform dropout on the output
        #   (2) add the output of second MHA block as a skip connection
        ''' TODO '''
        ffn_output = self.dropout3(ffn_output)
        ffn_output = ffn_output + mha2_output

        # apply Layer Normalization on this resulting value
        ''' TODO '''
        ffn_output = self.layernorm3(ffn_output)

        # return the network output and both attention weights (for mha1 and mha2)
        # @NOTE: returning the self attention weights first
        return ffn_output, mha1_attn_weights, mha2_attn_weights



class Decoder(torch.nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout,
            target_vocab_size, max_seq_length, eos_token, sos_token, pad_token):
        super().__init__()

        self.EOS_TOKEN      = eos_token
        self.SOS_TOKEN      = sos_token
        self.PAD_TOKEN      = pad_token

        self.max_seq_length = max_seq_length
        self.num_layers     = num_layers

        # use torch.nn.ModuleList() with list comprehension looping through num_layers
        # @NOTE: think about what stays constant per each DecoderLayer (how to call DecoderLayer)
        # @HINT: We've implemented this for you.
        self.dec_layers = torch.nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.target_embedding       = torch.nn.Embedding(target_vocab_size, d_model)
        # self.positional_encoding    = PositionalEncoding(d_model=d_model, dropout_rate = dropout,max_len=max_seq_length)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_seq_length)
        self.final_linear           = torch.nn.Linear(d_model, target_vocab_size)
        self.dropout                = torch.nn.Dropout(dropout)


    def forward(self, padded_targets, enc_output, enc_input_lengths):

        # create a padding mask for the padded_targets with <PAD_TOKEN>
        ''' TODO '''
        pad_mask = create_mask_1(padded_input=padded_targets, input_lengths=None, pad_idx=self.PAD_TOKEN)

        # creating an attention mask for the future subsequences (look-ahead mask)
        ''' TODO '''
        look_ahead_mask = create_mask_2(seq=padded_targets)

        # creating attention mask to ignore padding positions in the input sequence during attention calculation
        ''' TODO '''
        dec_enc_attn_mask = create_mask_3(padded_input=enc_output, input_lengths=enc_input_lengths, expand_length=padded_targets.size(1))

        # computing embeddings for the target sequence
        ''' TODO '''
        x = self.target_embedding(padded_targets)


        # computing Positional Encodings with the embedded targets and apply dropout
        ''' TODO '''
        x = self.positional_encoding(x)

        # passing through decoder layers
        # @NOTE: store your mha1 and mha2 attention weights inside a dictionary
        # @NOTE: you will want to retrieve these later so store them with a useful name
        ''' TODO '''
        mha1_attn_weights = {}
        mha2_attn_weights = {}
        for i in range(self.num_layers):
            x, mha1_attn_weights[i], mha2_attn_weights[i] = self.dec_layers[i](padded_targets=x,
                                                                               enc_output=enc_output,
                                                                               enc_input_lengths=enc_input_lengths,
                                                                               dec_enc_attn_mask=dec_enc_attn_mask,
                                                                               pad_mask=pad_mask,
                                                                               slf_attn_mask=look_ahead_mask)

        # linear layer (Final Projection) for next character prediction
        ''' TODO '''
        seq_out = self.final_linear(x)

        # return the network output and the dictionary of attention weights
        return seq_out, (mha1_attn_weights, mha2_attn_weights)


    def recognize_greedy_search(self, enc_outputs, enc_input_lengths):
        ''' passes the encoder outputs and its corresponding lengths through autoregressive network

            @NOTE: You do not need to make changes to this method.
        '''

        batch_size = enc_outputs.size(0)

        # start with the <SOS> token for each sequence in the batch
        target_seq = torch.full((batch_size, 1), self.SOS_TOKEN, dtype=torch.long).to(enc_outputs.device)

        finished = torch.zeros(batch_size, dtype=torch.bool).to(enc_outputs.device)

        for _ in range(self.max_seq_length):

            # preparing attention masks
            # filled with ones becaues we want to attend to all the elements in the sequence
            pad_mask = torch.ones_like(target_seq).float().unsqueeze(-1)  # (batch_size x i x 1)
            slf_attn_mask_subseq = create_mask_2(target_seq)

            x = self.positional_encoding(self.target_embedding(target_seq))

            for i in range(self.num_layers):
                x, block1, block2 = self.dec_layers[i](
                    x, enc_outputs, enc_input_lengths, None, pad_mask, slf_attn_mask_subseq)

            seq_out = self.final_linear(x[:, -1])
            logits = torch.nn.functional.log_softmax(seq_out, dim=1)

            # selecting the token with the highest probability
            # @NOTE: this is the autoregressive nature of the network!
            next_token = logits.argmax(dim=-1).unsqueeze(1)

            # appending the token to the sequence
            target_seq = torch.cat([target_seq, next_token], dim=-1)

            # Checking if <EOS> or <PAD> token is generated
            eos_mask = next_token.squeeze(-1) == self.EOS_TOKEN
            pad_mask = next_token.squeeze(-1) == self.PAD_TOKEN
            finished |= eos_mask | pad_mask  # Mark sequences as finished

            # end if all sequences have generated the EOS token
            if finished.all(): break

        # Remove everything after EOS token
        eos_indices = (target_seq == self.EOS_TOKEN).float().argmax(dim=1)
        mask = torch.arange(target_seq.size(1), device=target_seq.device)[None, :] < eos_indices[:, None]
        target_seq = target_seq.masked_fill(~mask, self.PAD_TOKEN)

        return target_seq