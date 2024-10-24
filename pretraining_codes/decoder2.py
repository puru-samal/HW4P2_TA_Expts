
from masks2 import *
from modules2 import *

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # @TODO: fill in the blanks appropriately (given the modules above)
        self.mha1       = torch.nn.MultiheadAttention(embed_dim=d_model,
                                                        num_heads=num_heads,
                                                        dropout=dropout,
                                                        batch_first=True)
        self.mha2       = torch.nn.MultiheadAttention(embed_dim=d_model,
                                                        num_heads=num_heads,
                                                        dropout=dropout,
                                                        batch_first=True)
        self.ffn        = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.pre_norm   = torch.nn.LayerNorm(d_model)
        self.layernorm1 = torch.nn.LayerNorm(d_model)
        self.layernorm2 = torch.nn.LayerNorm(d_model)
        self.layernorm3 = torch.nn.LayerNorm(d_model)

        self.dropout1   = torch.nn.Dropout(dropout)
        self.dropout2   = torch.nn.Dropout(dropout)
        self.dropout3   = torch.nn.Dropout(dropout)


    def forward(self, padded_targets, enc_output, enc_input_lengths, pad_mask_enc, pad_mask, slf_attn_mask, pretrain=False):


        

        x = self.pre_norm(padded_targets)

        residual = x
        # Masked Multi-Head Attention
        #   (1) apply MHA with the lookahead mask
        ''' TODO '''
        mha1_output, mha1_attn_weights = self.mha1(query=x,
                                                      key=x,
                                                      value=x,
                                                      key_padding_mask=pad_mask,
                                                      need_weights=True,
                                                      attn_mask = slf_attn_mask,
                                                      average_attn_weights=True,
                                                      is_causal=True)

        # Skip (Residual) Connections
        #   (1) perform dropout on padded attention output
        #   (2) add the true outputs (padded_targets) as a skip connection
        ''' TODO '''
        mha1_output = residual + self.dropout1(mha1_output)
        

        # Layer Normalization
        #   (1) call layernorm on this resulting value
        ''' TODO '''
        mha1_output = self.layernorm1(mha1_output)

        residual = mha1_output

        # Masked Multi-Head Attention on Encoder Outputs and Targets
        #   (1) apply MHA with the self-attention mask
        ''' TODO '''
        if pretrain:
            ffn_output = self.ffn(mha1_output)

            # Skip (Residual) Connections
            #   (1) perform dropout on the output
            #   (2) add the output of second MHA block as a skip connection
            ''' TODO '''
            ffn_output = self.dropout3(ffn_output)
            ffn_output = ffn_output + residual

            # apply Layer Normalization on this resulting value
            ''' TODO '''
            ffn_output = self.layernorm3(ffn_output)

            # return the network output and both attention weights (for mha1 and mha2)
            # @NOTE: returning the self attention weights first
            return ffn_output, mha1_attn_weights, mha1_attn_weights
        else:
            mha2_output, mha2_attn_weights = self.mha2(query=padded_targets,
                                                        key=enc_output,
                                                        value=enc_output,
                                                        key_padding_mask=pad_mask_enc,
                                                        need_weights=True,
                                                        average_attn_weights=True,
                                                        is_causal=False)

            # Skip (Residual) Connections
            #   (1) perform dropout on this second padded attention output
            #   (2) add the output of first MHA block as a skip connection
            ''' TODO '''
            mha2_output = residual + self.dropout2(mha2_output)
        

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
        self.num_heads = num_heads
        # use torch.nn.ModuleList() with list comprehension looping through num_layers
        # @NOTE: think about what stays constant per each DecoderLayer (how to call DecoderLayer)
        # @HINT: We've implemented this for you.
        self.dec_layers = torch.nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.target_embedding       = torch.nn.Embedding(target_vocab_size, d_model)
        self.positional_encoding    = PositionalEncoding(d_model=d_model, max_len=max_seq_length)
        self.final_linear           = torch.nn.Linear(d_model, target_vocab_size)
        self.dropout                = torch.nn.Dropout(dropout)
        self.nll_loss = nn.NLLLoss(ignore_index=self.PAD_TOKEN, reduction='sum')

    def forward(self, padded_targets, enc_output, enc_input_lengths, target_lengths, pretrain):

        # create a padding mask for the padded_targets with <PAD_TOKEN>
        ''' TODO '''
        
        pad_mask = create_pad_mask_dec(padded_input=padded_targets, pad_idx=self.PAD_TOKEN).to(padded_targets.device)
        # creating an attention mask for the future subsequences (look-ahead mask)
        ''' TODO '''
        # look_ahead_mask = create_mask_2(seq=padded_targets)
        look_ahead_mask = create_mask_2(seq=padded_targets, repeat=self.num_heads).to(padded_targets.device)
        # creating attention mask to ignore padding positions in the input sequence during attention calculation
        ''' TODO '''
        # dec_enc_attn_mask = create_mask_3(padded_input=enc_output,  expand_length=padded_targets.size(1), pad_idx=self.PAD_TOKEN)
        
        # computing embeddings for the target sequence
        ''' TODO '''
        x = self.target_embedding(padded_targets)
        if pretrain:
            pad_mask_enc = None
        else:
            pad_mask_enc = create_mask_1(enc_output, pad_idx=self.PAD_TOKEN)
        # computing Positional Encodings with the embedded targets and apply dropout
        ''' TODO '''
        x = self.positional_encoding(x)

        # passing through decoder layers
        # @NOTE: store your mha1 and mha2 attention weights inside a dictionary
        # @NOTE: you will want to retrieve these later so store them with a useful name
        ''' TODO '''
        runnint_att = {}
        
        for i in range(self.num_layers):
            x, runnint_att['layer{}_dec_self'.format(i + 1)], runnint_att['layer{}_dec_self'.format(i + 1)] = self.dec_layers[i](padded_targets=x,
                                                                               enc_output=enc_output,
                                                                               enc_input_lengths=enc_input_lengths,
                                                                               pad_mask_enc=pad_mask_enc,
                                                                               pad_mask=pad_mask,
                                                                               slf_attn_mask=look_ahead_mask,pretrain=pretrain)

        # linear layer (Final Projection) for next character prediction
        ''' TODO '''
        seq_out = self.final_linear(x)

        # return the network output and the dictionary of attention weights
        return seq_out, runnint_att


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
            
            slf_attn_mask_subseq = create_mask_2(target_seq,repeat=self.num_heads).to(enc_outputs.device) #To be used in LM
            pad_mask_enc = create_mask_1(enc_outputs,pad_idx=self.PAD_TOKEN).to(enc_outputs.device)
            x = self.positional_encoding(self.target_embedding(target_seq))
            x  = x + self.dropout(x)
            for i in range(self.num_layers):
                x, block1, block2 = self.dec_layers[i](
                    x, enc_outputs, enc_input_lengths, None, None, slf_attn_mask_subseq)

            seq_out = self.final_linear(x[:, -1])
            logits = torch.nn.functional.log_softmax(seq_out, dim=1)

            # selecting the token with the highest probability
            # @NOTE: this is the autoregressive nature of the network!
            next_token = logits.argmax(dim=-1).unsqueeze(1)

            # appending the token to the sequence
            target_seq = torch.cat([target_seq, next_token], dim=-1)

            # checking if <EOS> token is generated
            eos_mask = next_token.squeeze(-1) == self.EOS_TOKEN
            # or opration, if both or one of them is true store the value of the finished sequence in finished variable
            finished |= eos_mask

            # end if all sequences have generated the EOS token
            if finished.all(): break

        # remove the initial <SOS> token and pad sequences to the same length
        target_seq = target_seq[:, 1:]
        max_length = target_seq.size(1)
        target_seq = torch.nn.functional.pad(target_seq,
            (0, self.max_seq_length - max_length), value=self.PAD_TOKEN)

        return target_seq



    def recognize_greedy_search_lm(self, batch_size, initial_input=None, target_seq=None):
        ''' Language model decoding with conditioning on initial input tokens.
            initial_input: a tensor of shape (batch_size, initial_seq_len) containing
                        the initial tokens to condition the decoding process.
        '''

        # Start with <SOS> token for sequences that don't have an initial input
        if initial_input is None:
            current_seq = torch.full((batch_size, 1), self.SOS_TOKEN, dtype=torch.long).to(self.final_linear.weight.device)  # (batch_size x 1)
        else:
            # Use the initial input directly, assuming it already includes <SOS>
            current_seq = initial_input

        # Track which sequences have finished (based on EOS or PAD)
        finished = torch.zeros(batch_size, dtype=torch.bool).to(self.final_linear.weight.device)  # (batch_size)

        # prompt length
        prompt_len = current_seq.shape[1]


        total_loss = 0.0
        total_tokens = 0
        for i in range(self.max_seq_length - prompt_len):
            # Create pad_mask for the padded parts of the current_seq
            pad_mask = create_mask_1(current_seq, pad_idx=self.PAD_TOKEN)

            # No need for look-ahead masking, since there's no future tokens to mask
            # Compute positional embeddings and apply target embeddings
            x = (self.target_embedding(current_seq))  # (batch_size x current_seq_len x d_model)
            
            x = self.positional_encoding(x)

            # Pass through the decoder layers (iterate over all layers)
            mha1_attn_weights = {}
            for i in range(self.num_layers):
                # x, mha1_attn_weights[i] = self.dec_layers[i](padded_targets=x,
                #                                               pad_mask=pad_mask,
                #                                                 slf_attn_mask=None,
                #                                                   is_causal=False)
                x, mha1_attn_weights, mha1_attn_weights = self.dec_layers[i](x, 1, 1, None, None, None,True)

            # Project to vocabulary space (final linear layer)
            seq_out = self.final_linear(x[:, -1])  # (batch_size x vocab_size)

            # Apply log softmax to get softmax probabilities
            probs = torch.nn.functional.softmax(seq_out, dim=1)  # (batch_size x vocab_size)

            # Selecting the next token
           
            next_token = torch.argmax(probs, dim=-1, keepdim=True)  # (batch_size x 1)
            

            # Append the next token. This is autoregressive decoding!
            current_seq = torch.cat([current_seq, next_token], dim=-1)  # (batch_size x current_seq_len+1)

            # compute NLL
            # If target_seq is provided, compute loss for this step
            if target_seq is not None:
                log_probs = torch.log(probs)
                loss = self.nll_loss(log_probs, target_seq[:, prompt_len + i])
                total_loss += loss.item()

                valid_tokens = (target_seq[:, prompt_len + i] != self.PAD_TOKEN).sum().item()
                total_tokens += valid_tokens

            # Checking if <EOS> or <PAD> token is generated
            eos_mask = next_token.squeeze(-1) == self.EOS_TOKEN
            pad_mask = next_token.squeeze(-1) == self.PAD_TOKEN
            finished |= eos_mask | pad_mask  # Mark sequences as finished

            # End if all sequences have generated the EOS or PAD token
            if finished.all():
                break

        # Remove any extra padding and return the target sequence
        current_seq = current_seq[:, 1:]  # Remove the initial token if needed (i.e., <SOS>)

        # Remove everything after EOS token
        eos_indices = (current_seq == self.EOS_TOKEN).float().argmax(dim=1)
        mask = torch.arange(current_seq.size(1), device=current_seq.device)[None, :] < eos_indices[:, None]
        current_seq = current_seq.masked_fill(~mask, self.PAD_TOKEN)

        return current_seq, mha1_attn_weights, total_loss, total_tokens
    def generate(self, initial_input, max_gen_length=500):
        ''' Generate a sequence given an initial input till max_gen_length, does not support batch generation '''



        # check if input is batched, in which case throw an error
        if initial_input.shape[0] != 1:
            raise ValueError("Input should not be batched")

        # Start with < SOS > token for sequences that don't have an initial input
        if initial_input is None:
            current_seq = torch.full((1, 1), self.SOS_TOKEN, dtype=torch.long).to(self.final_linear.weight.device)
        else:
            # Use the initial input directly, assuming it already includes < SOS >
            current_seq = initial_input

        all_mha1_attn_weights = []


        for i in range(max_gen_length - current_seq.size(1)):
            # Compute positional embeddings and apply target embeddings
            x = self.target_embedding(current_seq)  # (batch_size x current_seq_len x d_model)
            
            x = self.positional_encoding(x)

            mha1_attn_weights = {}
            # Pass through the decoder layers
            for j in range(self.num_layers):
                # x, mha1_attn_weights[j] = self.dec_layers[j](padded_targets=x,
                #                                              pad_mask=None,
                #                                              slf_attn_mask=None,
                #                                              is_causal=False)
                x, mha1_attn_weights, mha1_attn_weights = self.dec_layers[i](x, 1, 1, None, None, None,True)

            all_mha1_attn_weights.append(mha1_attn_weights)

            # Project to vocabulary space (final linear layer)
            seq_out = self.final_linear(x[:, -1])  # (1 x vocab_size)

            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(seq_out, dim=1)  # (1 x vocab_size)

            # Selecting the next token
            next_token = torch.argmax(probs, dim=-1).unsqueeze(0)  # (1 x 1)

            # Appending the token to the sequence
            current_seq = torch.cat([current_seq, next_token], dim=-1)  # (1 x current_seq_len+1)

            # Checking if <EOS> token is generated
            if next_token.item() == self.EOS_TOKEN:
                break

        # Remove the batch dimension
        current_seq = current_seq.squeeze(0)

        return current_seq, all_mha1_attn_weights
 