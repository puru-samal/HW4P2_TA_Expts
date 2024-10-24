from encoder2 import *
from decoder2 import *


class Transformer(torch.nn.Module):
    def __init__(self, input_dim, enc_num_layers, dec_num_layers, enc_num_heads, dec_num_heads,
                 d_model, d_ff, target_vocab_size, eos_token, sos_token,
                 pad_token, enc_dropout, dec_dropout, trans_max_seq_length=550, mfcc_max_seq_length=3260, embed_type:Literal['Conv1DMLP', 'ResBlockMLP', 'BiLSTM']='Conv1DMLP'):

        super(Transformer, self).__init__()

        self.encoder = Encoder(embed_type=embed_type, input_dim=input_dim, d_model=d_model, num_heads=enc_num_heads, 
                               num_layers=enc_num_layers, d_ff=d_ff, max_len=mfcc_max_seq_length, dropout=enc_dropout,pad_token=pad_token)

        self.decoder = Decoder(num_layers=dec_num_layers, d_model=d_model, num_heads=dec_num_heads, d_ff=d_ff,
                               dropout=dec_dropout, target_vocab_size=target_vocab_size, max_seq_length=trans_max_seq_length, 
                               eos_token=eos_token, sos_token=sos_token, pad_token=pad_token)

    def forward(self, padded_input, input_lengths, padded_target, target_lengths, pre_train=False):
        # passing through Encoder
        if pre_train:
            encoder_output = None
        else:
            encoder_output,olens = self.encoder(padded_input, input_lengths)
        # print("encoder", encoder_output[0])
        
        # passing Encoder output and Attention masks through Decoder
        output, attention_weights = self.decoder(padded_target, encoder_output, input_lengths, target_lengths,pre_train)
        # print("decoder", output[0])
        return output, attention_weights

    def recognize(self, inp, inp_len):
        """ sequence-to-sequence greedy search -- decoding one utterance at a time """
        encoder_outputs,olens  = self.encoder(inp, inp_len)
        out = self.decoder.recognize_greedy_search(encoder_outputs, inp_len)

        return out
    
    def recognize_lm(self, batch_size, initial_input=None, target_seq=None):
        """ sequence-to-sequence greedy search -- decoding one utterance at a time """
       
        current_seq, mha1_attn_weights, total_loss, total_tokens = self.decoder.recognize_greedy_search_lm(batch_size, initial_input, target_seq)

        return current_seq, mha1_attn_weights, total_loss, total_tokens