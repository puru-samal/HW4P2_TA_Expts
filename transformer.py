from encoder import *
from decoder import *

class Transformer(torch.nn.Module):
    def __init__(self, input_dim, enc_num_layers, dec_num_layers, enc_num_heads, dec_num_heads,
                 d_model, d_ff, target_vocab_size, eos_token, sos_token,
                 pad_token, dropout=0.1, trans_max_seq_length=550, mfcc_max_seq_length=3260):

        super(Transformer, self).__init__()

        self.encoder = Encoder(input_dim=input_dim, num_layers=enc_num_layers, d_model=d_model, num_heads=enc_num_heads, d_ff=d_ff, max_len=mfcc_max_seq_length, dropout=0.1)

        self.decoder = Decoder(dec_num_layers, d_model, dec_num_heads, d_ff,
                               dropout, target_vocab_size, trans_max_seq_length, eos_token, sos_token, pad_token)

    def forward(self, padded_input, input_lengths, padded_target, target_lengths):
        # passing through Encoder
        encoder_output = self.encoder(padded_input)
        # print("encoder", encoder_output[0])

        # passing Encoder output and Attention masks through Decoder
        output, attention_weights = self.decoder(padded_target, encoder_output, input_lengths)
        # print("decoder", output[0])
        return output, attention_weights

    def recognize(self, inp, inp_len):
        """ sequence-to-sequence greedy search -- decoding one utterance at a time """
        encoder_outputs  = self.encoder(inp)
        out = self.decoder.recognize_greedy_search(encoder_outputs, inp_len)

        return out