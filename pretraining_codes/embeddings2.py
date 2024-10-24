
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
'''
Feel free to add more embeddings. Only restriction is that they must be simple (Max 2 layers BiLSTM or small simple CNN-MLP combos)s
# Should follow: 
# in : B x T x input_dim
# out: B x T x d_model
'''

class BiLSTMEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(BiLSTMEmbedding, self).__init__()

        # BiLSTM with 2 layers
        self.bilstm = nn.LSTM(
                input_dim, output_dim // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout
            )

    def forward(self, x,  x_len):
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
        Returns:
            Output tensor (batch_size, seq_len, output_dim)
        """
        # BiLSTM expects (batch_size, seq_len, input_dim)
        # Pack the padded sequence to avoid computing over padded tokens
        packed_input = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)

        # Pass through the BiLSTM
        packed_output, _ = self.bilstm(packed_input)

        # Unpack the sequence to restore the original padded shape
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        return output

class Conv1DMLPEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(Conv1DMLPEmbedding, self).__init__()
        # Conv1D + MLP embedding path
        self.conv = nn.Conv1d(
            in_channels=input_dim, out_channels=output_dim, kernel_size=3, padding=1
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # MLP for projection
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
        Returns:
            Output tensor (batch_size, seq_len, output_dim)
        """
        # Conv1D expects (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        x = self.conv(x)  # (batch_size, output_dim, seq_len)
        x = self.activation(x)
        x = self.dropout(x)
        output = self.mlp(x.transpose(1, 2))  # (batch_size, seq_len, output_dim)
        return output
    
class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)

    def forward(self, x, x_mask):
        
        print(self.out.weight.shape)
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        print(x.shape)
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class ResBlockMLPEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn1 = nn.BatchNorm1d(output_dim)

        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm1d(output_dim)

        self.skip = nn.Sequential()

        if stride != 1 or input_dim != output_dim:
            # Adjust input shape to match the output shape with a 1x1 convolution
            self.skip = nn.Sequential(
                nn.Conv1d(input_dim, output_dim, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_dim)
            )

        self.act = nn.GELU()
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
        Returns:
            Output tensor (batch_size, seq_len, output_dim)
        """
        # Conv1D expects (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        out = self.conv1(x)    # (batch_size, output_dim, seq_len)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out + self.skip(x))
        out = self.mlp(out.transpose(1, 2))  # (batch_size, seq_len, output_dim)
        return out