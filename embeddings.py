
import torch.nn as nn

'''
Feel free to add more embeddings. Only restriction is that they must be simple (Max 2 layers BiLSTM or small simple CNN-MLP combos)s
'''

class BiLSTMEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(BiLSTMEmbedding, self).__init__()

        # BiLSTM with 2 layers
        self.bilstm = nn.LSTM(
                input_dim, output_dim // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout
            )

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
        Returns:
            Output tensor (batch_size, seq_len, output_dim)
        """
        # BiLSTM expects (batch_size, seq_len, input_dim)
        output, _ = self.bilstm(x)  # Output: (batch_size, seq_len, output_dim)
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
                nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride),
                nn.BatchNorm2d(output_dim)
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