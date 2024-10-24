import torch
import logging
def create_mask_1(padded_input, pad_idx=1):
    """ Create a mask to identify padding positions in the time dimension (T).

    Args:
        padded_input: The input tensor with padding, shape (N, T).
        pad_idx: The index used for padding tokens.

    Returns:
        A mask tensor with shape (N, T), where padding positions are marked with 1 and non-padding positions are marked with 0.
    """
    # Create a boolean mask where padding tokens are True (1) and non-padding tokens are False (0)
    pad_mask =  (padded_input.sum(dim=-1) == pad_idx) # (N, T)

    # Return the mask as boolean (1 for padding, 0 for non-padding)
    return pad_mask

def create_pad_mask_dec(padded_input, pad_idx=1):
    """ Create a mask to identify padding positions in the time dimension (T).

    Args:
        padded_input: The input tensor with padding, shape (N, T).
        pad_idx: The index used for padding tokens.

    Returns:
        A mask tensor with shape (N, T), where padding positions are marked with 1 and non-padding positions are marked with 0.
    """
    # Create a boolean mask where padding tokens are True (1) and non-padding tokens are False (0)
    pad_mask = (padded_input == pad_idx) # (N, T)

    # Return the mask as boolean (1 for padding, 0 for non-padding)
    return pad_mask

def create_mask_2(seq, repeat, mask_type='binary'):
    """ Create a mask to prevent positions from attending to subsequent positions.

    Args:
        seq: The input sequence tensor, shape (batch_size, sequence_length).
        repeat: The number of attention heads to repeat for.
        mask_type: Type of mask, either 'binary' (True/False) or 'float' (-inf/0).

    Returns:
        A mask tensor with shape (batch_size * num_heads, sequence_length, sequence_length),
        where positions are allowed to attend to previous positions but not to subsequent positions.
        If 'binary', True means no attention allowed. If 'float', -inf means no attention allowed.
    """
    batch_size, seq_len = seq.size()

    # Create a lower triangular matrix with ones, ensuring causal masking
    attn_shape = (seq_len, seq_len)
    subsequent_mask = torch.tril(torch.ones(attn_shape))

    if mask_type == 'binary':
        # Convert to a binary mask where True = cannot attend (future tokens), False = can attend
        subsequent_mask = subsequent_mask == 0
    elif mask_type == 'float':
        # Convert to a float mask where -inf = cannot attend (future tokens), 0 = can attend
        subsequent_mask = (1.0 - subsequent_mask) * float('-inf')

    # Repeat the mask for batch size and heads, but collapse into (batch_size * num_heads)
    mask = subsequent_mask.unsqueeze(0).repeat(batch_size * repeat, 1, 1)
    # Shape: (batch_size * num_heads, seq_len, seq_len)

    return mask

def create_mask_3(padded_input, expand_length, pad_idx=1):
    """ Create an attention mask to ignore padding positions in the input sequence during attention calculation.

    Args:
        padded_input: The input tensor with padding, shape (N, Ti, ...).
        input_lengths: The actual lengths of each sequence before padding, shape (N,).
        expand_length: The length to which the attention mask should be expanded,
            usually equal to the length of the sequence that the attention scores will be applied to.

    Returns:
        An attention mask tensor with shape (N, expand_length, Ti),
            where padding positions in the input sequence are marked with 1 and other positions are marked with 0.
    """

    # Create a mask to identify non-padding positions, shape (N, Ti, 1)
    # (N x Ti x 1)
    non_pad_mask    = create_mask_1(padded_input,pad_idx=pad_idx )

    # Invert the mask to identify padding positions, shape (N, Ti)
    # N x Ti, lt(1) like-not operation
    pad_mask        = non_pad_mask.squeeze(-1).lt(1)


    # Expand the mask to match the dimensions of the attention matrix, shape (N, expand_length, Ti)
    attn_mask       = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)

    return attn_mask



