import numpy as np

def apply_rope_with_offset(x, cos, sin, start_pos):
    """
    Apply rotary positional embedding to x starting at absolute position start_pos.

    Args:
        x: array-like of shape (batch, n_heads, seq_len, head_dim)
        cos: array-like of shape (max_seq_len, head_dim)
        sin: array-like of shape (max_seq_len, head_dim)
        start_pos: int, absolute position of the first token in x

    Returns:
        Nested list with the same shape as x.
    """
    x = np.asarray(x)
    batch, n_heads, seq_len, head_dim = x.shape

    # Select the absolute positions of the new tokens
    cos_slice = cos[start_pos:start_pos + seq_len]
    sin_slice = sin[start_pos:start_pos + seq_len]

    # Construct the 90-degree rotation of each pair
    half_dim = head_dim // 2
    first_half = x[..., :half_dim]
    second_half = x[..., half_dim:]
    rotated = np.concatenate(
        [-second_half, first_half], axis=-1
    )

    # Apply the position-dependent rotation
    return (x * cos_slice + rotated * sin_slice).tolist()

