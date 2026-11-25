import numpy as np

def grouped_query_attention(Q, K, V, num_heads, num_kv_heads):
    """
    Compute Grouped Query Attention.
    
    Args:
        Q: Query tensor, shape (batch_size, seq_len, num_heads * head_dim)
        K: Key tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        V: Value tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
    
    Returns:
        Output tensor, shape (batch_size, seq_len, num_heads * head_dim)
    """
    B, T, hidden_dim = Q.shape
    d = hidden_dim // num_heads
    G = num_heads // num_kv_heads
    H_kv = num_kv_heads

    # Separate KV groups and query heads within each group
    Q = Q.reshape(B, T, H_kv, G, d).transpose(0, 2, 3, 1, 4)

    # Introduce a singleton dimension for broadcasting
    K = K.reshape(B, T, H_kv, 1, d).transpose(0, 2, 3, 1, 4)
    V = V.reshape(B, T, H_kv, 1, d).transpose(0, 2, 3, 1, 4)

    # Scaled dot-product attention
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d)

    # Numerically stable softmax
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= np.sum(weights, axis=-1, keepdims=True)

    context = weights @ V

    # Concatenate all query heads
    return context.transpose(0, 3, 1, 2, 4).reshape(B, T, hidden_dim)
