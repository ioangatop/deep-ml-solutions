import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
    """
    Compute Scaled Dot-Product Attention.
    
    Args:
        Q: Query matrix of shape (seq_len_q, d_k)
        K: Key matrix of shape (seq_len_k, d_k)
        V: Value matrix of shape (seq_len_k, d_v)
        mask: Optional binary mask of shape (seq_len_q, seq_len_k)
    
    Returns:
        Tuple of (output, attention_weights)
    """
    seq_len_q, d_k = Q.shape

    scores = Q @ K.T / np.sqrt(d_k)
    scores[mask == 0] = -np.inf

    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= np.sum(weights, axis=-1, keepdims=True)

    context = weights @ V
    return context, weights
