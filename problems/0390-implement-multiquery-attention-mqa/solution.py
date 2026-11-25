import numpy as np

def multiquery_attention(X: np.ndarray, W_queries: list, W_key: np.ndarray, W_value: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    """
    Compute Multi-Query Attention.
    
    Args:
        X: Input array of shape (seq_len, d_model)
        W_queries: List of query weight matrices, each (d_model, d_k), one per head
        W_key: Shared key weight matrix of shape (d_model, d_k)
        W_value: Shared value weight matrix of shape (d_model, d_v)
        W_out: Output projection matrix of shape (num_heads * d_v, d_model)
    
    Returns:
        Output array of shape (seq_len, d_model), rounded to 4 decimal places
    """
    W_query = np.stack(W_queries, axis=0)

    Q = X @ W_query
    K = X @ W_key
    V = X @ W_value

    n_queries, seq_len, d_k = Q.shape

    # for visibility, optional here
    # for batch input: K[:, np.newaxis, :, :]
    K = K[np.newaxis, :, :]
    V = V[np.newaxis, :, :]

    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= np.sum(weights, axis=-1, keepdims=True)

    context = weights @ V
    context = context.transpose(1, 0, 2).reshape(seq_len, -1)
    return np.round(context @ W_out, 4)
