import numpy as np

def batched_causal_self_attention(X, W_query, W_key, W_value):
    """
    Batched causal self-attention.

    Args:
        X: nested list of shape (B, T, d_in)
        W_query, W_key, W_value: nested lists of shape (d_in, d_out)

    Returns:
        Nested list of shape (B, T, d_out) -- the context vectors.
    """

    X = np.asarray(X)
    B, T, _ = X.shape

    mask = np.triu(np.full((T, T), -np.inf), k=1)

    Q = X @ W_query
    K = X @ W_key
    V = X @ W_value

    d_out = Q.shape[-1]

    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_out)
    scores += mask

    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= np.sum(weights, axis=-1, keepdims=True)

    context = weights @ V
    return context
