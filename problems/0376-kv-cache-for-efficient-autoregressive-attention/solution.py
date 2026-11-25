import numpy as np

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    d_model, d_k = W_Q.shape
    
    q = x_new @ W_Q  # (d_k,)
    k = x_new @ W_K  # (d_k,)
    v = x_new @ W_V  # (d_k,)

    k = k[np.newaxis, :]
    v = v[np.newaxis, :]
    if cache:
        K, V = cache
        K = np.concatenate([K, k], axis=0)
        V = np.concatenate([V, v], axis=0)
    else:
        K, V = k, v

    cache = (K, V)

    scores = q @ K.T / np.sqrt(d_k)
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= np.sum(weights, axis=-1, keepdims=True)

    context = weights @ V
    return context, cache
