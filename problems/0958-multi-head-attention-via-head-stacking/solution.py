def multi_head_attention_stack(x, heads):
    X = np.asarray(x)
    B, T, _ = X.shape

    mask = np.triu(np.full((T, T), -np.inf), k=1)

    context_heads = []
    for W_q, W_k, W_v in heads:
        Q, K, V = X @ W_q, X @ W_k, X @ W_v
        d_head = Q.shape[-1]

        score = Q @ K.swapaxes(-2, -1) / np.sqrt(d_head)
        score += mask

        weights = np.exp(score - np.max(score, axis=-1, keepdims=True))
        weights /= np.sum(weights, axis=-1, keepdims=True)

        context = weights @ V
        context_heads.append(context)

    return np.concatenate(context_heads, axis=-1)
