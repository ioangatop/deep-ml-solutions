import numpy as np

def moe(x: np.ndarray, We: np.ndarray, Wg: np.ndarray, n_experts: int, top_k: int) -> np.ndarray:
    """
    Args:
        x: Input tensor of shape (n_batch, l_seq, d_model)
        We: Expert weights of shape (n_experts, d_model, d_model)
        Wg: Gating weights of shape (d_model, n_experts)
        n_experts: Number of experts
        top_k: Number of experts to route each token to
    Returns:
        Output tensor of shape (n_batch, l_seq, d_model)
    """
    # Calculate routing logits
    # (B, T, D) @ (D, E) -> (B, T, E)
    gate = x @ Wg

    # Mask non-selected experts
    # (B, T, E) -> (B, T, E-K)
    partitioned_experts = np.argpartition(gate, -top_k, axis=-1)
    masked_experts = partitioned_experts[..., :-top_k]
    np.put_along_axis(gate, masked_experts, -np.inf, axis=-1)

    # Calculate routing probabilities: (B, T, E)
    scores = np.exp(gate - np.max(gate, axis=-1, keepdims=True))
    scores /= np.sum(scores, axis=-1, keepdims=True)

    # Accumulate weighted expert outputs: (B, T, D)
    output = np.zeros_like(x)

    for e in range(n_experts):
        # Identify tokens routed to this expert: (B, T)
        mask = scores[..., e] > 0

        # Transform selected tokens: (N, D) @ (D, D) -> (N, D)
        expert_output = x[mask] @ We[e]

        # Extract routing probabilities: (N,)
        expert_weights = scores[..., e][mask]

        # Weight and accumulate: (N, D) * (N, 1) -> (N, D)
        output[mask] += expert_output * expert_weights[:, None]

    return output
