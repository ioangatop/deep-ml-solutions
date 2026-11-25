def estimate_kv_cache_size(
    model_config: dict, batch_size: int, seq_len: int
) -> dict:
    num_layers = model_config["num_layers"]
    num_attention_heads = model_config["num_attention_heads"]
    num_kv_heads = model_config.get("num_kv_heads", num_attention_heads)
    dtype_bytes = model_config["dtype_bytes"]

    # Head dimension is determined by the query heads
    head_dim = model_config["hidden_size"] // num_attention_heads

    # Each layer stores both keys and values for every token
    per_token_elements = 2 * num_kv_heads * head_dim
    per_layer_elements = batch_size * seq_len * per_token_elements

    kv_cache_elements = num_layers * per_layer_elements
    kv_cache_size_bytes = kv_cache_elements * dtype_bytes

    return {
        "kv_cache_elements": kv_cache_elements,
        "kv_cache_size_bytes": kv_cache_size_bytes,
        "kv_cache_size_mb": round(kv_cache_size_bytes / 1024**2, 4),
        "per_layer_size_mb": round(
            per_layer_elements * dtype_bytes / 1024**2, 4
        ),
        "per_token_size_kb": round(
            num_layers * per_token_elements * dtype_bytes / 1024, 4
        ),
    }