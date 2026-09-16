"""Request-local shared prefill for full-attention Transformers DynamicCache models."""
import copy
import torch


def common_prefix_length(rows):
    """Leave at least one suffix token, including identical/singleton prompts."""
    if not rows or any(not row for row in rows):
        raise ValueError("Token rows must be nonempty")
    limit = min(map(len, rows)) - 1
    for i in range(limit):
        if any(row[i] != rows[0][i] for row in rows[1:]):
            return i
    return limit


def validate_cache_backend(model):
    # Keep this deliberately narrower than the model-independent public API.
    # Sliding/hybrid/recurrent caches need their own correctness verification.
    config = model.config
    config = config.get_text_config() if hasattr(config, 'get_text_config') else config
    layer_types = getattr(config, 'layer_types', None)
    if (getattr(config, 'sliding_window', None) or getattr(config, 'attention_chunk_size', None)
            or getattr(config, 'num_kv_shared_layers', 0)
            or (layer_types and any(t != 'full_attention' for t in layer_types))):
        raise ValueError('shared_prefix supports full-attention caches only; use execution="batched"')


def shared_prefix_logits(model, rows, *, device, pad_token_id, batch_size, forward):
    """Yield (row offset, final logits); no cache persists across calls.

    Prefix state is immutable after prefill. Each suffix microbatch receives its
    own deep copy, then materialized batch copies via the public cache API.
    This saves prefix compute, not prefix storage. Copy cost is included.
    """
    from transformers import DynamicCache
    validate_cache_backend(model)
    length = common_prefix_length(rows)
    if not length:
        raise ValueError('Prompts have no reusable token prefix; use execution="batched"')
    prefix_ids = torch.tensor([rows[0][:length]], dtype=torch.long, device=device)
    prefetched = forward(input_ids=prefix_ids, attention_mask=torch.ones_like(prefix_ids),
                         position_ids=torch.arange(length, device=device)[None, :], use_cache=True)
    prefix_cache = prefetched.past_key_values
    if not isinstance(prefix_cache, DynamicCache):
        raise TypeError('shared_prefix requires Transformers DynamicCache; use execution="batched"')
    del prefetched
    for start in range(0, len(rows), batch_size):
        suffixes = [row[length:] for row in rows[start:start + batch_size]]
        width = max(map(len, suffixes))
        # Left pad suffixes AFTER the prefix, mask that gap, and give real tokens
        # their logical positions. Physical cache length includes masked padding.
        suffix_ids = torch.tensor([[pad_token_id] * (width-len(s)) + s for s in suffixes],
                                  dtype=torch.long, device=device)
        suffix_mask = torch.tensor([[0] * (width-len(s)) + [1] * len(s) for s in suffixes],
                                   dtype=torch.long, device=device)
        mask = torch.cat([torch.ones((len(suffixes), length), dtype=torch.long, device=device), suffix_mask], 1)
        positions = (mask.cumsum(-1) - 1).clamp_min(0)[:, length:]
        cache = copy.deepcopy(prefix_cache)
        if len(suffixes) > 1:
            cache.batch_repeat_interleave(len(suffixes))
        output = forward(input_ids=suffix_ids, attention_mask=mask, position_ids=positions,
                         past_key_values=cache, use_cache=True)
        logits = output.logits[:, -1, :].float()
        del output, cache
        yield start, logits
