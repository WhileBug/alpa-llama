import functools
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.lax as lax
import jax.numpy as jnp
from einops import rearrange

def blockwise_ffn(remat_ffn, inputs, chunk_size=2048, deterministic=True):
    # remat_ffn: a rematerialized ffn with policy jax.checkpoint_policies.nothing_saveable()
    # inputs: (batch, seq_len, dim)
    # chunk_size: the chunk size to split the sequence
    inputs = rearrange(inputs, 'b (c n) d -> b c n d', c=chunk_size)
    def scan_ffn(remat_ffn, carry, hidden_states):
        outputs = remat_ffn(hidden_states, deterministic=deterministic)
        return carry, outputs
    scan_axis = inputs.ndim - 2
    _, res = nn.scan(
        scan_ffn,
        variable_broadcast="params",
        split_rngs={"params": False, "dropout": True},
        in_axes=scan_axis,
        out_axes=scan_axis,
    )(remat_ffn, None, inputs)
    res = rearrange(res, 'b c n d -> b (c n) d')
    return res

"""
Compute attention blockwise without materializing the full attention matrix,
initially proposed in memory-efficient transformer https://arxiv.org/abs/2112.05682 Rabe et al. 2021;
flash attention https://arxiv.org/abs/2205.14135 Dao et al. 2022 proposes a CUDA
efficient implementation; blockwise parallel transformer https://arxiv.org/abs/2305.19370
Liu et al. 2023 proposes blockwise computing both attention and FFN, enabling 4x
longer sequences than memory-efficient/flash-attention and fusion of attention and FFN.
"""
def blockwise_attn(
        query, key, value,
        bias=None,
        deterministic=True,
        dropout_rng=None,
        attn_pdrop=0.0,
        causal=True,
        query_chunk_size=2048,
        key_chunk_size=2048,
        dtype=jnp.float32,
        policy=jax.checkpoint_policies.nothing_saveable(),
        precision=None,
        float32_logits=True,
        prevent_cse=True,
    ):
    # query, key, value: (batch, seq_len, num_heads, dim_per_head)
    # bias: (batch, seq_len) can be used to mask out attention (e.g. padding)
    # causal: whether to use causal mask
    # policy: one of jax.checkpoint_policies
    query = query / jnp.sqrt(query.shape[-1]).astype(dtype)
    if float32_logits:
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)

    batch, q_len, num_heads, dim_per_head = query.shape
    batch, kv_len, num_heads, dim_per_head = key.shape
    batch, kv_len, num_heads, dim_per_head = value.shape
        num_q = q_len // query_chunk_size
    num_kv = kv_len // key_chunk_size
    query = query.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    key = key.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    value = value.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))

    query = jnp.moveaxis(query, 1, 0)
    key = jnp.moveaxis(key, 1, 0)
    value = jnp.moveaxis(value, 1, 0)

    if bias is not None:
        for bias_dim, broadcast_dim in zip(bias.shape, (batch, num_heads, q_len, kv_len)):
            assert bias_dim == 1 or bias_dim == broadcast_dim
    if not deterministic and attn_pdrop > 0.0:
        attn_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
        attn_dropout = jax.random.bernoulli(attn_dropout_rng, attn_pdrop, (batch, num_heads, q_len, kv_len))
    else:
        attn_dropout = None

    _chunk_bias_fn = functools.partial(
        _chunk_attention_bias,
        query_chunk_size, key_chunk_size, bias, deterministic,
        attn_dropout, attn_pdrop, causal, dtype)

    def scan_attention(args):
        query_chunk, query_chunk_idx = args

        @functools.partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
        def scan_kv_block(carry, args):
            key_chunk, value_chunk, key_chunk_idx = args
            (numerator, denominator, prev_max_score) = carry
            attn_weights = jnp.einsum('bqhd,bkhd->bqhk', query_chunk, key_chunk, precision=precision)
            bias_chunk = _chunk_bias_fn(query_chunk_idx, key_chunk_idx)
            bias_chunk = jnp.moveaxis(bias_chunk, 1, 2)
            attn_weights = attn_weights + bias_chunk

            max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
            max_score = jnp.maximum(prev_max_score, max_score)
            max_score = jax.lax.stop_gradient(max_score)
            exp_weights = jnp.exp(attn_weights - max_score)
            exp_values = jnp.einsum(
                'bqhv,bvhd->bqhd', exp_weights, value_chunk, precision=precision
            )
            correction = jnp.exp(prev_max_score - max_score)
            numerator = numerator * correction + exp_values
            denominator = denominator * correction + exp_weights.sum(axis=-1, keepdims=True)
            return Carry(numerator, denominator, max_score), None

        def skip_upper_half(carry, args):
            key_chunk, value_chunk, key_chunk_idx = args
            skip_block = jnp.array(False)
            if causal:
                skip_block = query_chunk_idx < key_chunk_idx
            return jax.lax.cond(
                skip_block,
                lambda carry, args: (carry, None),
                scan_kv_block,
                carry,
                args,
            )

        init_carry = Carry(
            jnp.zeros((batch, query_chunk_size, num_heads, dim_per_head), dtype=query.dtype),
            jnp.zeros((batch, query_chunk_size, num_heads, dim_per_head), dtype=query.dtype),
            (-jnp.inf) * jnp.ones((batch, query_chunk_size, num_heads, 1), dtype=query.dtype),
        )
        (numerator, denominator, max_score), _ = lax.scan(
            skip_upper_half, init_carry, xs=(key, value, jnp.arange(0, num_kv))
        )
        outputs = (numerator / denominator).astype(dtype)
        return outputs

    _, res = lax.scan(
        lambda _, x: ((), scan_attention(x)),
        (), xs=(query, jnp.arange(0, num_q))
    )
    res = rearrange(res, 'n b c h d -> b (n c) h d')
    return res

def get_gradient_checkpoint_policy(name):
    return {
        'everything_saveable': jax.checkpoint_policies.everything_saveable,
        'nothing_saveable': jax.checkpoint_policies.nothing_saveable,
        'checkpoint_dots': jax.checkpoint_policies.checkpoint_dots,
        'checkpoint_dots_with_no_batch_dims': jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
    }[name]