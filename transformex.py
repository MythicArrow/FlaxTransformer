import jax
import jax.numpy as jnp
from flax import nnx
import optax
from typing import Optional

# FlashAttention (with Grouped Query Attention)
class FlashAttention(nnx.Module):
    num_heads: int
    head_dim: int
    d_model: int
    use_gqa: bool = True  # Enables Grouped Query Attention

    def __init__(self, num_heads, d_model, use_gqa=True):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.use_gqa = use_gqa
        self.w_q = nnx.Dense(d_model)
        self.w_k = nnx.Dense(d_model)
        self.w_v = nnx.Dense(d_model)
        self.w_out = nnx.Dense(d_model)

    def __call__(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        num_kv_heads = self.num_heads if not self.use_gqa else max(1, self.num_heads // 2)

        q = self.w_q(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.w_k(x).reshape(batch_size, seq_len, num_kv_heads, self.head_dim)
        v = self.w_v(x).reshape(batch_size, seq_len, num_kv_heads, self.head_dim)

        q = q.transpose(0, 2, 1, 3)  # (B, H, S, D)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        attn_weights = jnp.einsum('bhqd, bhkd -> bhqk', q, k) / jnp.sqrt(self.head_dim)
        if mask is not None:
            attn_weights += (mask * -1e9)

        attn_probs = jax.nn.softmax(attn_weights, axis=-1)
        attn_output = jnp.einsum('bhqk, bhvd -> bhqd', attn_probs, v)

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        return self.w_out(attn_output)


# RMSNorm (Replaces LayerNorm)
class RMSNorm(nnx.Module):
    dim: int
    eps: float = 1e-6

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nnx.Param(jnp.ones((dim,)))

    def __call__(self, x):
        norm = x * jax.lax.rsqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return norm * self.scale


# SwiGLU Feedforward Network (FFN)
class SwiGLUFeedForward(nnx.Module):
    d_model: int
    d_ff: int

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nnx.Dense(d_model, d_ff)
        self.w2 = nnx.Dense(d_ff, d_model)
        self.gate = nnx.Dense(d_model, d_ff)

    def __call__(self, x):
        return self.w2(jax.nn.silu(self.w1(x)) * self.gate(x))


# Transformer Block with √0.5 scaling and NormFormer
class TransformerBlock(nnx.Module):
    d_model: int
    num_heads: int
    d_ff: int

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = FlashAttention(num_heads, d_model)
        self.ffn = SwiGLUFeedForward(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def __call__(self, x, mask=None):
        # Attention block with scaling
        x = x + jnp.sqrt(0.5) * self.attn(self.norm1(x), mask)
        # Feed-forward block with scaling
        x = x + jnp.sqrt(0.5) * self.ffn(self.norm2(x))
        return x


# Full Transformer Model with Embeddings
class FlaxTransformer(nnx.Module):
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int
    vocab_size: int

    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size):
        super().__init__()
        self.embedding = nnx.Embed(vocab_size, d_model)
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.norm = RMSNorm(d_model)

    def __call__(self, x, mask=None):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


# Optimizer & Training Utilities
def create_optimizer():
    return optax.adamw(learning_rate=1e-4, weight_decay=1e-5)

# Sample mask for padding tokens
def create_padding_mask(seq_len, pad_token=0):
    return (seq_len != pad_token).astype(jnp.float32)
#Hyperparameters
num_layers = 8
d_model = 6
num_heads = 12
d_ff = 4
vocab_size = 4000
model = FlaxTransformer(num_layers, d_model, num_heads, d_ff, vocab_size)
