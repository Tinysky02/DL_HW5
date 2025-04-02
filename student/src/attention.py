"""
Originally forked from Andrej Karpathy's minGPT,
Modified based on Stanford CS224N 2023-24: Homework 4

John Hewitt <johnhew@stanford.edu>
Ansh Khurana <anshk@stanford.edu>
Soumya Chatterjee <soumyac@stanford.edu>
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def precompute_rotary_emb(dim, max_positions):
    """
    RoPE uses the following sinusoidal functions to encode positions:

    cos(t theta_i) and sin(t theta_i)
        where t is the position and
              theta_i = 1/10000^(-2(i-1)/dim) for i in [1, dim/2]

    Since the maximum length of sequences is known, we can precompute
    these values to speed up training.

    Implement the precompute_rotary_emb function that returns a tensor of
    shape (max_positions, dim/2, 2) where the last dimension contains
    the cos and sin values for each position and each dimension of
    the embedding.
    """

    
    # TODO: [part g]
    ### YOUR CODE HERE ###
    half_dim = dim // 2
    i = torch.arange(half_dim, dtype=torch.float)
    alpha = 1.0 / (10000.0 ** (2 * i / dim))
    pos = torch.arange(max_positions, dtype=torch.float).unsqueeze(1) 
    pos_alpha = pos * alpha 
    cos_vals = torch.cos(pos_alpha)
    sin_vals = torch.sin(pos_alpha) 
    rope_cache = torch.stack([cos_vals, sin_vals], dim=-1)
    ### END YOUR CODE ###
    return rope_cache


def apply_rotary_emb(x, rope_cache):
    """Apply the RoPE to the input tensor x."""
    # TODO: [part g]
    # You might find the following functions useful to convert
    # between real and complex numbers:

    # torch.view_as_real - https://pytorch.org/docs/stable/generated/torch.view_as_real.html
    # torch.view_as_complex - https://pytorch.org/docs/stable/generated/torch.view_as_complex.html

    # Note that during inference, the length of the sequence might be different
    # from the length of the precomputed values. In this case, you should use
    # truncate the precomputed values to match the length of the sequence.

    ### YOUR CODE HERE ###
    B, n_head, T, D = x.shape
    half = D // 2

    x_real = x[..., :half]   
    x_imag = x[..., half:]   
    x_complex = torch.view_as_complex(torch.stack((x_real, x_imag), dim=-1))  # (B, n_head, T, half), complex

    cos_sin = rope_cache[:T] 
    cos_vals = cos_sin[..., 0]  
    sin_vals = cos_sin[..., 1] 
    cos_vals = cos_vals.unsqueeze(0).unsqueeze(0)
    sin_vals = sin_vals.unsqueeze(0).unsqueeze(0)

    rope_complex = torch.view_as_complex(torch.stack((cos_vals, sin_vals), dim=-1))

    x_rotated_complex = x_complex * rope_complex

    x_rotated = torch.view_as_real(x_rotated_complex)  
    rotated_x = torch.cat([x_rotated[..., 0], x_rotated[..., 1]], dim=-1)
    ### END YOUR CODE ###
    return rotated_x

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.rope = config.rope
        if self.rope:
            assert (config.n_embd % config.n_head) % 2 == 0

            # TODO: [part g] Precompute the cos and sin values for RoPE and
            # store them in rope_cache.
            # Hint: The maximum sequence length is given by config.block_size.

            ### YOUR CODE HERE ###
            self.rope_cache = precompute_rotary_emb(
                dim=(config.n_embd // config.n_head), 
                max_positions=config.block_size
            )
            ### END YOUR CODE ###

        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # TODO: causal mask to ensure that attention is only applied to the left in the input sequence

        ### YOUR CODE HERE ###
        block_size = config.block_size
        mask = torch.tril(torch.ones(block_size, block_size))
        mask = mask.view(1, 1, block_size, block_size)
        ### END YOUR CODE ###

        self.register_buffer("mask", mask)
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.rope:
            # TODO: [part g] Apply RoPE to the query and key.
            ### YOUR CODE HERE ###
            q = apply_rotary_emb(q, self.rope_cache)
            k = apply_rotary_emb(k, self.rope_cache)
            ### END YOUR CODE ###

        # TODO: causal self-attention
        # 1. compute attention map (pre-softmax)
        # 2. apply attention mask to the attention map
        # 3. apply softmax to the attention map (hint: masked parts should not be included in the softmax)
        # 4. apply attention dropout to the attention map
        # 5. compute the output by applying the attention map to the value
        # 6. re-assemble all head outputs side by side
        ### YOUR CODE HERE ###
        d_k = k.size(-1)
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) 
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = torch.matmul(att, v)  
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        ### END YOUR CODE ###

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
