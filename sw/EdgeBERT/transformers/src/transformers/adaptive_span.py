import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveSpan(nn.Module):
    def __init__(
        self,
        adapt_span_enabled,
        attn_span,
        adapt_span_loss_coeff,
        adapt_span_ramp,
        adapt_span_init,
        adapt_span_cache,
        nb_heads,
        bs,
        mask_size,
    ):

        super(AdaptiveSpan, self).__init__()
        self.attn_span = attn_span  # [attn_span]
        self.ramp_size = adapt_span_ramp
        self.bs = bs
        self.nb_heads = nb_heads
        self.init_val = adapt_span_init
        self.adapt_cache = adapt_span_cache
        self.loss_coeff = adapt_span_loss_coeff
        self.shape = (self.bs, self.nb_heads, 1, 1)

        self.current_val = nn.Parameter(
            torch.nn.init.kaiming_normal_(torch.empty(*self.shape)) + self.init_val
        )  # [bs,nb_heads,1,1]
        self.mask_size = mask_size

        mask_template_0 = torch.linspace(
            1 - self.mask_size[0], 0, steps=self.mask_size[0]
        )  # [attn_span]
        mask_template_1 = torch.linspace(
            1 - self.mask_size[1], 0, steps=self.mask_size[1]
        )
        self.register_buffer("mask_template_0", mask_template_0)
        self.register_buffer("mask_template_1", mask_template_1)

    def mask_forward(self, x):
        mask_size = x.size(3)
        if mask_size == self.mask_size[0]:
            mask = self.mask_template_0 + self.current_val * mask_size
        else:
            mask = self.mask_template_1 + self.current_val * mask_size
        mask = mask / self.ramp_size + 1
        mask = mask.clamp(0, 1)

        #print ("x.size(): ", x.size())
        #print ("mask.size(): ", mask.size())

        if x.size(0) == mask.size(0):
            x = x * mask  # [bs, nb_heads, 36, 64]) [bs, nb_heads, 1, 64]
            #print ("masked x.size(): ", x.size())
            return x
        else:
            return x

    def get_current_avg_span(self, include_ramp=True):
        current_size = math.ceil(self.current_val.mean().item() * self.attn_span)
        if include_ramp:
            current_size += self.ramp_size
        current_size = max(0, min(self.attn_span, current_size))
        return current_size

    def get_current_max_span(self, include_ramp=True):
        current_size = math.ceil(self.current_val.max().item() * self.attn_span)
        if include_ramp:
            current_size += self.ramp_size
        current_size = max(0, min(self.attn_span, current_size))
        return current_size

    def clamp_param(self):
        self.current_val.data.clamp_(0, 1)

    def get_trim_len(self):
        L = self.attn_span
        trim_len = min(L - 1, L - self.get_current_max_span())
        trim_len = math.floor(trim_len / 64) * 64
        return trim_len

    def trim_memory(self, query, key, value, key_pe):
        """trim out unnecessary memory beforehand to reduce computation"""
        trim_len = self.get_trim_len()
        cache_size = key.size(1) - query.size(1)
        trim_len_cache = trim_len - (self.attn_span - cache_size)
        if trim_len_cache > 0:
            key = key[:, trim_len_cache:, :]
            value = value[:, trim_len_cache:, :]
        elif trim_len_cache < 0:
            key = F.pad(key, [0, 0, -trim_len_cache, 0])
            value = F.pad(value, [0, 0, -trim_len_cache, 0])
        if trim_len > 0:
            if key_pe is not None:
                key_pe = key_pe[:, :, trim_len:]
        return key, value, key_pe

    def get_cache_size(self):
        """determine how long the cache should be"""
        if self.adapt_cache:
            trim_len = self.get_trim_len()
            # give a buffer of 64 steps since a span might increase
            # in future updates
            return min(self.attn_span, self.attn_span - trim_len + 64)
        else:
            return self.attn_span

    def get_loss(self):
        """a loss term for regularizing the span length"""
        return self.loss_coeff * self.attn_span * self.current_val.mean()

    def forward(self, attn):
        attn = self.mask_forward(attn)
        attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)
        return attn
