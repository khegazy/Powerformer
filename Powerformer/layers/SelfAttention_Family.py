import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import math
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
import os


class CausalLocalMasks(nn.Module):
    def __init__(self, attn_decay_type=None, attn_decay_scale=0, patch_num=1, train_attn_decay=False, **kwargs) -> None:
        super().__init__()
        self.mask_type = attn_decay_type
        self.mask_scale = attn_decay_scale
        self.train_mask_scale = False

        self.get_decay_mask = None
        self.decay_mask = 0
        self.times = nn.Parameter(
            torch.arange(patch_num, dtype=torch.float32).unsqueeze(1)\
                - torch.arange(patch_num, dtype=torch.float32).unsqueeze(0),
            requires_grad=False
        )

        if self.mask_type is None:
            self.decay_mask = torch.zeros((1))
        elif self.mask_type.lower() == 'causal':
            self.decay_mask = torch.zeros((self.patch_num, self.patch_num))
        elif self.mask_type.lower() == 'step':
            self.mask_scale = int(self.mask_scale)
            if self.mask_scale[0] < 1:
                raise ValueError("Attention decay scale must be >= 1 for step distribution")
            self.decay_mask = self._enforce_causality(self._step_distribution(self.times))
        elif 'butter' in self.mask_type.lower():
            order=int(self.mask_type[6:])
            self.decay_mask = self._enforce_causality(
                self._butterworth_filter(order, self.times)
            )
        elif self.mask_type.lower() == 'powerlaw':
            self.train_mask_scale = train_attn_decay
            self.mask_scale = -1*np.abs(self.mask_scale)
            if train_attn_decay:
                self.get_decay_mask = self._power_law_mask
            else:
                self.decay_mask = self._power_law_mask()
        elif self.mask_type.lower() == 'simpowerlaw':
            self.train_mask_scale = train_attn_decay
            if train_attn_decay:
                self.get_decay_mask = self._sim_power_law_mask
            else:
                self.decay_mask = self._sim_power_law_mask()
        elif self.mask_type.lower() == 'gauss':
            self.decay_mask = self._enforce_causality(
                self._gauss_attn_decay(self.mask_scale, self.times)
            )
        elif self.mask_type.lower() == 'tdist':
            self.decay_mask = self._enforce_causality(
                self._tdist_attn_decay(self.mask_scale, self.times)
            )
        else:
            raise ValueError(f"Cannot handle attention decay type {self.mask_type}")
            
        #if self.mask_type is not None and not train_attn_decay:
        
        if self.get_decay_mask is None:
            requires_grad = train_attn_decay and (self.mask_type is not None and self.mask_type.lower() == "causal")

            self.decay_mask = nn.Parameter(
                self.decay_mask, requires_grad=requires_grad
            )
            if train_attn_decay:
                self.get_decay_mask = self._train_decay_mask
            else:
                self.get_decay_mask = self._return_decay_mask
        
        if self.mask_type is not None:
            self.mask_scale = nn.Parameter(
                torch.tensor(self.mask_scale), requires_grad=requires_grad
            )
 

    def _return_decay_mask(self):
        return self.decay_mask

    def _train_decay_mask(self):
        return self._enforce_causality(self.decay_mask)
    
    def _no_attn_decay(self, r_len, c_len):
        return 0

    def _enforce_causality(self, mask, replacement=-1*torch.inf):
        mask[self.times < -1e-10] = replacement
        return mask

    def _step_distribution(self, times):
        mask = torch.zeros_like(times)
        mask[torch.abs(times)>self.mask_scale] = -1*torch.inf
        return self._enforce_causality(mask, times)

    def _power_law(self, times):
        return torch.abs(times)**self.mask_scale
    
    def _sim_power_law_mask(self):
        return self._enforce_causality(
            -1*self._power_law(self.times)
        )

    def _power_law_mask(self):
        local_mask = torch.log(
            self._power_law(
                self._enforce_causality((self.times+1), replacement=1)
            )
        )
        return self._enforce_causality(local_mask)
    
    def _butterworth_filter(self, order, times):
        times = times.detach().numpy().astype(int)
        b, a = sp.signal.butter(order, 0.8, 'lowpass', analog=False)
        t, decay = sp.signal.freqz(b, a)
        t = self.mask_scale*t/2
        dc = 5*np.log(np.abs(decay))
        decay_interp = sp.interpolate.interp1d(t, dc)
        mask = np.zeros(times.shape)
        for i in range(int(t[-1])+1):
            mask[times == i] = decay_interp(i)
        mask[times > int(t[-1])] = -np.inf

        return self._enforce_causality(torch.tensor(mask))

    def _gauss_attn_decay(self):
        print("comparing adding vs multiplying attn") #DEBUG

        mask = 1 - torch.exp(-0.5*(self.times/self.mask_scale)**2)
        mask = mask.unsqueeze(0).unsqueeze(0)
        return self._enforce_causality(mask, self.times)

    def _tdist_attn_decay(self):
        print("comparing adding vs multiplying attn") #DEBUG
        mask = 1 - (1 + self.times**2/self.mask_scale)**(-0.5*(self.mask_scale + 1))
        return self._enforce_causality(mask, self.times)

        self.alpha = -1*torch.tensor(np.abs(alpha))
        self.scale_weight = nn.Parameter(torch.zeros(1))


class FullAttention(CausalLocalMasks):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                attn_decay_type=None, attn_decay_scale=0, patch_num=1, train_attn_decay=False):
        super(FullAttention, self).__init__(
            attn_decay_type=attn_decay_type, attn_decay_scale=attn_decay_scale, patch_num=patch_num, train_attn_decay=train_attn_decay
        )
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.do_dropout = self.mask_type is None

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)
        
        decay_mask = self.get_decay_mask().to(scores.device)
        scores = scale*scores + decay_mask

        A = torch.softmax(scores, dim=-1)
        if self.do_dropout:
            A = self.dropout(A)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
