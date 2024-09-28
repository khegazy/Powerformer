__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN

# Cell
class PatchTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, attn_decay_type=None, train_attn_decay=True, attn_decay_scale=0.25, record_scores=False, **kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose,
                                attn_decay_type=attn_decay_type, train_attn_decay=train_attn_decay, attn_decay_scale=attn_decay_scale, record_scores=record_scores,
                                **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
            
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]
        
        # model
        z = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)                                                                    # z: [bs x nvars x target_window] 
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
        
        
    
    
class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False,
                 attn_decay_type=None, train_attn_decay=True, attn_decay_scale=0.25, record_scores=False, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn,
                                   attn_decay_type=attn_decay_type, train_attn_decay=train_attn_decay, attn_decay_scale=attn_decay_scale, record_scores=record_scores)

        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u)                                                      # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z    
            
            
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False,
                        attn_decay_type=None, train_attn_decay=True, attn_decay_scale=0.25, record_scores=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn,
                                                      attn_decay_type=attn_decay_type, train_attn_decay=train_attn_decay,
                                                      attn_decay_scale=attn_decay_scale, name=f"_layer-{i}", record_scores=record_scores) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False,
                 attn_decay_type=None, train_attn_decay=True, attn_decay_scale=0.25, name="", record_scores=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention,
            attn_decay_type=attn_decay_type, train_attn_decay=train_attn_decay, attn_decay_scale=attn_decay_scale, name=name, record_scores=record_scores)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src




class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False,
                attn_decay_type=None, train_attn_decay=True, attn_decay_scale=0.25, name="", record_scores=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa,
                attn_decay_type=attn_decay_type, train_attn_decay=train_attn_decay, attn_decay_scale=attn_decay_scale, name=name, record_scores=record_scores)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False, attn_decay_type=None, train_attn_decay=True, attn_decay_scale=0.25, name="", record_scores=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa
        self.record_scores = record_scores
        self.name = name
        self.eval_count = 0
        self.attn_score_bins = np.linspace(-50, 50, 201)
        self.attn_score_record = np.zeros(len(self.attn_score_bins)-1)
        self.attn_mask_score_record = np.zeros(len(self.attn_score_bins)-1)
        self.attn_score_dt = []

        self.decay_attn = False
        if attn_decay_type is not None:
            self.decay_attn = True
            
            if attn_decay_type.lower() == 'step':
                self.attn_decay = self._step_distribution
                attn_decay_scale = int(attn_decay_scale)
                if attn_decay_scale < 1:
                    raise ValueError("Attention decay scale must be >= 1 for step distribution")
            elif attn_decay_type.lower() == 'zeta':
                self.attn_decay = self._zeta_distribution
            elif attn_decay_type.lower() == 'gauss':
                self.attn_decay = self._gauss_attn_decay
            elif attn_decay_type.lower() == 'tdist':
                self.attn_decay = self._tdist_attn_decay
            else:
                raise ValueError(f"Cannot handle attention decay type {attn_decay_type}")
            
            self.attn_decay_scale = nn.Parameter(torch.tensor(attn_decay_scale), requires_grad=train_attn_decay)
        else:
            self.attn_decay = self._no_attn_decay

    def _no_attn_decay(self, r_len, c_len):
        return 0

    def _step_distribution(self, r_len, c_len):
        if not self.decay_attn:
            return 0

        times = torch.arange(r_len).unsqueeze(1) - torch.arange(c_len).unsqueeze(0)
        times = times.to(self.attn_decay_scale.device).to(torch.float32)
        mask = torch.zeros_like(times)
        mask[torch.abs(times)>self.attn_decay_scale] = -1*torch.inf
        return mask


    def _zeta_distribution(self, r_len, c_len):
        if not self.decay_attn:
            return 0

        times = torch.arange(r_len).unsqueeze(1) - torch.arange(c_len).unsqueeze(0)
        times = times.to(self.attn_decay_scale.device).to(torch.float32)
        return -1*torch.abs(times)**self.attn_decay_scale

    def _gauss_attn_decay(self, r_len, c_len):
        print("comparing adding vs multiplying attn") #DEBUG
        if not self.decay_attn:
            return 0

        #print(self.attn_decay_scale)
        times = torch.arange(r_len).unsqueeze(1) - torch.arange(c_len).unsqueeze(0)
        times = times.to(self.attn_decay_scale.device).to(torch.float32)
        out = torch.exp(-0.5*(times/self.attn_decay_scale)**2).unsqueeze(0).unsqueeze(0)
        #print(times)
        #print(times/self.attn_decay_scale)
        #print(out)
        return out

    def _tdist_attn_decay(self, r_len, c_len):
        print("comparing adding vs multiplying attn") #DEBUG
        if not self.decay_attn:
            return 0

        times = torch.arange(r_len).unsqueeze(1) - torch.arange(c_len).unsqueeze(0)
        times = times.to(self.attn_decay_scale.device)
        times = times.unsqueeze(0).unsqueeze(0)
        return (1 + times**2/self.attn_decay_scale)**(-0.5*(self.attn_decay_scale + 1))


    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        #attn_scores = torch.matmul(q, k) * self.scale  * self.attn_decay(q.shape[2], k.shape[-1])    # attn_scores : [bs x n_heads x max_q_len x q_len]
        attn_scores = torch.matmul(q, k) * self.scale    # attn_scores : [bs x n_heads x max_q_len x q_len]
        #print("ATTN B4", attn_scores[0,0])
        #attn_scores = attn_scores * self.attn_decay(q.shape[2], k.shape[-1])    # attn_scores : [bs x n_heads x max_q_len x q_len]
        #print("ATTN after", attn_scores[0,0])

        if self.record_scores:
            self.raw_scores = attn_scores

        plot_hists = False
        if plot_hists:
            np_scores = torch.flatten(attn_scores).detach().cpu().numpy()
            #print("MIN MAX", np.mean(np_scores), np.std(np_scores), np.amin(np_scores), np.amax(np_scores))
            #self.attn_score_record = self.attn_score_record + np.histogram(np_scores, self.attn_score_bins)[0]
            self.attn_score_record = np.histogram(np_scores, self.attn_score_bins)[0]
            """
            fig, ax = plt.subplots()
            ax.bar((self.attn_score_bins[1:]+self.attn_score_bins[:-1])/2, self.attn_score_record)
            #ax.set_xscale('log')
            ax.set_yscale('log')
            fig.savefig("attn_score_hist.png")
            """

            dt_masks = []
            dts = torch.arange(len(attn_scores), dtype=int)
            dts = dts[None,:] - dts[:,None]
            for i in range(len(attn_scores)):
                dt_masks.append(torch.abs(dts) == i)
                dt_hist = np.histogram(
                    attn_scores[dt_masks[i]].detach().cpu().numpy()
                )
                if self.attn_score_dt <= i:
                    self.attn_score_dt.append(dt_hist[:])
                else:
                    self.attn_score_dt[i] = dt_hist
                fig, ax = plt.subplots()
                ax.bar((self.attn_score_bins[1:]+self.attn_score_bins[:-1])/2, self.attn_score_dt[i])
                fig.savefig(f"attn_score_hist_dt-{i}"+self.name+".png")


            

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        if self.record_scores:
            self.raw_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        # Power law attention decay 
        attn_decay = self.attn_decay(q.shape[2], k.shape[-1])
        #print(attn_decay)
        if False:
            print("mask", torch.mean(attn_decay), torch.std(attn_decay), torch.amin(attn_decay), torch.amax(attn_decay))
            plotme = attn_decay.detach().cpu().numpy()
            fig, ax = plt.subplots()
            ax.plot(plotme[0])
            ax.plot(plotme[7])
            ax.plot(plotme[14])
            ax.plot(plotme[28])
            ax.plot(plotme[35])
            ax.plot(plotme[41])
            #ax.set_yscale('log')
            fig.savefig("test_decay.png")
        attn_scores = attn_scores + attn_decay
        if self.record_scores:
            self.masked_scores = attn_scores
            self.powerlaw_mask = attn_decay
        
        if plot_hists:
            np_scores = torch.flatten(attn_scores).detach().cpu().numpy()
            print("MIN MAX", np.mean(np_scores), np.std(np_scores), np.amin(np_scores), np.amax(np_scores))
            #self.attn_mask_score_record = self.attn_mask_score_record + np.histogram(np_scores, self.attn_score_bins)[0]
            self.attn_mask_score_record = np.histogram(np_scores, self.attn_score_bins)[0]
            fig, ax = plt.subplots()
            ax.bar((self.attn_score_bins[1:]+self.attn_score_bins[:-1])/2, self.attn_score_record, color='teal')
            ax.bar((self.attn_score_bins[1:]+self.attn_score_bins[:-1])/2, self.attn_mask_score_record, alpha=0.4, color='black')
            #ax.set_xscale('log')
            ax.set_yscale('log')
            fig.savefig("attn_score_hist"+self.name+".png")


        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        #print("SCORES", attn_scores[0,0])
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        if self.record_scores:
            self.attn_weights = attn_weights
        #attn_weights = attn_weights*self.attn_decay(q.shape[2], k.shape[-1])
        #attn_weights = attn_weights/torch.sum(attn_weights, -1, keepdim=True)
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        #print("WEIGHTS", attn_weights[0,0])
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

