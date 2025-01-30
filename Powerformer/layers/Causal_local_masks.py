from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


class CausalLocalMasks(nn.Module):
    def __init__(
        self,
        attn_decay_type=None,
        attn_decay_scale=0,
        patch_num=1,
        train_attn_decay=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mask_type = attn_decay_type
        self.mask_scale = attn_decay_scale
        self.train_mask_scale = False

        self.get_decay_mask = None
        self.decay_mask = 0
        self.times = nn.Parameter(
            torch.arange(patch_num, dtype=torch.float32).unsqueeze(1)
            - torch.arange(patch_num, dtype=torch.float32).unsqueeze(0),
            requires_grad=False,
        )

        if self.mask_type is None or self.mask_type.lower() == "none":
            self.decay_mask = torch.zeros((1))
        elif self.mask_type.lower() == "step":
            self.mask_scale = int(self.mask_scale)
            if self.mask_scale < 1:
                raise ValueError(
                    "Attention decay scale must be >= 1 for step distribution"
                )
            self.decay_mask = self._enforce_causality(
                self._step_distribution(self.times)
            )
        elif "butter" in self.mask_type.lower():
            order = int(self.mask_type[6:])
            self.decay_mask = self._enforce_causality(
                self._butterworth_filter(order, self.times)
            )
        elif self.mask_type.lower() == "powerlaw":
            self.train_mask_scale = train_attn_decay
            self.mask_scale = -1 * np.abs(self.mask_scale)
            if train_attn_decay:
                self.get_decay_mask = self._power_law_mask
            else:
                self.decay_mask = self._power_law_mask()
        elif self.mask_type.lower() == "simpowerlaw":
            self.train_mask_scale = train_attn_decay
            if train_attn_decay:
                self.get_decay_mask = self._sim_power_law_mask
            else:
                self.decay_mask = self._sim_power_law_mask()
        else:
            raise ValueError(f"Cannot handle attention decay type {self.mask_type}")

        if self.get_decay_mask is None:
            requires_grad = train_attn_decay and (
                self.mask_type is not None and self.mask_type.lower() != "step"
            )

            self.decay_mask = nn.Parameter(self.decay_mask, requires_grad=requires_grad)
            if train_attn_decay:
                self.get_decay_mask = self._train_decay_mask
            else:
                self.get_decay_mask = self._return_decay_mask

        if self.mask_type is not None:
            requires_grad = train_attn_decay and (
                'utter' not in self.mask_type and self.mask_type != 'step'
            )
            self.mask_scale = nn.Parameter(
                torch.tensor(self.mask_scale), requires_grad=requires_grad 
            )

    
    def _return_decay_mask(self):
        return self.decay_mask

    
    def _train_decay_mask(self):
        return self._enforce_causality(self.decay_mask)

    
    def _no_attn_decay(self, r_len, c_len):
        return 0

    
    def _enforce_causality(self, mask, replacement=-1 * torch.inf):
        print("SHAPES", mask.shape, self.times.shape)
        mask[self.times < -1e-10] = replacement
        return mask

    
    def _step_distribution(self, times):
        mask = torch.zeros_like(times)
        mask[torch.abs(times) > self.mask_scale] = -1 * torch.inf
        return mask

    
    def _power_law(self, times):
        return torch.abs(times) ** self.mask_scale

    
    def _sim_power_law_mask(self):
        return self._enforce_causality(-1 * self._power_law(self.times))

    
    def _power_law_mask(self):
        local_mask = torch.log(
            self._power_law(self._enforce_causality((self.times + 1), replacement=1))
        )
        return self._enforce_causality(local_mask)

    
    def _butterworth_filter(self, order, times):
        times = times.detach().numpy().astype(int)
        b, a = sp.signal.butter(order, 0.8, "lowpass", analog=False)
        t, decay = sp.signal.freqz(b, a)
        t = self.mask_scale * t / 2
        dc = 5 * np.log(np.abs(decay))
        decay_interp = sp.interpolate.interp1d(t, dc)
        mask = np.zeros(times.shape)
        for i in range(int(t[-1]) + 1):
            mask[times == i] = decay_interp(i)
        mask[times > int(t[-1])] = -np.inf

        return self._enforce_causality(torch.tensor(mask))
