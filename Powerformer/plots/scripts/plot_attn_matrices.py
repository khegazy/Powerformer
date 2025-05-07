import os
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch
from torch import nn
import cmocean


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
        elif self.mask_type.lower() == "causal":
            self.decay_mask = self._enforce_causality(
                torch.zeros_like(self.times)
            )
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


def plot(data, v_max, v_min, name, ilr, ivar, idx, cmap='seismic'):
    fig, ax = plt.subplots(figsize=(5,5), gridspec_kw={'left': 0.01, 'right' : 0.99, 'top' : 0.99, 'bottom' : 0.01 })
    ax.imshow(data, cmap=cmap, vmin=v_min, vmax=v_max)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    fig.tight_layout()
    if 'mask' in name:
        fig.savefig(
            os.path.join(save_dir, f"{name}.png")
        )
    else:
        fig.savefig(
            os.path.join(save_dir, f"total_{name}_lyr{ilr}_var{ivar}_idx{idx}.png")
        )

if __name__ == '__main__':
    model_names = [
        'Weather_Powerformer_custom_ftM_sl336_ll48_pl720_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_atpowerLaw_'
    ]
    decay_times = ['0.1']

    all_masks = {}
    all_mask_labels = [
        ('powerLaw', 0.1), ('powerLaw', 0.25), ('powerLaw', 1.0),
        ('simPowerLaw', 0.1), ('simPowerLaw', 0.5), ('simPowerLaw', 1.0),
        ('butter1', 5), ('butter1', 20),
        ('butter2', 5), ('butter2', 20),
        ]

    for name in model_names:
        for decay_time in decay_times:
            folder = os.path.join('../', '..', 'results', name + decay_time + "_Exp_0")
            with open(os.path.join(folder, 'attn_matrices_indices.npy'), 'rb') as file:
                indices = np.load(file)
            with open(os.path.join(folder, 'decay_mask.npy'), 'rb') as file:
                decay_mask = np.load(file)
            for mask_name, alpha in all_mask_labels:
                clm = CausalLocalMasks(mask_name, alpha, len(decay_mask))
                all_masks[(mask_name, alpha)] = clm.get_decay_mask()
            save_dir = os.path.join("..", name + decay_time + "_Exp_0")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print("MASK", np.amax(decay_mask))
            #plot(decay_mask, 0, -4, 'decay_mask', None, None, None, cmap='Reds')
            plot(decay_mask, 0, -4, 'decay_mask', None, None, None, cmap=cmocean.cm.balance)
            for (mn, ma), msk in all_masks.items():
                plot(msk, 0, -4, f"calculated_mask_{mn}_{ma}", None, None, None, cmap='Reds')
            raw_scores, scores = [], []
            raw_weights, weights = [], []
            for i in range(3):
                if os.path.exists(os.path.join(folder, f"total_weights_{i}.npy")):
                    with open(os.path.join(folder, f"total_raw_scores_{i}.npy"), 'rb') as file:
                        raw_scores.append(np.mean(np.load(file), axis=1))
                    with open(os.path.join(folder, f'total_raw_weights_{i}.npy'), 'rb') as file:
                        raw_weights.append(np.mean(np.load(file), axis=1))
                    with open(os.path.join(folder, f'total_scores_{i}.npy'), 'rb') as file:
                        scores.append(np.mean(np.load(file), axis=1))
                    with open(os.path.join(folder, f'total_weights_{i}.npy'), 'rb') as file:
                        weights.append(np.mean(np.load(file), axis=1))
            for ilr in range(1):
                n_vars = int(len(weights[ilr])//len(indices))
                plot_idxs = [904, 1824, indices[23]]
                for plot_name, data in [('raw_score', raw_scores), ('raw_weight', raw_weights), ('score', scores), ('weight', weights)]:
                    #if plot_name != 'weight':
                    #    continue
                    for i, idx in enumerate(plot_idxs):
                        for ivar in range(n_vars):
                            if ivar != 4:
                                continue
                            plot_data = data[ilr][i*n_vars+ivar]
                            if 'score' == plot_name:
                                v_max = 0
                                v_min = -3.5
                                #v_min = -1.0*v_max
                                #if 'score' == plot_name:
                                #    plot_data[plot_data<-1e10] = -1e10
                                cmap = 'Reds'
                                #plot_data = plot_data + v_max*1.1
                            else:
                                v_rng = np.max([np.amax(plot_data), np.abs(np.amin(plot_data))])
                                cmap = 'seismic'
                                cmap = cmocean.cm.balance
                                if plot_name == 'weight':
                                    v_rng = 0.4*v_rng
                                    #v_min = -0.0
                                    #cmap = 'Reds'
                                v_max = v_rng
                                v_min = -1*v_rng
                            

                            plot(plot_data, v_max, v_min, plot_name, ilr, ivar, idx, cmap=cmap)
                        
                        if i > 10:
                            break
                print(v_min, v_max)
                v_max = 0.3
                v_min = -1*v_max
                for i, idx in enumerate(plot_idxs):
                    score = torch.tensor(raw_scores[ilr][i*n_vars+ivar])
                    attn = torch.softmax(score, dim=-1)
                    plot_name = f"calculated_attn_noncausal"
                    plot(attn, v_max, v_min, plot_name, ilr, ivar, idx, cmap=cmocean.cm.balance)
                    for (mn, ma), mask in all_masks.items():
                        for ivar in range(n_vars):
                            if ivar != 4:
                                continue
                            #print(mask)
                            #print(raw_scores[ilr][i*n_vars+ivar])
                            masked_score = torch.tensor(raw_scores[ilr][i*n_vars+ivar]) + mask
                            masked_attn = torch.softmax(masked_score, dim=-1)
                            plot_name = f"calculated_attn_{mn}_{ma}"
                            plot(masked_attn, v_max, v_min, plot_name, ilr, ivar, idx, cmap=cmocean.cm.balance)



