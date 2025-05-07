import torch
from torch import nn
import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


pl_color_dict = {
    0.0 : 'k',
    0.1 : 'r',
    0.25 : 'yellow',
    0.5 : 'g',
    0.75 : 'teal',
    1.0 : 'blue',
    2.0 : 'purple'
}

spl_color_dict = {
    0.0 : 'k',
    0.1 : 'r',
    0.5 : 'yellow',
    1.0 : 'teal',
    2.0 : 'blue',
}

bw_color_dict = {
    20. : 'r',
    15. : 'yellow',
    10. : 'g',
    5. : 'b',
    2. : 'purple'
}

def butterworth_filter(times, scale, order):
    times = times.detach().numpy()
    b, a = sp.signal.butter(order, 0.8, 'lowpass', analog=False)
    t, decay = sp.signal.freqz(b, a)
    t = scale*t/2
    dc = 5*np.log(np.abs(decay))
    decay_interp = sp.interpolate.interp1d(t, dc)
    filter = -np.inf*np.ones_like(times)
    mask = times<t[-1] 
    filter[mask] = decay_interp(times[mask])
    return times[mask], filter[mask]

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

        if self.mask_type is None or self.mask_type.lower() == 'none':
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
            self.decay_mask = self._butterworth_filter(order, self.times)
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
        
        if self.get_decay_mask is None and False:
            requires_grad = train_attn_decay and (self.mask_type is not None and self.mask_type.lower() == "causal")

            self.decay_mask = nn.Parameter(
                self.decay_mask, requires_grad=requires_grad
            )
            if train_attn_decay:
                self.get_decay_mask = self._train_decay_mask
            else:
                self.get_decay_mask = self._return_decay_mask
        requires_grad=False
        if self.mask_type is not None and 'butter' not in self.mask_type:
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
        _, idxs = np.unique(t, return_index=True)
        print(idxs, self.mask_scale)
        print(t)
        print(t[idxs])
        if self.mask_scale > 0:
            decay_interp = sp.interpolate.interp1d(t[idxs], dc[idxs], kind='linear')
        else:
            decay_interp = sp.interpolate.interp1d(t, dc)
        times = times[times<t[idxs[-1]]]
        mask = decay_interp(times)
        """
        mask = np.zeros(times.shape)
        for i in range(int(t[-1])+1):
            mask[times == i] = decay_interp(i)
        mask[times > int(t[-1])] = -np.inf
        """
        return times, mask
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



def plot_filters(ax_log, ax_w, label, times, scales, color_dict, order=None, linestyle="-", color=None, lw=3):
    CLM = CausalLocalMasks(attn_decay_type=label)
    CLM.times = nn.Parameter(torch.tensor(times), requires_grad=False)
    for i, sc in enumerate(scales):
        if label == 'powerLaw':
            CLM.mask_scale = nn.Parameter(torch.tensor(sc))
            filter = CLM._power_law_mask()
        elif label == 'simPowerLaw':
            CLM.mask_scale = nn.Parameter(torch.tensor(sc))
            filter = CLM._sim_power_law_mask()
        elif 'utter' in label:
            #CLM.mask_scale = sc
            #t, filter = CLM._butterworth_filter(order, times)
            t, filter = butterworth_filter(times, sc, order) 
            print(t.shape, filter.shape)
        lbl = str(sc) if linestyle=="-" else None
        _color = color if color is not None else color_dict[sc]
        if 'utter' in label:
            if ax_log is not None:
                ax_log.plot(t, -1*torch.abs(torch.tensor(filter)), linestyle, label=lbl, color=_color, alpha=0.5, linewidth=lw)
            ax_w.plot(t, torch.exp(-1*torch.abs(torch.tensor(filter))), linestyle, color=_color, alpha=0.5, linewidth=lw)
        else:
            if ax_log is not None:
                ax_log.plot(times, -1*torch.abs(filter.detach()), linestyle, label=lbl, color=_color, alpha=0.5, linewidth=lw)
            ax_w.plot(times, torch.exp(-1*torch.abs(torch.tensor(filter))), linestyle, color=_color, alpha=0.5, linewidth=lw)
        if ax_log is not None:
            ax_log.set_xlim(times[0], times[-1])
            ax_log.set_ylim(-50, 1)
        ax_w.set_xlim(times[0], times[-1])
        #ax_log.set_yscale('log')
        ax_w.set_ylim(0, 1.05)
    return ax_log, ax_w




def make_plot(decay_type, times, scales, color_dict, label, exp_label, order=None, legend_loc='left', fs_label=23, fs_ticks=19, fig=None, axs=None):
    if axs is None:
        fig, axs = plt.subplots(2,1, figsize=(10, 8),
            gridspec_kw={'left':0.1, 'top':0.93, 'right':0.98, 'hspace':0, "height_ratios": [2,3]})
    axs[0], axs[1] = plot_filters(
        axs[0], axs[1], decay_type, times, scales, color_dict, order=order
    )
    axs[0].xaxis.set_visible(False)
    axs[0].set_ylabel(label, fontsize=fs_label)
    axs[0].set_ylim(-4.9, 0.1)
    axs[1].set_ylabel(exp_label, fontsize=fs_label)
    axs[1].set_xlabel("Time Steps", fontsize=fs_label)
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=fs_ticks)
        #ax.set_xlim(0,8)

    axl = fig.add_axes((0, 0.96, 1, 0.04))
    for sc in scales:
        axl.plot([0, 1], [0, 1], color=color_dict[sc], label=str(sc), alpha=0.5, linewidth=3)
    axl.spines['top'].set_visible(False)
    axl.spines['bottom'].set_visible(False)
    axl.spines['left'].set_visible(False)
    axl.spines['right'].set_visible(False)
    axl.set_xlim(10, 11)
    axl.xaxis.set_ticks([])
    axl.yaxis.set_ticks([])
    axl.legend(loc='center', bbox_to_anchor=(0.5, 0.3), fontsize=fs_label, ncols=len(scales), columnspacing=0.7, frameon=False)
    #dashed_line, = axs[0].plot([-10, -1], [0, 0], '--k', label=r'$f{}^{(SPL)}(t)$', linewidth=3)
    #solid_line, = axs[0].plot([-10, -1], [0, 0], '-k', label=r'$f{}^{(PL)}(t)$', linewidth=3)
    #axs[0].legend(handles=[solid_line, dashed_line], loc="lower left", bbox_to_anchor=(0.7, 0.02), fontsize=fs_label, ncol=1, frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    if legend_loc=='left':
        axs[0].text(0.03, 0.075, 'a', fontsize=40, fontweight='bold', transform=axs[0].transAxes)
        axs[1].text(0.03, 0.075, 'b', fontsize=40, fontweight='bold', transform=axs[1].transAxes)
    elif legend_loc=='right':
        axs[0].text(0.925, 0.85, 'a', fontsize=40, fontweight='bold', transform=axs[0].transAxes)
        axs[1].text(0.925, 0.85, 'b', fontsize=40, fontweight='bold', transform=axs[1].transAxes) 
    fig.savefig(f"../{decay_type}_filters.png")
    fig.savefig(f"../{decay_type}_filters.pdf", dpi=500)


times = torch.linspace(0, 30, 1000)
make_plot('powerLaw', times, [0.1, 0.25, 0.5, 0.75, 1.0], pl_color_dict, r'$f{}^{(PL)}(t)$', r'$\exp\left[f{}^{(PL)}(t)\right]$')
make_plot('simPowerLaw', times, [0.1, 0.5, 1.0, 2.0], spl_color_dict, r'$f{}^{(SPL)}(t)$', r'$\exp\left[f{}^{(SPL)}(t)\right]$', legend_loc='right')
make_plot('butter1', times, [20., 15., 10., 5., 2.], bw_color_dict, r'$f{}^{(BW)}_1(t)$', r'$\exp\left[f{}^{(BW)}_1(t)\right]$', order=1, legend_loc='right')
make_plot('butter2', times, [20., 15., 10., 5., 2.], bw_color_dict, r'$f{}^{(BW)}_2(t)$', r'$\exp\left[f{}^{(BW)}_2(t)\right]$', order=2, legend_loc='right')








fs_label = 40
fs_ticks = 33
times = torch.linspace(0, 11, 1000)
fig, axs = plt.subplots(1, 2, figsize=(25, 8),
    gridspec_kw={'left':0.075, 'top':0.97, 'bottom':0.28, 'right':0.99, 'hspace':0.0025})
axs[0], axs[1] = plot_filters(
    axs[0], axs[1], 'butter2', times, [7.5], None, order=2, color='b', lw=5
)
axs[0], axs[1] = plot_filters(
    axs[0], axs[1], 'butter1', times, [7.15], None, order=1, color='g', lw=5
)
axs[0], axs[1] = plot_filters(
    axs[0], axs[1], 'powerLaw', times, [2.02], None, color='r', lw=5
)
axs[0], axs[1] = plot_filters(
    axs[0], axs[1], 'simPowerLaw', times, [0.68], None, color='y', lw=5
)
axs[0].set_ylabel(r'$f(t)$', fontsize=fs_label)
axs[0].set_xlabel("Time Steps", fontsize=fs_label)
axs[0].set_ylim(-4.9, 0.1)
axs[1].set_ylabel(r'$\exp\left[f(t)\right]$ ', fontsize=fs_label)
axs[1].set_xlabel("Time Steps", fontsize=fs_label)
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=fs_ticks)
    #ax.set_xlim(0,8)

axl = fig.add_axes((0, 0.05, 1, 0.05))
#axl = fig.add_axes((0, 0.94, 1, 0.04))
#axl = fig.add_axes((0.5, 0.05, 0.1, 0.9))
axl.plot([0, 1], [0, 1], color='r', label=r'$f{}^{(PL)}(t)$', alpha=0.5, linewidth=5)
axl.plot([0, 1], [0, 1], color='y', label=r'$f{}^{(SPL)}(t)$', alpha=0.5, linewidth=5)
axl.plot([0, 1], [0, 1], color='g', label=r'$f{}^{(BW)}_1(t)$', alpha=0.5, linewidth=5)
axl.plot([0, 1], [0, 1], color='b', label=r'$f{}^{(BW)}_2(t)$', alpha=0.5, linewidth=5)
axl.spines['top'].set_visible(False)
axl.spines['bottom'].set_visible(False)
axl.spines['left'].set_visible(False)
axl.spines['right'].set_visible(False)
axl.set_xlim(10, 11)
axl.xaxis.set_ticks([])
axl.yaxis.set_ticks([])
axl.legend(
    loc='center', bbox_to_anchor=(0.5, 0.3), 
    fontsize=fs_label, ncols=len(pl_color_dict), columnspacing=4, frameon=False)
#axl.legend(loc='center', bbox_to_anchor=(0.5, 0.3), fontsize=fs_label, ncols=1, columnspacing=0.7, frameon=False)
fig.tight_layout()
fig.subplots_adjust(hspace=0)
#axs[0].text(0.925, 0.825, 'a', fontsize=40, fontweight='bold', transform=axs[0].transAxes, verticalalignment='center')
#axs[1].text(0.925, 0.825, 'b', fontsize=40, fontweight='bold', transform=axs[1].transAxes, verticalalignment='center')
fig.savefig("../all_filters_horizontal.png")
fig.savefig("../all_filters_horizontal.pdf", dpi=500)





fs_label = 23
fs_ticks = 19



times = torch.linspace(0, 11, 1000)
fig, axs = plt.subplots(2,1, figsize=(10, 8),
    gridspec_kw={'left':0.1, 'top':0.93, 'right':0.98, 'hspace':0, 'height_ratios': [2,3]})
axs[0], axs[1] = plot_filters(
    axs[0], axs[1], 'butter2', times, [7.5], None, order=2, color='b'
)
axs[0], axs[1] = plot_filters(
    axs[0], axs[1], 'butter1', times, [7.15], None, order=1, color='g'
)
axs[0], axs[1] = plot_filters(
    axs[0], axs[1], 'powerLaw', times, [2.02], None, color='r'
)
axs[0], axs[1] = plot_filters(
    axs[0], axs[1], 'simPowerLaw', times, [0.68], None, color='y'
)
axs[0].xaxis.set_visible(False)
axs[0].set_ylabel(r'$f(t)$', fontsize=fs_label)
axs[0].set_ylim(-4.9, 0.1)
axs[1].set_ylabel(r'$\exp\left[f(t)\right]$ ', fontsize=fs_label)
axs[1].set_xlabel("Time Steps", fontsize=fs_label)
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=fs_ticks)
    #ax.set_xlim(0,8)

axl = fig.add_axes((0, 0.96, 1, 0.04))
axl.plot([0, 1], [0, 1], color='r', label=r'$f{}^{(PL)}(t)$', alpha=0.5, linewidth=3)
axl.plot([0, 1], [0, 1], color='y', label=r'$f{}^{(SPL)}(t)$', alpha=0.5, linewidth=3)
axl.plot([0, 1], [0, 1], color='g', label=r'$f{}^{(BW)}_1(t)$', alpha=0.5, linewidth=3)
axl.plot([0, 1], [0, 1], color='b', label=r'$f{}^{(BW)}_2(t)$', alpha=0.5, linewidth=3)
axl.spines['top'].set_visible(False)
axl.spines['bottom'].set_visible(False)
axl.spines['left'].set_visible(False)
axl.spines['right'].set_visible(False)
axl.set_xlim(10, 11)
axl.xaxis.set_ticks([])
axl.yaxis.set_ticks([])
axl.legend(loc='center', bbox_to_anchor=(0.5, 0.3), fontsize=fs_label, ncols=len(pl_color_dict), columnspacing=0.7, frameon=False)
fig.tight_layout()
fig.subplots_adjust(hspace=0)
axs[0].text(0.925, 0.825, 'a', fontsize=40, fontweight='bold', transform=axs[0].transAxes, verticalalignment='center')
axs[1].text(0.925, 0.825, 'b', fontsize=40, fontweight='bold', transform=axs[1].transAxes, verticalalignment='center')
fig.savefig("../all_filters.png")
fig.savefig("../all_filters.pdf", dpi=500)








times = torch.linspace(0, 20, 1000)
fig, axs = plt.subplots(2,1, figsize=(10, 8),
    gridspec_kw={'left':0.1, 'top':0.93, 'right':0.98, 'hspace':0})
axs[0], axs[1] = plot_filters(
    axs[0], axs[1], 'powerLaw', times, [0.1, 0.25, 0.5, 0.75, 1.0], pl_color_dict
)
axs[0], axs[1] = plot_filters(
    axs[0], axs[1], 'simPowerLaw', times, [0.1, 0.5, 1.0, 2.0], pl_color_dict, linestyle='--'
)
axs[0].xaxis.set_visible(False)
axs[0].set_ylabel(r'$f(t)$', fontsize=fs_label)
axs[0].set_ylim(-4.9, 0.1)
axs[1].set_ylabel(r'$\exp\left[f(t)\right]$ ', fontsize=fs_label)
axs[1].set_xlabel("Time Steps", fontsize=fs_label)
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=fs_ticks)
    ax.set_xlim(0,8)

axl = fig.add_axes((0, 0.96, 1, 0.04))
for sc in [0.1, 0.25, 0.5, 0.75, 1.0, 2.0]:
    axl.plot([0, 1], [0, 1], color=pl_color_dict[sc], label=str(sc), alpha=0.5, linewidth=3)
axl.spines['top'].set_visible(False)
axl.spines['bottom'].set_visible(False)
axl.spines['left'].set_visible(False)
axl.spines['right'].set_visible(False)
axl.set_xlim(10, 11)
axl.xaxis.set_ticks([])
axl.yaxis.set_ticks([])
axl.legend(loc='center', bbox_to_anchor=(0.5, 0.3), fontsize=fs_label, ncols=len(pl_color_dict), columnspacing=0.7, frameon=False)
dashed_line, = axs[0].plot([-10, -1], [0, 0], '--k', label=r'$f{}^{(SPL)}(t)$', linewidth=3)
solid_line, = axs[0].plot([-10, -1], [0, 0], '-k', label=r'$f{}^{(PL)}(t)$', linewidth=3)
axs[0].legend(handles=[solid_line, dashed_line], loc="lower left", bbox_to_anchor=(0.7, 0.02), fontsize=fs_label, ncol=1, frameon=False)
fig.tight_layout()
fig.subplots_adjust(hspace=0)
axs[0].text(0.03, 0.075, 'a', fontsize=40, fontweight='bold', transform=axs[0].transAxes)
axs[1].text(0.03, 0.075, 'b', fontsize=40, fontweight='bold', transform=axs[1].transAxes)
fig.savefig("../all_powerlaw_filters.png")
fig.savefig("../all_powerlaw_filters.pdf", dpi=500)




times = torch.linspace(0, 40, 1000)
fig, axs = plt.subplots(2,1, figsize=(10, 8),
    gridspec_kw={'left':0.1, 'top':0.93, 'right':0.98, 'hspace':0, 'height_ratios':[2, 3]})
for i, (ord, line) in enumerate([(1, "--"), (2,"-")]):
    axs[0], axs[1] = plot_filters(
        axs[0], axs[1], f"butter{ord}", times, [20., 15., 10., 5., 2.], bw_color_dict, order=ord, linestyle=line)
    #axs[0,i].xaxis.set_visible(False)
    #if i > 0:
axs[0].xaxis.set_visible(False)
axs[0].set_ylabel(r'$f_n^{(BW)}(t)$', fontsize=fs_label)
axs[0].set_ylim(-4.9, 0.1)
axs[1].set_ylabel(r'$\exp\left[f_n^{(BW)}(t)\right]$', fontsize=fs_label)
axs[1].set_xlabel("Time Steps", fontsize=fs_label)
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=fs_ticks)
    ax.set_xlim(0,32.5)

axl = fig.add_axes((0, 0.96, 1, 0.04))
for sc in [2., 5., 10., 15., 20.0]:
    axl.plot([0, 1], [0, 1], color=bw_color_dict[sc], label=str(sc), alpha=0.5, linewidth=3)
axl.spines['top'].set_visible(False)
axl.spines['bottom'].set_visible(False)
axl.spines['left'].set_visible(False)
axl.spines['right'].set_visible(False)
axl.set_xlim(10, 11)
axl.xaxis.set_ticks([])
axl.yaxis.set_ticks([])
axl.legend(loc='center', bbox_to_anchor=(0.5, 0.3), fontsize=fs_label, ncols=len(pl_color_dict), columnspacing=1, frameon=False)
dashed_line, = axs[1].plot([-10, -1], [0, 0], '--k', label=r'$f_1^{(BW)}(t)$', linewidth=3)
solid_line, = axs[1].plot([-10, -1], [0, 0], '-k', label=r'$f_2^{(BW)}(t)$', linewidth=3)
axs[1].legend(handles=[solid_line, dashed_line], loc="lower left", bbox_to_anchor=(0.7, 0.5), fontsize=fs_label, ncol=1, frameon=False)

fig.subplots_adjust(hspace=0)
#fig.legend()
axs[0].text(0.925, 0.85, 'a', fontsize=40, fontweight='bold', transform=axs[0].transAxes)
axs[1].text(0.925, 0.075, 'b', fontsize=40, fontweight='bold', transform=axs[1].transAxes)
fig.savefig("../all_butter_filters.png")
fig.savefig("../all_butter_filters.pdf", dpi=500)




lstyle = {
    "pl" : "-",
    "spl" : "--",
    "bw1" : "-.",
    "bw2" : ":"
}
times = torch.linspace(0, 30, 3000)
fig, axs = plt.subplots(2,2, figsize=(12, 6),
    gridspec_kw={'left':0.1, 'top':0.93, 'right':0.98, 'hspace':0})
axs[0,0], axs[1,0] = plot_filters(
    axs[0,0], axs[1,0], 'butter2', times, [7.5], None, order=2, color='k', linestyle=lstyle['bw2']
)
axs[0,0], axs[1,0] = plot_filters(
    axs[0,0], axs[1,0], 'butter1', times, [7.15], None, order=1, color='k', linestyle=lstyle['bw1']
)
axs[0,0], axs[1,0] = plot_filters(
    axs[0,0], axs[1,0], 'powerLaw', times, [2.02], None, color='k', linestyle=lstyle['pl']
)
axs[0,0], axs[1,0] = plot_filters(
    axs[0,0], axs[1,0], 'simPowerLaw', times, [0.68], None, color='k', linestyle=lstyle['spl']
)

_, axs[0,1] = plot_filters(
    None, axs[0,1], 'powerLaw', times, [0.1, 0.25, 0.5, 0.75, 1.0], pl_color_dict, linestyle=lstyle['pl']
)
_, axs[0,1] = plot_filters(
    None, axs[0,1], 'simPowerLaw', times, [0.1, 0.5, 1.0, 2.0], pl_color_dict, linestyle=lstyle['spl']
)
for i, (ord, line) in enumerate([(1, "--"), (2,"-")]):
    _, axs[1,1] = plot_filters(
        None, axs[1,1], f"butter{ord}", times, [20., 15., 10., 5., 2.],
        bw_color_dict, order=ord, linestyle=lstyle[f"bw{ord}"]
    )

for ir in range(2):
    axs[ir,0].set_xlim(0,11)
    axs[ir,1].set_xlim(0,30)
axs[0,0].set_ylim(-6, 0.25)


"""
axs[0].xaxis.set_visible(False)
axs[0].set_ylabel(r'$f(t)$', fontsize=fs_label)
axs[0].set_ylim(-4.9, 0.1)
axs[1].set_ylabel(r'$\exp\left[f(t)\right]$ ', fontsize=fs_label)
axs[1].set_xlabel("Time Steps", fontsize=fs_label)
for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=fs_ticks)
    #ax.set_xlim(0,8)

axl = fig.add_axes((0, 0.96, 1, 0.04))
axl.plot([0, 1], [0, 1], color='r', label=r'$f{}^{(PL)}(t)$', alpha=0.5, linewidth=3)
axl.plot([0, 1], [0, 1], color='y', label=r'$f{}^{(SPL)}(t)$', alpha=0.5, linewidth=3)
axl.plot([0, 1], [0, 1], color='g', label=r'$f{}^{(BW)}_1(t)$', alpha=0.5, linewidth=3)
axl.plot([0, 1], [0, 1], color='b', label=r'$f{}^{(BW)}_2(t)$', alpha=0.5, linewidth=3)
axl.spines['top'].set_visible(False)
axl.spines['bottom'].set_visible(False)
axl.spines['left'].set_visible(False)
axl.spines['right'].set_visible(False)
axl.set_xlim(10, 11)
axl.xaxis.set_ticks([])
axl.yaxis.set_ticks([])
axl.legend(loc='center', bbox_to_anchor=(0.5, 0.3), fontsize=fs_label, ncols=len(pl_color_dict), columnspacing=0.7, frameon=False)
fig.tight_layout()
fig.subplots_adjust(hspace=0)
axs[0].text(0.925, 0.825, 'a', fontsize=40, fontweight='bold', transform=axs[0].transAxes, verticalalignment='center')
axs[1].text(0.925, 0.825, 'b', fontsize=40, fontweight='bold', transform=axs[1].transAxes, verticalalignment='center')
"""
fig.tight_layout()
fig.savefig("../all_filters_alpha.png")
fig.savefig("../all_filters_alpha.pdf", dpi=500)