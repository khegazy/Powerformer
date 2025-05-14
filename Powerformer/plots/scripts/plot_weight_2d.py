import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


data0 = np.load("results/ETTh1_Powerformer_ETTh1_ftM_sl336_ll48_pl96_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtTrue_atpowerLaw_1.0_Exp_0/encoder_SA_attn_raw_weights_2d.npy")
data1 = np.load("results/ETTh1_Powerformer_ETTh1_ftM_sl336_ll48_pl96_dm16_nh4_el3_dl1_df128_fc1_ebtimeF_dtTrue_atpowerLaw_1.0_Exp_0/encoder_SA_attn_powerlaw_weights_2d.npy")
data0 = np.load("results/Weather_Powerformer_custom_ftM_sl512_ll48_pl96_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_atpowerLaw_1.0_Exp_0/encoder_SA_attn_raw_weights_2d.npy")
data1 = np.load("results/Weather_Powerformer_custom_ftM_sl512_ll48_pl96_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_atpowerLaw_1.0_Exp_0/encoder_SA_attn_powerlaw_weights_2d.npy")

lsize = 19
tsize = 16
fig = plt.figure(figsize=(15, 5))
ax0s, ax1s = [], []
N_time = data0[0].shape[0]//2
X, Y = np.meshgrid(
    np.linspace(-1*N_time, N_time, data0[0].shape[0]+1),
    np.linspace(0, 1.0, data0[0].shape[1]+1)
)

lbuf = 0.065
width = 0.28
shift = 0.435
vmin, vmax = 1, 5e7
for i, (idx, ltr) in enumerate(zip([0, 2], [('a', 'c'), ('b', 'd')])):
    ax0s.append(fig.add_axes(
        (lbuf+i*(width+0.005), 0.135, width, 0.84) 
    ))
    ax1s.append(fig.add_axes(
        (lbuf+2*width+0.017+i*(0.005+0.5*width), 0.135, width/2, 0.84) 
    ))
    print(np.amax(data0[idx]))
    im = ax0s[-1].pcolormesh(
        X, Y, data0[idx].transpose(),
        norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap='YlOrRd'
    )
    ax1s[-1].pcolormesh(
        X[:,N_time:], Y[:,N_time:], data1[idx,N_time:].transpose(),
        norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap='YlOrRd'
    )
    ax0s[-1].text(0.02, 0.88, ltr[0], fontsize=33, weight='semibold', transform=ax0s[-1].transAxes)
    ax1s[-1].text(0.1, 0.88, ltr[1], fontsize=33, weight='semibold', transform=ax1s[-1].transAxes)
    ax0s[-1].text(0.8, 0.85, "Without Mask", fontsize=17, ha='center', transform=ax0s[-1].transAxes)
    ax1s[-1].text(0.7, 0.85, "With Mask", fontsize=17, ha='center', transform=ax1s[-1].transAxes)
    ax0s[-1].text(0.8, 0.93, f"Layer {idx+1}", fontsize=17, ha='center', transform=ax0s[-1].transAxes)
    ax1s[-1].text(0.7, 0.93, f"Layer {idx+1}", fontsize=17, ha='center', transform=ax1s[-1].transAxes)
cax = fig.add_axes((lbuf+0.01+width*3/2+shift, 0.135, 0.03, 0.84))

for i, ax in enumerate(ax0s + ax1s):
    ax.set_yscale('log')
    ax.set_ylim(0.001, 1.0)
    if i < 2:
        ax.set_xlim(-1*N_time+1, N_time)
        ax.set_xscale('symlog')
    else:
        ax.set_xscale('symlog')
        #ax.set_xlim(1e-1, 60)
    if i > 0:
        ax.yaxis.set_visible(False)
    ax.set_xlabel("Time Delay", fontsize=lsize)
    ax.tick_params(axis='both', labelsize=tsize)
    ax.grid(False)
ax0s[0].set_ylabel("Attention Weight", fontsize=lsize)
fig.colorbar(im, cax)
cax.tick_params(axis='both', labelsize=tsize)
fig.savefig("./plots/weight_distributions.png")
#fig.savefig("./plots/weight_distributions.eps")
#fig.savefig("./plots/weight_distributions.svg")
#fig.savefig("./plots/weight_distributions.pdf", dpi=500)

"""
for i, data in enumerate([data0, data1]):
    data = data[-1]
    N_time = data.shape[0]//2
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(
        np.linspace(-1*N_time, N_time, data.shape[0]+1),
        np.linspace(0, 1.0, data.shape[1]+1)
    )
    ax.pcolormesh(X, Y, data.transpose(), norm=colors.LogNorm())
    ax.set_yscale('log')
    ax.set_ylim(0.001, 1.0)
    #fig.colorbar()

    fig.savefig(f"test{i}.png")
"""