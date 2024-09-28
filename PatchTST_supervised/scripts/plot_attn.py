import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


for dataset in ['ETTh1']:
    for powerlaw in ['zeta']:
        data = {}
        folders = glob.glob(f"./results/*{dataset}*{powerlaw}*")
        sbins = np.load(os.path.join(folders[0], 'score_bins.npy'))
        scbins = (sbins[:-1] + sbins[1:])/2
        swidth = sbins[1] - sbins[0]
        wbins = np.load(os.path.join(folders[0], 'weight_bins.npy'))
        wcbins = (wbins[:-1] + wbins[1:])/2
        wwidth = wbins[1] - wbins[0]
        for fld in folders:
            idx0 = fld.find("_sl")
            idx1 = fld.find("_", idx0+2)
            seq_len = fld[idx0+3:idx1]
            idx0 = fld.find("_pl")
            idx1 = fld.find("_", idx0+2)
            pred_len = fld[idx0+3:idx1]
            idx0 = fld.find(powerlaw)
            idx1 = fld.find("_", idx0+len(powerlaw)+1)
            attn_decay = fld[idx0+len(powerlaw)+1:idx1]

            key = (seq_len, pred_len, attn_decay)
            data[key] =\
                {
                    "raw_scores" : np.load(os.path.join(fld, 'attn_raw_scores.npy')),
                    "powerlaw_scores" : np.load(os.path.join(fld, 'attn_powerlaw_scores.npy')),
                    "raw_weights" : np.load(os.path.join(fld, 'attn_raw_weights.npy')),
                    "powerlaw_weights" : np.load(os.path.join(fld, 'attn_powerlaw_weights.npy')),
                }
            
            fig, axes = plt.subplots(3, 2)
            #print(cbins.shape, data[key]["raw_scores"].shape)
            for lyr in range(3):
                axes[lyr][0].bar(scbins, data[key]["raw_scores"][lyr], color='lightblue', width=swidth)
                axes[lyr][0].bar(scbins, data[key]["powerlaw_scores"][lyr], color='k', alpha=0.4, width=swidth)
                axes[lyr][1].bar(wcbins, data[key]["raw_weights"][lyr], color='lightblue', width=wwidth)
                axes[lyr][1].bar(wcbins, data[key]["powerlaw_weights"][lyr], color='k', alpha=0.4, width=wwidth)
                axes[lyr][0].set_yscale('log')
                axes[lyr][1].set_yscale('log')
                axes[lyr][0].set_xlim(-75, 75)
                axes[lyr][1].set_xlim(0, 1)

            plt_fld = f"plots/{dataset}_{powerlaw}"
            if not os.path.exists(plt_fld):
                os.makedirs(plt_fld)
            plt.tight_layout()
            fig.savefig(os.path.join(plt_fld, "scores_sq{}_pl{}_atd{}.png".format(*key)))
        
        #fig, axes = plt.subplots(1, 3)
        #    for ax in axes






