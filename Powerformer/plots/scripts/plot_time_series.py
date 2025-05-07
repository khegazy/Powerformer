import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

models = ["Powerformer"]
masks = ["powerLaw"]
datasets = ["Weather"]
pred_lens = ["720"]
decay_times = ["1.0"]
label_size = 22
tick_size = 17
colors = ['r', 'g', 'b', 'purple']


def get_data(dataset, model, seq_len, pred_len, decay_type, decay_times):
    
    folders = glob.glob(f"../../results/{dataset}*{model}*sl{seq_len}*pl{pred_len}*{decay_type}_*xp_0")
    print(f"../../results/{dataset}*{model}*sl{seq_len}*pl{pred_len}*{decay_type}_*xp_0", len(folders))
    folders = folders + glob.glob(f"../../results/{dataset}*{model}*sl{seq_len}*pl{pred_len}*None_*xp_0")
    plt_fld = f"../{model}_{dataset}_{decay_type}"
    if not os.path.exists(plt_fld):
        os.makedirs(plt_fld)
    
    folder_dict = {}
    for fld in folders:
        idx0 = fld.find("_sl")
        idx1 = fld.find("_", idx0+2)
        seq_len = fld[idx0+3:idx1]
        idx0 = fld.find("_pl")
        idx1 = fld.find("_", idx0+2)
        pred_len = fld[idx0+3:idx1]
        dcay = 'None' if 'None' in fld else decay_type
        idx0 = fld.find(dcay)
        idx1 = fld.find("_", idx0+len(dcay)+1)
        decay_time = fld[idx0+len(dcay)+1:idx1]
        if len(decay_times) > 0 and decay_time not in decay_times:
            continue
        folder_dict[decay_time] = (fld, seq_len, pred_len)
        if not os.path.exists(os.path.join(fld, 'pred.npy')):
            print(f"Missing files in {fld}")
    return folder_dict

for mdl in models:
    for msk in masks:
        for dts in datasets:
            for pl in pred_lens:
                if mdl.lower() == 'powerformer':
                    if dts.lower() == 'traffic':
                        seq_len = 336
                    else:
                        seq_len = 336
                else:
                    seq_len = 256
                folder_dict = get_data(dts, mdl, seq_len, pl, msk, decay_times + ["0.0"])

                plot_dir = os.path.join("..", f"time_series_{mdl}_{dts}_{msk}")
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                print(list(folder_dict.keys()))
                #fld_none, _, _ = folder_dict['0.0']
                #pred_none = np.load(os.path.join(fld_none, "pred.npy"))
                trues = np.load(os.path.join("..", "..", "results", f"data_{dts}_pl{pl}.npy"))
                for idx in [167, 3, 789, 657, 1358, 1697, 2675, 2369]:
                    if idx > 700:
                        sys.exit()
                    if idx > len(trues):
                        continue
                    for mv_idx in range(trues.shape[-1]):
                        fig, ax = plt.subplots(figsize=(10, 7),
                            gridspec_kw={"top":0.92, "bottom":0.1, "left":0.125, "right":0.98})
                        ax.plot(trues[idx,:,mv_idx], '-k', linewidth=2, label="Ground Truth")
                        for idc, dc_time in enumerate(decay_times):
                            if float(dc_time) == 0:
                                lbl == "PatchTST"
                            else:
                                lbl = dc_time
                            fld, sl, pl = folder_dict[dc_time]
                            pred = np.load(os.path.join(fld, "pred.npy"))
                            mse = np.load(os.path.join(fld, "mse.npy"))
                            ax.plot(pred[idx,:,mv_idx], '-', color=colors[idc], alpha=0.6, linewidth=2, label=lbl)
                        ax.set_xlim(0, trues.shape[1])
                        ax.set_xlabel("Time Steps", fontsize=label_size)
                        ax.set_ylabel(f"{dts} Variable {mv_idx+1}", fontsize=label_size)
                        ax.tick_params(axis='both', labelsize=tick_size)
                        plt.tight_layout()
                        ax.legend(loc='center', bbox_to_anchor=(0.45, 1.04), fontsize=label_size, ncols=5, frameon=False, columnspacing=2)
                        fig.savefig(
                            os.path.join(plot_dir, f"sl{seq_len}_pl{pl}_var{mv_idx}_id{idx}.png")
                        )


        
