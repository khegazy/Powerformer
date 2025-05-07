import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



label_size = 18
legend_size = 17
tick_size=15
layer_types = [("encoder_SA_", "Encoder Self-Attention"), ("decoder_SA_", "Decoder Self-Attention"), ("decoder_CA_", "Decoder Cross-Attention")]
color_dict = {
        '0.0' : 'k',
        '0.1' : 'r',
        '0.25' : 'yellow',
        '0.5' : 'g',
        '0.75' : 'b',
        '1.0' : 'purple'
    }
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
line_styles = ['-', '-.', ':']
alpha = 0.5
lwidth = 3
def plot_data(data, axes, roff, coff, bin_info,
        score_xlim=(-75, 75), score_ylim=(1e2, 9e8), weight_ylim=(1e2, 9e9),
        use_markers=False, plot_masked=True, plot_unmasked=True, plot_reference=False,
        color='b', label=""
    ):
    sbins, scbins, swidth, wbins, wcbins, wwidth = bin_info

    if use_markers:
        if plot_reference:
            axes[roff][0+coff].plot(scbins, data["scores"],
                linestyle=':', alpha=1, linewidth=lwidth, color=color_dict['0.0'], label='Reference')
            axes[roff][1+coff].plot(wcbins, data["weights"],
                linestyle=':', alpha=1, linewidth=lwidth, color=color_dict['0.0'], label='Reference')
        if plot_unmasked:
            axes[roff][0+coff].plot(scbins, data["raw_scores"],
                linestyle='-.', alpha=alpha, linewidth=lwidth, color=color, label=label)
            axes[roff][1+coff].plot(wcbins, data["raw_weights"],
                linestyle='-.', alpha=alpha, linewidth=lwidth, color=color, label=label)
        if plot_masked:
            axes[roff][0+coff].plot(scbins, data["powerlaw_scores"],
                linestyle='-', alpha=alpha, linewidth=lwidth, color=color, label=label)
            axes[roff][1+coff].plot(wcbins, data["powerlaw_weights"],
                linestyle='-', alpha=alpha, linewidth=lwidth, color=color, label=label)
    else:
        if plot_reference:
            axes[roff][0+coff].bar(scbins, data["scores"],
                color='k', alpha=1, width=swidth, label='Reference')
            axes[roff][1+coff].bar(wcbins, data["weights"],
                color='k', alpha=1, label='Reference', width=wwidth)
        if plot_unmasked:
            axes[roff][0+coff].bar(scbins, data["raw_scores"],
                color='r', alpha=0.3, width=swidth, label=label)
            axes[roff][1+coff].bar(wcbins, data["raw_weights"],
                color='r', alpha=0.3, width=wwidth, label=label)
        if plot_masked:
            axes[roff][0+coff].bar(scbins, data["powerlaw_scores"], color='b', alpha=0.3, width=swidth, label=label)
            axes[roff][1+coff].bar(wcbins, data["powerlaw_weights"], color='b', alpha=0.3, width=wwidth, label=label)
    axes[roff][0+coff].set_yscale('log')
    axes[roff][1+coff].set_yscale('log')
    axes[roff][0+coff].set_xlim(*score_xlim)
    axes[roff][1+coff].set_xlim(0, 1)
    axes[roff][0+coff].set_ylim(*score_ylim)
    axes[roff][1+coff].set_ylim(*weight_ylim)
    #axes[roff][1+coff].yaxis.set_ticks([])


def get_data(dataset, model, seq_len, pred_len, decay_type, decay_times):
    
    folders = glob.glob(f"../../results/{dataset}*{model}*sl{seq_len}*pl{pred_len}*{decay_type}_*xp_0")
    print(f"../../results/{dataset}*{model}*sl{seq_len}*pl{pred_len}*{decay_type}_*xp_0", len(folders))
    folders = folders + glob.glob(f"../../results/{dataset}*{model}*sl{seq_len}*pl{pred_len}*None_*xp_0")
    plt_fld = f"../{model}_{dataset}_{decay_type}"
    if not os.path.exists(plt_fld):
        os.makedirs(plt_fld)
    
    failed = False
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
        if not os.path.exists(os.path.join(fld, 'score_bins.npy')):
            print(f"Missing files in {fld}")
            failed = True
        if not failed:
            print("loading", fld)
            sbins = np.load(os.path.join(fld, 'score_bins.npy'))
            scbins = (sbins[:-1] + sbins[1:])/2
            swidth = sbins[1] - sbins[0]
            wbins = np.load(os.path.join(fld, 'weight_bins.npy'))
            wcbins = (wbins[:-1] + wbins[1:])/2
            wwidth = wbins[1] - wbins[0]
 

    return folder_dict, plt_fld, (sbins, scbins, swidth, wbins, wcbins, wwidth)



def plot_attn_summary(
        model, dataset, decay_type, pred_len="", decay_times=[], plot_single=False, plot_only_encoder=False,
        score_xlim=(-75, 75), score_ylim=(1e2, 9e8), weight_ylim=(1e2, 9e9)
    ):
    is_transformer = model.lower() == 'transformer'
    data = {}
    sl = 256 if is_transformer else 512

    folder_dict, plt_fld, bin_info = get_data(
        dataset, model, sl, pred_len, decay_type, decay_times + ['0.0']
    )
    fig_w = 7
    fig_h = 4
    if plot_single:
        raise NotImplementedError()
    elif is_transformer and not plot_only_encoder:
        fig, axes = plt.subplots(
            3, 1,
            figsize=(fig_w, fig_h*3),
            gridspec_kw={
                #'width_ratios' : [0.01,3],
                #'wspace' : 0.07,
                'hspace' : 0.02,
                'top' : 0.96,
                'bottom' : 0.05,
                'right' : 0.975,
                'left' : 0.08
            }
        )
    else:
        fig, axes = plt.subplots(
            1, 1, figsize=(fig_w, fig_h*1.1),
            gridspec_kw={
                'top' : 0.91,
                'bottom' : 0.15,
                'right' : 0.975,
                'left' : 0.08
            }

        )
        axes = [axes]
    
    for ilr, (layer_type, layer_label) in enumerate(layer_types):
        if (plot_only_encoder or not is_transformer) and ilr > 0:
            break
        fld, seq_len, pred_len = folder_dict['0.0']
        ref_key = (seq_len, pred_len, '0.0')
        ref_data =\
            {
                "raw_scores" : np.load(os.path.join(fld, layer_type + 'attn_raw_scores.npy')),
                "raw_weights" : np.load(os.path.join(fld, layer_type + 'attn_raw_weights.npy'))
            }
 
        x0_t = 0.7
        x0 = 0.44
        wax = 0.52
        hax = 0.12
        if is_transformer and not plot_only_encoder:
            axes[ilr].text(
                0.115, 0.9, letters[ilr], fontweight='bold',
                horizontalalignment='center', verticalalignment='center',
                transform=axes[ilr].transAxes, fontsize=30
            ) 
            if ilr == 0:
                axs = [fig.add_axes((x0, 0.835, wax, hax))]
                axes[ilr].text(
                    x0_t, 0.05, layer_label, 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[ilr].transAxes, fontsize=label_size
                )
            elif ilr == 1:
                axs = [fig.add_axes((x0, 0.53, wax, hax))]
                axes[ilr].text(
                    x0_t, 0.05, layer_label,  
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[ilr].transAxes, fontsize=label_size
                )
            elif ilr == 2:
                axs = [fig.add_axes((x0, 0.225, wax, hax))]
                axes[ilr].text(
                    x0_t, 0.4, layer_label,
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[ilr].transAxes, fontsize=label_size
                ) 
        else:
            axs = [fig.add_axes((x0, 0.6, wax, hax*2.5))]
            axes[ilr].text(
                0.01, 0.06, layer_label, 
                horizontalalignment='left', verticalalignment='center',
                transform=axes[ilr].transAxes, fontsize=label_size
            )
        axs.append(axes[ilr])


        axs[0].text(
            0.85, 0.85, "Score",
            horizontalalignment='center', verticalalignment='center',
            transform=axs[0].transAxes, fontsize=label_size
        ) 
        if is_transformer and not plot_only_encoder:
            axs[1].text(
                0.1, 0.05, f"Layer {len(ref_data['raw_scores'])}",
                horizontalalignment='center', verticalalignment='center',
                transform=axs[1].transAxes, fontsize=label_size
            )
        else:
            axs[1].text(
                0.01, 0.15, f"Layer {len(ref_data['raw_scores'])}",
                horizontalalignment='left', verticalalignment='center',
                transform=axs[1].transAxes, fontsize=label_size
            ) 
        
        for itm, decay_time in enumerate(decay_times):
            fld, seq_len, pred_len = folder_dict[decay_time]
            #if ilr > 0 and not is_transformer:
            #    break
            key = (seq_len, pred_len, decay_time)
            data[key] =\
                {
                    "raw_scores" : np.load(os.path.join(fld, layer_type + 'attn_raw_scores.npy')),
                    "powerlaw_scores" : np.load(os.path.join(fld, layer_type + 'attn_powerlaw_scores.npy')),
                    "raw_weights" : np.load(os.path.join(fld, layer_type + 'attn_raw_weights.npy')),
                    "powerlaw_weights" : np.load(os.path.join(fld, layer_type + 'attn_powerlaw_weights.npy')),
                }
            
            #print(cbins.shape, data[key]["raw_scores"].shape)
            plot_data_dict = {
                "scores" : ref_data["raw_scores"][-1],
                "weights" : ref_data["raw_weights"][-1],
                "raw_scores" : data[key]["raw_scores"][-1],
                "powerlaw_scores" : data[key]["powerlaw_scores"][-1],
                "raw_weights" : data[key]["raw_weights"][-1],
                "powerlaw_weights" : data[key]["powerlaw_weights"][-1]
            }
            plot_data(
                data=plot_data_dict,
                axes=[axs],
                roff=0,
                coff=0,
                bin_info=bin_info,
                score_xlim=score_xlim,
                score_ylim=score_ylim,
                weight_ylim=weight_ylim,
                use_markers=True,
                plot_unmasked=False,
                plot_reference=True,
                color=color_dict[decay_time],
                label=decay_time
            )
        if ilr < len(layer_types) - 1 and is_transformer and not plot_only_encoder:
            axes[ilr].xaxis.set_ticks([])
        else:
            axes[ilr].set_xlabel("Attention Weight", fontsize=label_size)
        for ax in axs:
            ax.tick_params(axis='both', labelsize=tick_size)
    
    ax1 = fig.add_axes([0, 0.96, 0.5, 0.04])
    ax1.plot([0, 1], [0,1], color=color_dict['0.0'], linestyle=':', label="Reference", linewidth=lwidth)
    ax1.plot([0, 1], [0,1], color=color_dict['0.0'], linestyle='-', label="Masked", linewidth=lwidth)
    ax2 = fig.add_axes([0.5, 0.96, 0.5, 0.04])
    for t in decay_times:
        ax2.plot([0, 1], [0,1], color=color_dict[t], linestyle='-', label=t, linewidth=lwidth, alpha=alpha)
    if is_transformer and not plot_only_encoder:
        loops = [(ax1, 2, (0.5, 0.4)), (ax2, 3, (0.55, 0.4))]
    else:
        loops = [(ax1, 2, (0.5, 0.1)), (ax2, 3, (0.55, 0.1))]
    for ax, ncols, loc in loops:
        ax.set_xlim(10,11)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.legend(loc='center', bbox_to_anchor=loc, fontsize=legend_size, ncols=ncols, frameon=False, columnspacing=0.8, handlelength=1.5)
    #axes[0].legend(loc='center', bbox_to_anchor=(0.475, 1.064), fontsize=legend_size, ncols=len(decay_times)+1)
    if plot_only_encoder:
        fig.savefig(os.path.join(plt_fld, f"attn_encoder_effects_sq{sl}_pl{pred_len}.png"))
        fig.savefig(os.path.join(plt_fld, f"attn_encoder_effects_sq{sl}_pl{pred_len}.pdf"), dpi=500)
    else:
        fig.savefig(os.path.join(plt_fld, f"attn_general_effects_sq{sl}_pl{pred_len}.png"))
        fig.savefig(os.path.join(plt_fld, f"attn_general_effects_sq{sl}_pl{pred_len}.pdf"), dpi=500)
    



def plot_attn(
        model, dataset, decay_type, pred_len="", decay_times=[], plot_single=False,
        score_xlim=(-75, 75), score_ylim=(1e2, 9e8), weight_ylim=(1e2, 9e9)
    ):
    is_transformer = model.lower() == 'transformer'
    data = {}
    sl = 256 if is_transformer else 512
    folders = glob.glob(f"../../results/{dataset}*{model}*sl{sl}*pl{pred_len}*{decay_type}*exp_0")
    plt_fld = f"../{model}_{dataset}_{decay_type}"
    if not os.path.exists(plt_fld):
        os.makedirs(plt_fld)
    
    folder_dict, plt_fld, bin_info = get_data(
        dataset, model, sl, pred_len, decay_type, decay_times + ['0.0']
    )
    fig_w = 7
    fig_h = 4
    attn_label_size = 1.3*label_size
    attn_legend_size = 1.3*legend_size
    attn_tick_size = 1.35*tick_size
    if plot_single:
        fig, axes = plt.subplots(
            figsize=(fig_w*1.05, fig_h*1.1),
            gridspec_kw={
                'top' : 0.92,
                'bottom' : 0.15,
                'right' : 0.97,
                'left' : 0.075
            }

        )
        axes = [[axes]]
    elif is_transformer:
        fig, axes = plt.subplots(
            #6, 3*len(decay_times) - 1,
            6, len(decay_times),
            figsize=(fig_w*(0.2 + len(decay_times)), fig_h*(4 + 2*0.05)),
            gridspec_kw={
                #'width_ratios' : [1,3] + [0.1, 1, 3]*(len(decay_times)-1),
                'height_ratios' : [1,1,0.1,1,0.1,1],
                'wspace' : 0.065,
                'hspace' : 0.05,
                'top' : 0.955,
                'bottom' : 0.06,
                'right' : 0.985,
                'left' : 0.06
            }
        )
    else:
        fig, axes = plt.subplots(
            3, len(decay_times),
            figsize=(fig_w*len(decay_times), fig_h*3),
            gridspec_kw={
                'wspace' : 0.065,
                'hspace' : 0.05,
                'top' : 0.95,
                'bottom' : 0.07,
                'right' : 0.985,
                'left' : 0.055
            }
        )

    loff = 0
    for ilr, (layer_type, layer_label) in enumerate(layer_types):
        if is_transformer:
            n_attn = 2 if ilr == 0 else 1
            if loff > 0:
                for ax in axes[loff-1]:
                    ax.set_visible(False)
        else:
            n_attn = 3
            if ilr > 0:
                break
        
        if plot_single:
            if ilr > 0:
                break

        fld, seq_len, pred_len = folder_dict['0.0']
        ref_key = (seq_len, pred_len, '0.0')
        ref_data =\
            {
                "raw_scores" : np.load(os.path.join(fld, layer_type + 'attn_raw_scores.npy')),
                "raw_weights" : np.load(os.path.join(fld, layer_type + 'attn_raw_weights.npy'))
            }
        for itm, decay_time in enumerate(decay_times):
            fld, seq_len, pred_len = folder_dict[decay_time]
            #if ilr > 0 and not is_transformer:
            #    break
            key = (seq_len, pred_len, decay_time)
            data[key] =\
                {
                    "raw_scores" : np.load(os.path.join(fld, layer_type + 'attn_raw_scores.npy')),
                    "powerlaw_scores" : np.load(os.path.join(fld, layer_type + 'attn_powerlaw_scores.npy')),
                    "raw_weights" : np.load(os.path.join(fld, layer_type + 'attn_raw_weights.npy')),
                    "powerlaw_weights" : np.load(os.path.join(fld, layer_type + 'attn_powerlaw_weights.npy')),
                }
            
            assert n_attn == len(data[key]["raw_scores"])
            for lyr in range(n_attn):
                lyr_idx = -1 if plot_single else lyr
                if plot_single and lyr > 0:
                    break
                    
                plot_data_dict = {
                    "scores" : ref_data["raw_scores"][lyr_idx],
                    "weights" : ref_data["raw_weights"][lyr_idx],
                    "raw_scores" : data[key]["raw_scores"][lyr_idx],
                    "powerlaw_scores" : data[key]["powerlaw_scores"][lyr_idx],
                    "raw_weights" : data[key]["raw_weights"][lyr_idx],
                    "powerlaw_weights" : data[key]["powerlaw_weights"][lyr_idx]
                }
                if plot_single:
                    #x0 = 0.58
                    #y0 = 0.63
                    #wax = 0.37
                    #hax = 0.27
                    x0 = 0.44
                    y0 = 0.6
                    wax = 0.52
                    hax = 0.12*2.5
                elif is_transformer:
                    wax = 0.15
                    hax = 0.09
                    if itm == 0:
                        x0 = 0.195
                    elif itm == 1:
                        x0 = 0.51
                    elif itm == 2:
                        x0 = 0.825
                    if ilr == 0:
                        if lyr == 0:
                            y0 = 0.8617
                        elif lyr == 1:
                            y0 = 0.65
                    elif ilr == 1:
                        y0 = 0.4105
                    elif ilr == 2:
                        y0 = 0.1712
                else:
                    wax = 0.15
                    hax = 0.09
                    if itm == 0:
                        x0 = 0.195
                    elif itm == 1:
                        x0 = 0.51
                    elif itm == 2:
                        x0 = 0.825
                    if lyr == 0:
                        y0 = 0.855
                    elif lyr == 1:
                        y0 = 0.5565
                    elif lyr == 2:
                        y0 = 0.259

                axs = [fig.add_axes((x0, y0, wax, hax)), axes[loff+lyr][itm]]
                if plot_single:
                    axs[0].text(
                        0.85, 0.85, "Score",
                        horizontalalignment='center', verticalalignment='center',
                        transform=axs[0].transAxes, fontsize=label_size
                    )
                    axs[1].text(
                        0.02, 0.14, f"Layer {n_attn}",
                        horizontalalignment='left', verticalalignment='center',
                        transform=axs[1].transAxes, fontsize=label_size
                    )
                    axs[1].text(
                        0.02, 0.05, layer_label, 
                        verticalalignment='center', horizontalalignment='left',
                        transform=axs[1].transAxes, fontsize=label_size
                    )
                    axs[1].text(
                        0.15, 0.77, r'$\alpha$ = ' + decay_time, 
                        verticalalignment='center', horizontalalignment='center',
                        transform=axs[1].transAxes, fontsize=label_size*1.15
                    )
                    if decay_type == 'powerLaw':
                        f_text = r'$f^{(\text{PL})}(t)$'
                    elif decay_type == 'simPowerLaw':
                        f_text = r'$f^{(\text{SPL})}(t)$'
                    axs[1].text(
                        0.15, 0.885, f_text, 
                        verticalalignment='center', horizontalalignment='center',
                        transform=axs[1].transAxes, fontsize=label_size*1.3
                    )
                else:
                    axs[0].text(
                        0.85, 0.82, "Score",
                        horizontalalignment='center', verticalalignment='center',
                        transform=axs[0].transAxes, fontsize=attn_label_size
                    )
                    axs[1].text(
                            0.1, 0.05, f"Layer {lyr+1}",
                        horizontalalignment='center', verticalalignment='center',
                        transform=axs[1].transAxes, fontsize=attn_label_size
                    ) 
                    if is_transformer and lyr == 0 and itm == 0:
                        if ilr == 0:
                            axs[1].text(-0.15, 0.0, layer_label, rotation='vertical',
                                verticalalignment='center', horizontalalignment='center',
                                transform=axs[1].transAxes, fontsize=attn_label_size
                            )
                        else:
                            text = layer_label.split()
                            axs[1].text(-0.17, 0.5, text[0], rotation='vertical',
                                verticalalignment='center', horizontalalignment='center',
                                transform=axs[1].transAxes, fontsize=attn_label_size
                            )
                            axs[1].text(-0.12, 0.5, text[1], rotation='vertical',
                                verticalalignment='center', horizontalalignment='center',
                                transform=axs[1].transAxes, fontsize=attn_label_size
                            )
                    elif not is_transformer and lyr == 1 and itm == 0:
                        axs[1].text(-0.15, 0.5, layer_label, rotation='vertical',
                            verticalalignment='center', horizontalalignment='center',
                            transform=axs[1].transAxes, fontsize=attn_label_size
                        )

                plot_data(
                    data=plot_data_dict,
                    axes=[axs],
                    roff=0,
                    coff=0,
                    bin_info=bin_info,
                    score_xlim=score_xlim,
                    score_ylim=score_ylim,
                    weight_ylim=weight_ylim,
                    plot_masked='SA' in layer_type,
                    plot_reference=True,
                    use_markers=True,
                    color=color_dict[decay_time],
                )
                for ax in axs:
                    if plot_single:
                        ax.tick_params(axis='both', labelsize=tick_size)
                    else:
                        ax.tick_params(axis='both', labelsize=attn_tick_size)
                if itm > 0:
                    axes[loff+lyr][itm].yaxis.set_ticks([])
                if not plot_single and ((is_transformer and ilr < len(layer_types) - 1) or (not is_transformer and lyr < n_attn-1)):
                    axes[loff+lyr][itm].xaxis.set_ticks([])
                else:
                    if plot_single:
                        axes[loff+lyr][itm].set_xlabel("Attention Weights", fontsize=label_size)
                    else:
                        axes[loff+lyr][itm].set_xlabel("Attention Weights", fontsize=attn_label_size)
        loff += n_attn + 1
    if plot_single:
        ax1 = fig.add_axes([0.0, 0.95, 1.0, 0.05])
        ax1.plot([0, 1], [0,1], color=color_dict[decay_time], linestyle='-', label='Masked', alpha=alpha, linewidth=lwidth)
        ax1.plot([0, 1], [0,1], color=color_dict[decay_time], linestyle='-.', label="Without Mask", alpha=alpha, linewidth=lwidth)
        ax1.plot([0, 1], [0,1], color=color_dict['0.0'], linestyle=':', label="Reference", linewidth=lwidth)
        loop = [(ax1, 3, (0.5, 0.3))]
        columnspacing=1
    else:
        ax1 = fig.add_axes([0.5, 0.96, 0.4, 0.04])
        for t in decay_times:
            ax1.plot([0, 1], [0,1], color=color_dict[t], linestyle='-', label=t, linewidth=lwidth, alpha=alpha)
        ax2 = fig.add_axes([0.1, 0.96, 0.4, 0.04])
        ax2.plot([0, 1], [0,1], color=color_dict['0.0'], linestyle='-', label="Masked", linewidth=lwidth)
        ax2.plot([0, 1], [0,1], color=color_dict['0.0'], linestyle='-.', label="Without Mask", linewidth=lwidth)
        ax2.plot([0, 1], [0,1], color=color_dict['0.0'], linestyle=':', label="Reference", linewidth=lwidth)
        loop = [(ax1, 3, (0.5, 0.4)), (ax2, 3, (0.5, 0.4))]
        columnspacing=2
    for ax, ncols, loc in loop:
        ax.set_xlim(10,11)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        if plot_single:
            ax.legend(loc='center', bbox_to_anchor=loc, fontsize=legend_size, ncols=ncols, frameon=False, columnspacing=columnspacing)
        else:
            ax.legend(loc='center', bbox_to_anchor=loc, fontsize=attn_legend_size, ncols=ncols, frameon=False, columnspacing=columnspacing)

    if plot_single:
        fig.savefig(os.path.join(plt_fld, f"{model}_single_scores_sq{sl}_pl{pred_len}.png"))
        fig.savefig(os.path.join(plt_fld, f"{model}_single_scores_sq{sl}_pl{pred_len}.pdf"), dpi=500)
    else:
        fig.savefig(os.path.join(plt_fld, f"{model}_scores_sq{sl}_pl{pred_len}.png"))
        fig.savefig(os.path.join(plt_fld, f"{model}_scores_sq{sl}_pl{pred_len}.pdf"), dpi=500)
    
