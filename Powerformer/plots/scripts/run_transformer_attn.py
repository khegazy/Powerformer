from plot_attn import plot_attn_summary, plot_attn

powerLaw_lims = {
    'score_xlims' : {
        'ETTh1' : (-24, 24),
        'ETTh2' : (-24, 24),
        'ETTm1' : (-59, 59),
        'ETTm2' : (-59, 59),
        'Weather' : (-84, 84),
        'Electricity' : (-34, 34),
        'Traffic' : (-34, 34)
    },
    'score_ylims' : {
        'ETTh1' : (1e1, 9e8),
        'ETTh2' : (1e1, 9e8),
        'ETTm1' : (1e1, 9e9),
        'ETTm2' : (1e1, 9e9),
        'Weather' : (1, 9e10),
        'Electricity' : (1e1, 9e8),
        'Traffic' : (1e1, 9e8)
    },
    'weight_ylims' : {
        'ETTh1' : (1, 9e8),
        'ETTh2' : (1, 9e8),
        'ETTm1' : (1e1, 3e10),
        'ETTm2' : (1e1, 3e10),
        'Weather' : (1e3, 9e10),
        'Electricity' : (1e1, 9e9),
        'Traffic' : (1e1, 9e9)
    }
}

simPowerLaw_lims = {
    'score_xlims' : {
        'ETTh1' : (-34, 34),
        'ETTh2' : (-34, 34),
        'ETTm1' : (-64, 64),
        'ETTm2' : (-64, 64),
        'Weather' : (-74, 74),
        'Electricity' : (-34, 34),
        'Traffic' : (-34, 34)
    },
    'score_ylims' : {
        'ETTh1' : (1e1, 9e8),
        'ETTh2' : (1e1, 9e8),
        'ETTm1' : (1e1, 9e9),
        'ETTm2' : (1e1, 9e9),
        'Weather' : (1e1, 9e10),
        'Electricity' : (1e1, 9e8),
        'Traffic' : (1e1, 9e8)
    },
    'weight_ylims' : {
        'ETTh1' : (5, 5e9),
        'ETTh2' : (5, 5e9),
        'ETTm1' : (1e1, 3e10),
        'ETTm2' : (1e1, 3e10),
        'Weather' : (1e3, 9e10),
        'Electricity' : (1e1, 9e9),
        'Traffic' : (1e1, 9e9)
    }
}


all_lims = {
    'powerLaw' : powerLaw_lims,
    'simPowerLaw' : simPowerLaw_lims
}
for model in ['Powerformer']:
    break # DEBUG
    #for data in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather']: #, 'Electricity', 'Traffic']:
    for data in []: #, 'Electricity', 'Traffic']:
        for mask in ['powerLaw', 'simPowerLaw']:
            for pl in ['96', '720']:
                plot_attn_summary(
                    model, data, mask, pl, ['0.1', '0.5', '1.0'], 
                    score_xlim=all_lims[mask]['score_xlims'][data], 
                    score_ylim=all_lims[mask]['score_ylims'][data],
                    weight_ylim=all_lims[mask]['weight_ylims'][data]
                )
                plot_attn(
                    model, data, mask, pl, ['0.1', '0.5', '1.0'], 
                    score_xlim=all_lims[mask]['score_xlims'][data],
                    score_ylim=all_lims[mask]['score_ylims'][data], 
                    weight_ylim=all_lims[mask]['weight_ylims'][data]
                )

                if data == 'Weather':
                    plot_attn(
                        model, data, mask, pl, ['1.0'], plot_single=True, 
                        score_xlim=all_lims[mask]['score_xlims'][data], score_ylim=all_lims[mask]['score_ylims'][data], weight_ylim=(1e4, 9e10)
                    )
                    plot_attn_summary(
                        model, data, mask, pl, ['0.1', '0.5', '1.0'], 
                        score_xlim=all_lims[mask]['score_xlims'][data], 
                        score_ylim=all_lims[mask]['score_ylims'][data],
                        weight_ylim=(1e4, 1e11)
                    )


score_xlims = {
    'ETTh1' : (-194, 194),
    'ETTh2' : (-194, 194),
    'ETTm1' : (-244, 244),
    'ETTm2' : (-244, 244),
    'Weather' : (-49, 49),
    'Electricity' : (-79, 79),
    'Traffic' : (-74, 74)
}

score_ylims = {
    'ETTh1' : (1e2, 9e9),
    'ETTh2' : (1e2, 9e9),
    'ETTm1' : (1e2, 9e9),
    'ETTm2' : (1e2, 9e9),
    'Weather' : (1e2, 9e10),
    'Electricity' : (1e2, 9e9),
    'Traffic' : (1e2, 9e8)
}

weight_ylims = {
    'ETTh1' : (10, 9e9),
    'ETTh2' : (10, 9e9),
    'ETTm1' : (10, 9e10),
    'ETTm2' : (10, 9e10),
    'Weather' : (1, 1e11),
    'Electricity' : (1, 1e11),
    'Traffic' : (1, 9e9)
}


#for data in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather']:
for data in ['Electricity']:
    for mask in ['powerLaw']:
        for model in ['Transformer']:
            for pl in ['96', '720']:
                plot_attn_summary(
                    model, data, mask, pl, ['0.1', '0.5', '1.0'],
                    score_xlim=score_xlims[data], score_ylim=score_ylims[data], weight_ylim=weight_ylims[data]
                )
                plot_attn_summary(
                    model, data, mask, pl, ['0.1', '0.5', '1.0'], plot_only_encoder=True,
                    score_xlim=score_xlims[data], score_ylim=score_ylims[data], weight_ylim=weight_ylims[data]
                )
                plot_attn(
                    model, data, mask, pl, ['0.1', '0.5', '1.0'], 
                    score_xlim=score_xlims[data], score_ylim=score_ylims[data], weight_ylim=weight_ylims[data]
                )

