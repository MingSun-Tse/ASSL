import numpy as np
import os
import sys
pjoin = os.path.join
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman" # set fonts globally
sys.path.insert(0, os.getcwd())
from utils import set_ax, parse_value, parse_ExpID, check_path

'''Usage Example:
    python tools/nips21_latex_plots/plot_SSA_reg_loss.py ../experiment/Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.9_RGP0.0001_RUL0.5_Pretrain_SPR/log_prune.txt model.body.4.body.0 
'''


markers = ['*', 'd', 'x', 'o']
colors = ['b', 'r', 'k', 'g']
linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
plot_ix_of_this_stage = {
    '0': 0,
    '1': 0,
    '2': 0,
    '3': 0,
    '4': 0
}

def get_std(w_abs):
    out = []
    for x in w_abs:
        out.append(np.std(x) / np.mean(x))
    return np.array(out)

# ---------------------------------
log_file = sys.argv[1]
layer = sys.argv[2]
# ---------------------------------

step = []
reg = []
mag_pruned = []
mag_kept = []
wn_scale_pruned = []
wn_scale_kept = []


# set up fig and needed axes
fig, ax1 = plt.subplots(figsize=(4, 2))
fs = 12
ax2 = ax1.twinx()
set_ax(ax1)
set_ax(ax2)
ax2.grid(None)
ax1_color = 'blue'
ax2_color = 'red'

# set x ylabel
ax1.set_xlabel('Iteration (k)', fontsize=fs)
ax1.set_ylabel('WN mean scale', fontsize=fs)
ax2.set_ylabel('SI reg. co-efficient $\\alpha$', fontsize=fs)

lines = open(log_file).readlines()
for i in range(len(lines)):
    line = lines[i].strip()
    if "'%s'" % layer in line:
        step_ = int(line.split('Iter = ')[1])
        if 'Just finished' not in lines[i+1]:
            reg_ = float(lines[i+1].strip().split(' ')[-1])
            mag_pruned_ = float(lines[i+2].strip().split('pruned')[1].split('kept')[0].strip())
            mag_kept_ = float(lines[i+2].strip().split('kept')[-1])
            wn_scale_pruned_ = float(lines[i+3].strip().split('pruned')[1].split('kept')[0].strip())
            wn_scale_kept_ = float(lines[i+3].strip().split('kept')[-1])
        else:
            reg_ = float(lines[i+2].strip().split(' ')[-1])
            mag_pruned_ = float(lines[i+3].strip().split('pruned')[1].split('kept')[0].strip())
            mag_kept_ = float(lines[i+3].strip().split('kept')[-1])
            wn_scale_pruned_ = float(lines[i+4].strip().split('pruned')[1].split('kept')[0].strip())
            wn_scale_kept_ = float(lines[i+4].strip().split('kept')[-1])
        step += [step_]
        reg += [reg_]
        mag_pruned += [mag_pruned_]
        mag_kept += [mag_kept_]
        wn_scale_pruned += [wn_scale_pruned_]
        wn_scale_kept += [wn_scale_kept_]

step = [x/1000 for x in step]
ax1.plot(step, wn_scale_pruned, color = 'blue', linestyle='dotted', label='Pruned filters')
ax1.plot(step, wn_scale_kept, color = 'blue', linestyle='solid', label='Kept filters')
ax2.plot(step, reg, color='red', linestyle='dashed')

# set y1, y2 axis
ax1.yaxis.label.set_color(ax1_color); ax1.tick_params(axis='y', colors=ax1_color)
ax2.yaxis.label.set_color(ax2_color); ax2.tick_params(axis='y', colors=ax2_color)

# set legend
ax1.legend(frameon=False)
ax1.set_title(layer, fontsize=fs)
# leg = ax1.get_legend()
# leg.legendHandles[0].set_color('k')
# leg.legendHandles[1].set_color('k')

# Save
out = 'wn_scale_reg_vs_step_%s.pdf' % layer
fig.savefig(out, bbox_inches='tight')
plt.close(fig)