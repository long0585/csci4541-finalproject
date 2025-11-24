import numpy as np
import matplotlib.pyplot as plt
import argparse

def hide_top_right(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
parser = argparse.ArgumentParser()
parser.add_argument('--alphabet', type=str, default='a b c d e f g h i j k l m n o p q r s t u v w x y z', help='Choose custom alphabet')
parser.add_argument('--gpt_engine', type=str, default='gpt-4o')
parser.add_argument('--effort', type=str)
args = parser.parse_args()
# Load data
effort_level = args.effort
alphabet = args.alphabet.replace(" ", "")
results = {}
# Add gpt5 results
for interval_size in range(1, 3):
    if interval_size == 1:
        if args.gpt_engine == 'gpt-4o':
            path = "./int1_results/gpt-4o_" + alphabet + "_int1/" + "acc_" + alphabet + ".npz"
        elif args.gpt_engine == 'gpt-5':
            path = "./int1_results/gpt-5_" + alphabet + "_int1/" + effort_level + "_" + alphabet + ".npz"
    elif interval_size == 2:
        if args.gpt_engine == 'gpt-4o':
            path = "./int2_results/gpt-4o_" + alphabet + "_int2/" + "acc_" + alphabet + ".npz"
        elif args.gpt_engine == 'gpt-5':
            path = "./int2_results/gpt-5_" + alphabet + "_int2/" + effort_level + "_" + alphabet + ".npz"
    results[interval_size] = np.load(path)



# gpt4_int2_results = np.load('./gpt-4-0125-preview_int2/acc.npz')
# older GPT-4 engine
# old_gpt4_int1_results = np.load('./gpt-4-1106-preview_int1/acc.npz')
# old_gpt4_int2_results = np.load('./gpt-4-1106-preview_int2/acc.npz')

# Get accuracy for each condition
# newer GPT-4 engine
# Interval size = 1
int_acc = [result['overall_acc'].item() for result in results.values()]
int_err = [result['overall_err'][0] for result in results.values()]
# Interval size = 2
# int2_acc = gpt4_int2_results['overall_acc'].item()
# int2_err = gpt4_int1_results['overall_err'][0]
# Combined
# gpt4_acc = np.array([gpt4_int1_acc, gpt4_int2_acc])
# gpt4_err = np.array([gpt4_int1_err, gpt4_int2_err])
# older GPT-4 engine
# Interval size = 1
# old_gpt4_int1_acc = old_gpt4_int1_results['overall_acc'].item()
# old_gpt4_int1_err = old_gpt4_int1_results['overall_err'][0]
# Interval size = 2
# old_gpt4_int2_acc = old_gpt4_int2_results['overall_acc'].item()
# old_gpt4_int2_err = old_gpt4_int1_results['overall_err'][0]
# Combined
# old_gpt4_acc = np.array([old_gpt4_int1_acc, old_gpt4_int2_acc])
# old_gpt4_err = np.array([old_gpt4_int1_err, old_gpt4_int2_err])

# Plot parameters
total_bar_width = 0.8
ind_bar_width = total_bar_width / 5
colors = ['powderblue', 'darkmagenta', 'salmon', 'mediumseagreen', 'royalblue']
plot_fontsize = 14
title_fontsize = 16
axis_label_fontsize = 14
## Plot separately for different interval-size conditions
all_prob_type_names = ['Extend\nsequence', 'Successor', 'Predecessor', 'Remove\nredundant\nletter', 'Fix\nalphabetic\nsequence', 'Sort']
N_cond = 6
x_points = np.arange(N_cond)
# Interval size = 1
ax = plt.subplot(111)
plt.bar(x_points - (ind_bar_width * 0.5), results[1]['all_acc'], yerr=results[1]['all_err'], color=colors[0], edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.bar(x_points + (ind_bar_width * 0.5), results[2]['all_acc'], yerr=results[2]['all_err'], color=colors[1], edgecolor='black', width=ind_bar_width, ecolor='gray')
plt.ylim([0,1])
plt.yticks([0,0.2,0.4,0.6,0.8,1],['0','0.2','0.4','0.6','0.8','1'], fontsize=plot_fontsize)
plt.ylabel('Accuracy', fontsize=axis_label_fontsize)
plt.xticks(x_points, all_prob_type_names, fontsize=9.5)
plt.xlabel('Transformation type', fontsize=axis_label_fontsize)
plt.title('Engine = ' + args.gpt_engine + '\n' + "Alphabet: " + alphabet, fontsize=title_fontsize)
plt.legend([str(name) for name in results.keys()],fontsize=plot_fontsize,frameon=False, bbox_to_anchor=(1.1, 1))
hide_top_right(ax)
if args.gpt_engine == 'gpt-4o':
    plt.savefig('./gpt4o_int1_int2_' + alphabet + '_combined_results.png', dpi=300, bbox_inches="tight")
elif args.gpt_engine == 'gpt-5':
    plt.savefig('./gpt5_int1_int2_' + alphabet + '_combined_results.png', dpi=300, bbox_inches="tight")
plt.close()