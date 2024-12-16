""" Optimal Contract Visualizer (The Multitype Version)

This script allows user to visualize the optimal contract for the monopolistic screening problem in collaborative machine learning when more than 2 types of agents are involved.

It contains two functions:
    - Visual3D: visualizes the optimal contributions and rewards for a wide range of N and p specified by the user.
    - WelfareAnalysis: visualizes the welfare implications of information asymmetry for a wide range of N and p specified by the user.
"""

# Import the relevant packages
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import scipy.stats as stats
import numpy as np


# Style hyperparameters

c_theme = ['#114B5F', '#1A936F','#88D498','#C6DABF','#F3E9D2','#456990','#E4FDE1', '#775253', '#351431']
accent = "#be0f34"
accent2 = "#a79d96"
accent3 = "#122f53"


# Calculate the reward distributions
def realizedDistribution(sol, N, p, t_bar, v, a):
    """
    Calculates the realized reward distributions given first-moment solutions, N and p
 
    Args:
        sol (ndarray): Optimal contract delineating contributions m and rewards t.
        N (int): Total number of participants.
        p (ndarray): Probabilities for different private types.
        t_bar (ndarray): maximal rewardable model values expected by different agent types—this is used in propostional assignment
        v (func): Valuation function
        a (func): Accuracy function
    Returns:
        tuple: (case_by_case, prob_outcomes, t_ratios)
            case_by_case (ndarray): value of the collectively trained model under each realized outcome
            prob_outcomes (ndarray): probabilities corresponding to the realizaed outcomes
            t_ratios (ndarray): t_i^*/ t_bar_i, which is used in the proportional assignment policy
    """
    m, t = sol[:int(len(sol)/2)], sol[int(len(sol)/2):]
    def multinorm_pmf(x):
        return stats.multinomial.pmf(x, N, p)
        # x: list of length K [n_1, ..., n_K]
        # N: total numebr of trials
        # p: list of length K [p_1, ..., p_K]

    def generate_multinomial_outcomes(n, k, prefix=[]):
        if k == 1:
            # Only one category left, all remaining trials must go into this category
            yield prefix + [n]
        else:
            for i in range(n + 1):
                # Allocate i trials to the current category, and recursively allocate the rest
                yield from generate_multinomial_outcomes(n - i, k - 1, prefix + [i])

    # Precompute all the possible outcomes
    K = len(t)
    all_outcomes = np.array([outcome for outcome in generate_multinomial_outcomes(N, K)])
    prob_outcomes = multinorm_pmf(all_outcomes)
    
    t_ratios = t/t_bar
    
    def case_by_case_value(x):
        return v(a(all_outcomes @ x[:int(len(x)/2)]))
    case_by_case = case_by_case_value(sol)
    
    return case_by_case, prob_outcomes, t_ratios

def RewardDistributions(sol, N, p, t_bar, v, a):
    """
    Calculates the reward distributions for different agents

    Args:
        sol (ndarray): Optimal contract delineating contributions m and rewards t.
        N (int): Total number of participants.
        p (ndarray): Probabilities for different private types.
        t_bar (ndarray): maximal rewardable model values expected by different agent types—this is used in propostional assignment
        v (func): Valuation function
        a (func): Accuracy function
    Returns:
        t_dist (ndarray): the support of realized rewards for different agents        
    """
    case_by_case, _, t_ratios = realizedDistribution(sol, N, p, t_bar, v, a)
    t_dist = case_by_case.reshape(-1,1) * t_ratios
    return t_dist


# Visualize the results
def VisualizeReward(sol, t_dist, res_u, c, case_by_case, prob_outcomes, ax = None):
    """
    Visualize the reward distributions for different agents together with the first-moment solution

    Args:
        sol (ndarray): Optimal contract delineating contributions m and rewards t (first-moment solution).
        t_dist (ndarray): Support of realized rewards for different agents     
        res_u (ndarray): Reservation utilities of the agents, which are used to determine types that would not train a model on their own
        c (ndarray): private costs of the agents
        case_by_case (ndarray): Value of the collectively trained model under each realized outcome
        prob_outcomes (ndarray): Probabilities corresponding to the realizaed outcomes
        ax (Axes): Optional. If not None, the ax is used for the plot. Default is None.
    Returns:
        None. 
    """
    # Example data: a list of tuples (case_by_case, prob_outcomes)
    m, t = sol[:int(len(sol)/2)], sol[int(len(sol)/2):]

    data = [ (t_dist[:,i], prob_outcomes) for i in range(len(c))]


    # Create the ridge plot
    bool_noax = False
    if not ax:
        bool_noax = True
        fig, ax = plt.subplots(figsize=(10, 6))
    num_bins = int(np.max([10, np.round(len(prob_outcomes)/100,-1)]))

    y_offset = 0.4  # Adjust the offset to control the amount of overlap
    # Iterate through the data in the original order
    for i, (case_by_case, prob_outcomes) in enumerate(data):
        # Create histogram
        hist, bins = np.histogram(case_by_case, bins=num_bins*5, weights=prob_outcomes, density=False)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        # Normalize the histogram to create the overlapping effect
        hist = hist / hist.max() * 0.6
        # Plot the histogram as filled areas
        ax.fill_between(bin_centers, i * y_offset, i * y_offset + hist, step="mid", alpha=0.6, color = c_theme[i+1], zorder=len(data) - i)
        #ax.plot(bin_centers, i * y_offset + hist, color='b', zorder=len(data) - i)
        ax.plot(t[i], i * y_offset, '^', color=accent, markersize=10, zorder=len(data) + 1, label='First-moment solution $t$' if i == 0 else "")


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

    # Adjust y-ticks and y-tick labels
    ax.set_yticks([i * y_offset for i in range(len(data))])
    ax.set_yticklabels([f'Type {i+1}' for i in range(len(data))])
    ax.set_xlabel('Value of reward $v(r)$', fontsize = 12)
    ax.set_xlim(np.max([-10, np.min(t_dist)]),np.max(t_dist))
    ax.set_title('Reward distributions', fontsize = 14)

    # Types that will not train on their own
    if len(np.where(res_u==0)[0]):
        non_training_thresh = np.max(np.where(res_u==0))
        for tick in ax.get_yticklabels():
            tick_value = int(tick.get_text()[5:])-1
            if tick_value <= non_training_thresh:
                tick.set_color(accent2)


    handles, labels = ax.get_legend_handles_labels()
    if len(np.where(res_u==0)[0]):
        text_legend = Line2D([0], [0], marker=r'$\mathrm{T}$', color=accent2, label='Type of agent with $f_i = 0$', markerfacecolor=accent2, markersize=10, linestyle='None')
        handles.append(text_legend)
        labels.append('Type of agent with $f_i = 0$')
    ax.tick_params(axis='both', which='major', labelsize=12) 
    ax.legend(handles=handles, labels=labels, loc='upper left', fontsize = 12)
    if bool_noax:
        plt.tight_layout()
        plt.show()

    
def VisualizeContribution(sol, res_m, ax = None, flex_scale = False, margin = 1):
    """
    Visualize the optimal contributions from different agents

    Args:
        sol (ndarray): Optimal contract delineating contributions m and rewards t (first-moment solution).  
        res_m (ndarray): Reservation contributions of the agents
        ax (Axes): Optional. If not None, the ax is used for the plot. Default is None.
        flex_scale (bool): If True, the flexible scale will be used.
        margin (float): Set the margin to be used for the flexible scale.
        
    Returns:
        None. 
    """
    m = sol[:int(len(sol)/2)]
    def ceil_round(value, digit):
        unit = (10**(-digit))
        return np.ceil(value / unit) * unit
    def floor_round(value, digit):
        unit = (10**(-digit))
        return np.floor(value / unit) * unit

    colours = ["#a4d8ff","#FD6467"]

    types = [f'Type {i+1}' for i in range(len(m))]
    weight_counts = {
        "Base contribution": res_m,
        "Incentivized additional contribution": m-res_m,
    }
    width = 0.5
    
    bool_noax = False
    if not ax:
        bool_noax = True
        fig, ax = plt.subplots()
    bottom = np.zeros(5)


    for idx, (label, weight_count) in enumerate(weight_counts.items()):
        for i in range(len(weight_count)):
            color = colours[idx]
            hatch = 'xxx' if weight_count[i] < 0 else None
            edgecolor = 'white' if weight_count[i] < 0 else None
            alpha = 0.5 if weight_count[i] < 0 else 1
            ax.bar(types[i], weight_count[i], width, label=label if i == 0 else "", bottom=bottom[i], color=color, hatch=hatch, edgecolor = edgecolor, alpha = alpha)
            bottom[i] += weight_count[i]

    ax.set_title("Data contributions", fontsize = 14)
    if flex_scale:
        yrange = np.max([m,res_m]) - np.min(m)
        ax.set_ylim(np.min(m) - yrange*margin, np.max([m,res_m]) + yrange*margin)
    else:
        ax.set_ylim(0, ceil_round(np.max([m,res_m])*1.2, -1))

    # Combine all patches into a single legend
    handles, _ = ax.get_legend_handles_labels()
    if (m - res_m  < 0).any():
        neg_patch = mpatches.Patch(facecolor= "#FD6467", edgecolor='white', hatch='xxx', label='Incentivized reduction in contribution')
        handles.append(neg_patch)

    # Types that will not train on their own
    if len(np.where(res_m==0)[0]):
        non_training_thresh = np.max(np.where(res_m==0))
        for tick in ax.get_xticklabels():
            tick_value = int(tick.get_text()[5:])-1
            if tick_value <= non_training_thresh:
                tick.set_color(accent2)

    
    if bool_noax:
        if len(np.where(res_m==0)[0]):
            text_legend = Line2D([0], [0], marker=r'$\mathrm{T}$', color=accent2, label='Type of agent with $f_i = 0$', markerfacecolor=accent2, markersize=10, linestyle='None')
            handles.append(text_legend) 
    ax.legend(handles=handles, loc="upper left", fontsize = 12)
    ax.tick_params(axis='both', which='major', labelsize=12) 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if bool_noax:
        plt.show()

def VisualizeTM(sol, t_dist, res_u, res_m, c, case_by_case, prob_outcomes, one_legend = False, flex_scale = False, margin = 1):
    """
    Visualize the optimal contributions from different agents

    Args:
        sol (ndarray): Optimal contract delineating contributions m and rewards t (first-moment solution).
        t_dist (ndarray): Support of realized rewards for different agents     
        res_u (ndarray): Reservation utilities of the agents, which are used to determine types that would not train a model on their own
        res_m (ndarray): Reservation contributions of the agents
        c (ndarray): Private costs of the agents
        case_by_case (ndarray): Value of the collectively trained model under each realized outcome
        prob_outcomes (ndarray): Probabilities corresponding to the realizaed outcomes
        one_legend (bool): If True, a unified legend will be created for the plots.
        flex_scale (bool): If True, the flexible scale will be used for the Data Contributions plot.
        margin (float): Set the margin to be used for the flexible scale used for the Data Contributions plot.
        
    Returns:
        None. 
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [5, 3]}, figsize=(12, 4))
    VisualizeReward(sol, t_dist, res_u, c, case_by_case, prob_outcomes, ax1)
    VisualizeContribution(sol, res_m, ax2, flex_scale, margin)
    if one_legend:
        handles1, labels1 = ax1.get_legend().legend_handles, [text.get_text() for text in ax1.get_legend().texts]
        handles2, labels2 = ax2.get_legend().legend_handles, [text.get_text() for text in ax2.get_legend().texts]
        handles = handles1 + handles2
        labels = labels1 + labels2
        # Remove individual legends
        ax1.legend().remove()
        ax2.legend().remove()
        
        # Create a combined legend
        legend = fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(1.08, 0.1))
        legend.set_title('Legend', prop={'weight': 'bold'})
        legend.get_frame().set_linestyle('--')
    
    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()


def VisualizeContribution_ZoomIn(m, res_m, ax = None, flex_scale = False, margin = 1, types = None):
    """
    Visualize the optimal contributions from different agents (The Zoom-In Version)

    Args:
        m (ndarray): Optimal contributions of the types m.
        res_m (ndarray): Reservation contributions of the types m.
        ax (Axes): Optional. If not None, the ax is used for the plot. Default is None.
        flex_scale (bool): If True, the flexible scale will be used for the Data Contributions plot.
        margin (float): Set the margin to be used for the flexible scale used for the Data Contributions plot.
        types (list): Specify the types labels for the zoomed-in plot.

    Returns:
        None. 
    """    
    def ceil_round(value, digit):
        unit = (10**(-digit))
        return np.ceil(value / unit) * unit
    def floor_round(value, digit):
        unit = (10**(-digit))
        return np.floor(value / unit) * unit

    colours = ["#a4d8ff","#FD6467"]
    if type(types) == type(None):
        types = [f'Type {i+1}' for i in range(len(m))]
    weight_counts = {
        "Base contribution": res_m,
        "Incentivized additional contribution": m-res_m,
    }
    width = 0.5
    
    bool_noax = False
    if not ax:
        bool_noax = True
        fig, ax = plt.subplots()
    bottom = np.zeros(5)


    for idx, (label, weight_count) in enumerate(weight_counts.items()):
        for i in range(len(weight_count)):
            color = colours[idx]
            hatch = 'xxx' if weight_count[i] < 0 else None
            edgecolor = 'white' if weight_count[i] < 0 else None
            alpha = 0.5 if weight_count[i] < 0 else 1
            ax.bar(types[i], weight_count[i], width, label=label if i == 0 else "", bottom=bottom[i], color=color, hatch=hatch, edgecolor = edgecolor, alpha = alpha)
            bottom[i] += weight_count[i]

    #ax.set_title("Data contributions", fontsize = 14)
    if flex_scale:
        yrange = np.max([m,res_m]) - np.min(m)
        ax.set_ylim(np.min(m) - yrange*margin, np.max([m,res_m]) + yrange*margin)
    else:
        ax.set_ylim(0, ceil_round(np.max([m,res_m])*1.2, -1))

    # Combine all patches into a single legend
    handles, _ = ax.get_legend_handles_labels()
    if (m - res_m  < 0).any():
        neg_patch = mpatches.Patch(facecolor= "#FD6467", edgecolor='white', hatch='xxx', label='Incentivized reduction in contribution')
        handles.append(neg_patch)

    # Types that will not train on their own
    if len(np.where(res_m==0)[0]):
        non_training_thresh = np.max(np.where(res_m==0))
        for tick in ax.get_xticklabels():
            tick_value = int(tick.get_text()[5:])-1
            if tick_value <= non_training_thresh:
                tick.set_color(accent2)

    
    if bool_noax:
        if len(np.where(res_m==0)[0]):
            text_legend = Line2D([0], [0], marker=r'$\mathrm{T}$', color=accent2, label='Type of agent with $f_i = 0$', markerfacecolor=accent2, markersize=10, linestyle='None')
            handles.append(text_legend) 
    #ax.legend(handles=handles, loc="upper left", fontsize = 12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    formatter = ticker.ScalarFormatter(useOffset=False, useMathText=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
   # ax.spines['bottom'].set_visible(False)
   # ax.spines['left'].set_visible(False)
    if bool_noax:
        plt.show()


def GraphicalIllustration(sol, res_m, t_bar, c, v, a, ax = None):
    """
    Generate a graphical illustration of the optimal contract in 2D.

    Args:
        sol (ndarray): Optimal contributions of the types m.
        res_m (ndarray): Reservation contributions of the types m.
        t_bar (ndarray): maximal rewardable model values expected by different agent types—this is used in propostional assignment
        c (ndarray): Private costs of the agents.
        v (func): Valuation function.
        a (func): Accuracy function.
        ax (Axes): Optional. If not None, the ax is used for the plot. Default is None.

    Returns:
        None. 
    """  
    m, t = sol[:int(len(sol)/2)], sol[int(len(sol)/2):]
    bool_noax = False
    if type(ax) == type(None):
        bool_noax = True
        fig, ax = plt.subplots(1,2, figsize = (12,4))
    x = np.linspace(0, np.max([m, res_m])*1.1, 1000)

    def plot_f(m, c, x):
        intercept = v(a(m))-c*m
        y = intercept.reshape(-1,1) + c.reshape(-1,1)*x
        return y

    def plot_u(m, t, c, x):
        intercept = t-c*m
        y = intercept.reshape(-1,1) + c.reshape(-1,1)*x
        return y

    ax[0].plot(x, v(a(x)), color = accent3, linewidth = 2, label = '$v(a(m))$')

    # f lines
    f_lines = plot_f(res_m,c,x)
    u_lines = plot_u(m,t,c,x)
    for i, f_line in enumerate(f_lines):
        ax[0].plot(x, f_line, linestyle = '--', linewidth = 1, alpha = 0.5, color = c_theme[i+1])
    for i, u_line in enumerate(u_lines):
        ax[0].plot(x, u_line, linestyle = '-', linewidth = 0.8, alpha = 1, color = c_theme[i+1])
    ax[0].hlines(t_bar[-1], min(x), max(x), color = accent, linewidth = 2, label = '$t_\mathrm{max}$')
    ax[0].scatter(m,t, color = accent, label = 'first-moment solution')

    ax[0].scatter(res_m, v(a(res_m)), color = accent3, label = 'reservation contribution')
    ax[0].set_ylim(v(a(np.min(x)))*0.95,100)

    handles, labels = ax[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth = 1, alpha = 0.5, label='reservation line $t-c_i m = f_i$'))
    handles.append(Line2D([0], [0], color='black', linestyle='-', linewidth = 0.8, label='isoprofit line $t-c_i m = u_i$'))

    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].tick_params(axis='both', which='major', labelsize=12, pad=0) 
    ax[0].set_xlabel('data contribution $m$', fontsize = 12)
    ax[0].set_ylabel('model reward in value terms $v(r)$', fontsize = 12)


    ax[1].hlines(t_bar[-1], min(x), max(x), color = accent, linewidth = 2, label = '$t_\mathrm{max}$')
    ax[1].scatter(m,t, color = accent, label = 'first-moment solution')
    for i, u_line in enumerate(u_lines):
        ax[1].plot(x, u_line, linestyle = '-', linewidth = 0.8, alpha = 1, color = c_theme[i+1])
    ax[1].set_ylim(np.min(t)*0.999,np.max(t)*1.001)
    ax[1].set_xlim(np.min(m)*0.999,np.max(m)*1.001)
    ax[1].set_xlabel('data contribution $m$', fontsize = 12)
    ax[1].tick_params(axis='both', which='major', labelsize=12) 
    fig.suptitle("Graphical illustration of the solution", fontsize = 14)
    ax[1].legend(handles=handles, fontsize =10, loc = 'lower right')
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    if bool_noax:
        plt.show()