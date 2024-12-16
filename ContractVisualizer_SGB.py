""" Optimal Contract Visualizer (The Standard Generalization Bound Version)

This script allows user to visualize the optimal contract for the monopolistic screening problem in collaborative machine learning.

It contains two functions:
    - Visual3D: visualizes the optimal contributions and rewards for a wide range of N and p specified by the user.
    - WelfareAnalysis: visualizes the welfare implications of information asymmetry for a wide range of N and p specified by the user.
"""

# import the packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def Visual3D(pRange, NRange, result, save = False, filename = None):
    pmesh, nmesh = np.meshgrid(pRange[:,0], NRange)
    """
    Visualizes the optimal contract as 3D plots. 
 
    Args:
        pRange (ndarray): Probabilility pairs used for the simulation.
        NRange (ndarray): Total numbers of participants used for the simulation.
        result (ndarray): Optimal contracts associated with associated settings of p and N.
        save (bool): If true, save the plots as png files.
        filename (bool): If not None, save the plots using the provided filename.
    Returns:
        None
    """
    # Optimal contribution
    m_min, m_max = np.floor(np.min(result[:,:,:2])/100)*100, np.ceil(np.max(result[:,:,:2])/100)*100
    t_min, t_max = np.floor(np.min(result[:,:,2:])/10)*10, np.ceil(np.max(result[:,:,2:])/10)*10
    fig, ax = plt.subplots(1,2, figsize = (8,4), subplot_kw={"projection": "3d"})
    ax[0].plot_wireframe(pmesh, nmesh, result[:, :, 0], rstride=1, cstride=1, linewidth =0.3, color = 'crimson', label = 'high-cost type')
    ax[0].plot_wireframe(pmesh, nmesh, result[:, :, 1], rstride=1, cstride=1, linewidth =0.3, label = 'low-cost type')
    ax[0].zaxis.set_major_locator(LinearLocator(3))
    ax[0].zaxis.set_major_formatter('{x:.0f}')
    ax[0].yaxis.set_major_locator(LinearLocator(3))
    ax[0].yaxis.set_major_formatter('{x:.0f}')
    ax[0].set_ylim(0,100)
    
    ax[0].set_xlim(0, 1)
    ax[0].set_zlim(m_min, m_max)
    ax[0].xaxis.set_major_locator(LinearLocator(3))
    ax[0].xaxis.set_major_formatter('{x:.01f}')
    
    # make the panes transparent
    ax[0].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax[0].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax[0].zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax[0].xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax[0].yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax[0].zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    ax[0].set_xlabel('$p_1$', fontsize =12)
    ax[0].set_ylabel('$N$', fontsize =12)
    ax[0].tick_params(axis='both', which='major', labelsize=12, pad=0) 
    ax[0].set_title('Optimal contribution $m_i$', y = 1.02, fontsize = 14)

    # Optimal reward

    ax[1].plot_wireframe(pmesh, nmesh, result[:, :, 2], rstride=1, cstride=1, linewidth =0.3, color = 'crimson', label = 'high-cost type')
    ax[1].plot_wireframe(pmesh, nmesh, result[:, :, 3], rstride=1, cstride=1, linewidth =0.3, label = 'low-cost type')
    ax[1].zaxis.set_major_locator(LinearLocator(3))
    ax[1].zaxis.set_major_formatter('{x:.0f}')
    ax[1].yaxis.set_major_locator(LinearLocator(3))
    ax[1].yaxis.set_major_formatter('{x:.0f}')
    ax[1].set_ylim(0,100)
    
    ax[1].set_xlim(0, 1)
    ax[1].set_zlim(t_min, t_max)
    ax[1].xaxis.set_major_locator(LinearLocator(3))
    ax[1].xaxis.set_major_formatter('{x:.01f}')

    # make the panes transparent
    ax[1].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax[1].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax[1].zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax[1].xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax[1].yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax[1].zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    ax[1].set_xlabel('$p_1$', fontsize =12)
    ax[1].set_ylabel('$N$', fontsize =12)
    ax[1].tick_params(axis='both', which='major', labelsize=12, pad=0) 
    ax[1].set_title('Optimal reward $t_i$', y = 1.02, fontsize = 14)

    # Create custom legend handles with thicker lines
    line1 = mlines.Line2D([], [], color='crimson', linewidth=1, label='high-cost type')
    line2 = mlines.Line2D([], [], linewidth=1, label='low-cost type')

    # Create a single shared legend
    fig.legend(handles=[line1, line2], loc='lower center', bbox_to_anchor=(0.5, 0), ncol=2, frameon=False, fontsize=12)
    plt.subplots_adjust(wspace=0.1)
    plt.show()
    if save:
        if filename:
            fig.savefig(f'{filename}_3D.png')
        else:
            fig.savefig('simulation 1.png')  
    
    return None 

def WelfareAnalysis(pRange, NRange, data, save = False, filename = None):
    pmesh, nmesh = np.meshgrid(pRange[:,0], NRange)
    """
    Visualizes the welfare implications (information cost and information rents) of the information asymmetry
 
    Args:
        pRange (ndarray): Probabilility pairs used for the simulation.
        NRange (ndarray): Total numbers of participants used for the simulation.
        data (tuple): (info_rent, info_cost)
        save (bool): If true, save the plots as png files.
        filename (bool): If not None, save the plots using the provided filename.
    Returns:
        None
    """
    
    # Generate the colour palettes
    low_cost_palette = ['lightblue', '#1f77b4']
    high_cost_palette = ['tomato', 'crimson']
    low_cost_cmap = LinearSegmentedColormap.from_list("low_cost_cmap", low_cost_palette, N=256)
    high_cost_cmap = LinearSegmentedColormap.from_list("high_cost_cmap", high_cost_palette, N=256)
    colours = ["#a5acf3", "#e7a3e5"]
    accent =  "#6c7cb4"
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colours, N=256)

    info_rent, info_cost = data
    
    fig, ax = plt.subplots(1,2, figsize = (8,4), subplot_kw={"projection": "3d"})

    
    z_min, zmax = np.min(info_cost), np.max(info_cost)
    # Info cost
    ax[0].plot_wireframe(pmesh, nmesh, info_cost, rstride=1, cstride=1, linewidth =0.5, color = accent)
    ax[0].plot_surface(pmesh, nmesh, info_cost, cmap=custom_cmap, alpha = 0.75)

    ax[0].set_xlim(0, 1)
    ax[0].set_zlim(z_min, 0)
    ax[0].set_ylim(0,100)
    ax[0].xaxis.set_major_locator(LinearLocator(3))
    ax[0].xaxis.set_major_formatter('{x:.01f}')
    ax[0].yaxis.set_major_locator(LinearLocator(3))
    ax[0].yaxis.set_major_formatter('{x:.0f}')
    ax[0].zaxis.set_major_locator(LinearLocator(3))
    ax[0].zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # make the panes transparent
    ax[0].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax[0].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax[0].zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax[0].xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax[0].yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax[0].zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    ax[0].set_xlabel('$p_1$', fontsize =12)
    ax[0].set_ylabel('$N$', fontsize =12)
    ax[0].tick_params(axis='both', which='major', labelsize=12, pad=0) 
    ax[0].tick_params(axis='z', which='major', labelsize=12, pad=4) 
    ax[0].set_title('Information cost $\Delta v(a_{\max})$', y = 1, fontsize = 14)
#    ax[0].view_init(elev=30, azim=40)
    
    # Optimal contribution
    ir_min, ir_max = np.floor(np.min(info_rent)/5)*5, np.ceil(np.max(info_rent)/5)*5

    # Optimal reward

    ax[1].plot_wireframe(pmesh, nmesh, info_rent[:, :, 0], rstride=1, cstride=1, linewidth =0.3, color = 'crimson', label = 'high-cost type')
    ax[1].plot_surface(pmesh, nmesh, info_rent[:, :, 0], cmap=high_cost_cmap, alpha = 0.5)
    ax[1].plot_wireframe(pmesh, nmesh, info_rent[:, :, 1], rstride=1, cstride=1, linewidth =0.3, label = 'low-cost type')
    ax[1].plot_surface(pmesh, nmesh, info_rent[:, :, 1], cmap=low_cost_cmap, alpha = 0.5)

    ax[1].zaxis.set_major_locator(LinearLocator(3))
    ax[1].zaxis.set_major_formatter('{x:.0f}')
    ax[1].yaxis.set_major_locator(LinearLocator(3))
    ax[1].yaxis.set_major_formatter('{x:.0f}')
    ax[1].set_ylim(0,100)
    
    ax[1].set_xlim(0, 1)
    ax[1].set_zlim(0, ir_max)
    ax[1].xaxis.set_major_locator(LinearLocator(3))
    ax[1].xaxis.set_major_formatter('{x:.01f}')

    # make the panes transparent
    ax[1].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax[1].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax[1].zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax[1].xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax[1].yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax[1].zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    ax[1].set_xlabel('$p_1$', fontsize =12)
    ax[1].set_ylabel('$N$', fontsize =12)
    ax[1].tick_params(axis='both', which='major', labelsize=12, pad=0) 
    ax[1].set_title('Information rent $\Delta u_i$', y = 1, fontsize = 14)

    # Create custom legend handles with thicker lines
    line1 = mlines.Line2D([], [], color='crimson', linewidth=1, label='high-cost type')
    line2 = mlines.Line2D([], [], linewidth=1, label='low-cost type')

    # Create a single shared legend
    fig.legend(handles=[line1, line2], loc='lower center', bbox_to_anchor=(0.5, 0), ncol=2, frameon=False, fontsize=12)
    plt.subplots_adjust(wspace=0.15)
    plt.show()
    if save:
        if filename:
            fig.savefig(f'{filename}_3D.png')
        else:
            fig.savefig('simulation 1.png')  
    
    return None 
