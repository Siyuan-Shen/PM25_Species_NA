import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
from sklearn.metrics import mean_squared_error,r2_score
from Training_pkg.Statistic_Func import regress2, linear_regression
from visualization_pkg.utils import Loss_Accuracy_outdir

nrows = 2
ncols = 2
proj = ccrs.PlateCarree()
aspect = (179)/(60+70)
height = 5.0
width = aspect * height
vpad = 0.03 * height
hpad = 0.02 * width
hlabel = 0.12 * height*2
vlabel = 0.1 * height*2
hmargin = 0.03 * width
vmargin = 0.03 * height*2
cbar_height = 0.48 * height
cbar_width = 0.015 * width
cbar_height_2 = 0.9 * (height*2 - vlabel)
cbar_width_2 = 0.08 * (width + height*2)

figwidth = width + height + hmargin*2 + cbar_width_2
figheight = height*2 + vmargin*2

def regression_plot(plot_obs_pm25:np.array,plot_pre_pm25:np.array,
                    species, version, typeName, beginyear, MONTH, nchannel, special_name, width, height):
    
    fig_output_dir = Loss_Accuracy_outdir + '{}/{}/Figures/figures-Regression/'.format(species, version)

    if not os.path.isdir(fig_output_dir):
        os.makedirs(fig_output_dir)
    
    fig_outfile =  fig_output_dir +  '{}-{}-Regression_Figure_{}_{}_{}x{}_{}Channel{}.png'.format(typeName, species, beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name)
    

    H, xedges, yedges = np.histogram2d(plot_obs_pm25, plot_pre_pm25, bins=100)
    fig = plt.figure(figsize=(10, 10))
    #fig = plt.figure(figsize=(figwidth, figheight))
    extent = [0, max(xedges), 0, max(xedges)]
    RMSE = np.sqrt(mean_squared_error(plot_obs_pm25, plot_pre_pm25))
    RMSE = round(RMSE, 1)

    R2 = linear_regression(plot_obs_pm25, plot_pre_pm25)
    R2 = np.round(R2, 2)

    ax = plt.axes([0.1,0.1,0.8,0.8])  # [left, bottom, width, height]
    cbar_ax = plt.axes([0.91,0.2,0.03,0.6])
    regression_Dic = regress2(_x=plot_obs_pm25,_y=plot_pre_pm25,_method_type_1='ordinary least square',_method_type_2='reduced major axis',
    )
    b0,b1 = regression_Dic['intercept'], regression_Dic['slope']
    #b0, b1 = linear_slope(plot_obs_pm25,
    #                      plot_pre_pm25)
    b0 = round(b0, 2)
    b1 = round(b1, 2)

    extentlim = 2*np.mean(plot_obs_pm25)
    # im = ax.imshow(
    #    H, extent=extent,
    #    cmap= 'gist_rainbow',
    #   origin='lower',
    #  norm=colors.LogNorm(vmin=1, vmax=1e3))
    im = ax.hexbin(plot_obs_pm25, plot_pre_pm25,
                   cmap='autumn_r', norm=colors.LogNorm(vmin=1, vmax=100), extent=(0, extentlim, 0, extentlim),
                   mincnt=1)
    ax.plot([0, extentlim], [0, extentlim], color='black', linestyle='--')
    ax.plot([0, extentlim], [b0, b0 + b1 * extentlim], color='blue', linestyle='-')
    #ax.set_title('Comparsion of Modeled $PM_{2.5}$ and observations for '+area_name+' '+beginyear+' '+endyear)
    ax.set_xlabel('Observed {} concentration ($\mu g/m^3$)'.format(species), fontsize=32)
    ax.set_ylabel('Estimated {} concentration ($\mu g/m^3$)'.format(species), fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=28)

    ax.text(0, extentlim - 0.05 * extentlim, '$R^2 = $ {}'.format(R2), style='italic', fontsize=32)
    ax.text(0, extentlim - (0.05 + 0.064) * extentlim, '$RMSE = $' + str(RMSE)+'$\mu g/m^3$', style='italic', fontsize=32)
    if b1 > 0.0:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y = {}x {} {}'.format(abs(b1),return_sign(b0),abs(b0)) , style='italic',
            fontsize=32)
    elif b1 == 0.0:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y = ' + str(b0), style='italic',
            fontsize=32)
    else:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y=-{}x {} {}'.format(abs(b1),return_sign(b0),abs(b0)) , style='italic',
            fontsize=32)

    ax.text(0, extentlim - (0.05 + 0.064 * 3) * extentlim, 'N = ' + str(len(plot_pre_pm25)), style='italic',
            fontsize=32)
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical', shrink=1.0, ticks=[1, 10, 100])
    cbar.ax.set_yticklabels(['1', '10', r'$10^2$',], fontsize=24)
    cbar.set_label('Number of points', fontsize=28)

    fig.savefig(fig_outfile, dpi=1000,transparent = True,bbox_inches='tight' )
    plt.close()

def return_sign(number):
    if number < 0.0:
        return '-'
    elif number == 0.0:
        return ''
    else:
        return '+'   