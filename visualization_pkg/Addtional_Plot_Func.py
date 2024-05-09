from click import style
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy as crt
import numpy as np
#from .Statistic_Func import Calculate_PWA_PM25, linear_regression,linear_slope
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import xarray as xr
import numpy as np
import numpy.ma as ma
import netCDF4 as nc
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.ticker as tick
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from Training_pkg.Statistic_Func import Calculate_PWA_PM25
from visualization_pkg.utils import crop_map_data

def plot_BLCO_test_train_buffers(train_index, test_index, excluded_index, sitelat, sitelon, buffer_radius,extent,fig_outfile):
    ax = plt.axes(projection=ccrs.PlateCarree())
    bottom_lat = extent[0]
    left_lon     = extent[2]
    up_lat       = extent[1]
    right_lon   = extent[3]
    extent = [left_lon,right_lon,bottom_lat,up_lat]
    ax.set_extent(extent)
    
    ax.add_feature(cfeat.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', facecolor='white'))
    ax.add_feature(cfeat.NaturalEarthFeature('physical', 'land', '50m', edgecolor='none', facecolor=cfeat.COLORS['land']))
    ax.add_feature(cfeat.BORDERS, linewidth=0.1)
    
    
    for isite in range(len(test_index)):
        
        ax.add_patch(mpatches.Circle(xy=[sitelon[test_index[isite]], sitelat[test_index[isite]]], radius=buffer_radius*0.01, color='red', alpha=0.1, transform=ccrs.PlateCarree(), zorder=30))
    plt.scatter(sitelon[test_index], sitelat[test_index], s=3,
            linewidths=0.1, marker='o', edgecolors='red',c='red',
            alpha=0.8,label='Test Sites',zorder=2)
    plt.scatter(sitelon[train_index], sitelat[train_index], s=1,
            linewidths=0.1, marker='o', edgecolors='black',c='black',
            alpha=0.8,label='Training Sites',zorder=2)
    plt.scatter(sitelon[excluded_index], sitelat[excluded_index], s=1,
            linewidths=0.1, marker='X',c='blue',
            alpha=0.5,label='Excluded Sites',zorder=2)
    plt.legend(fontsize='small',markerscale = 3.0,loc=4)
    ax.text(0, 0.15, '# of test sites: {}'.format(len(test_index)), style='italic', fontsize=32)
    ax.text(0, 0.15, '# of train sites: {}'.format(len(train_index)),style='italic', fontsize=32)
    ax.text(0, 0.15, '# of Exclude sites: {}'.format(len(excluded_index)), style='italic', fontsize=32)
    plt.savefig(fig_outfile, format='png', dpi=2000, transparent=True,bbox_inches='tight')
    plt.close()
    return