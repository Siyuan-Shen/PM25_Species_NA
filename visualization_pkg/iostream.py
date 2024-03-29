import os
import netCDF4 as nc
from visualization_pkg.utils import *
from Estimation_pkg.utils import *


def save_loss_accuracy_figure(typeName, species,version,nchannel,width,height,special_name ):
    fig_outdir = Loss_Accuracy_outdir + '{}/{}/Figures/figures-Loss_Accuracy/'.format(species, version)
    if not os.path.isdir(fig_outdir):
        os.makedirs(fig_outdir)
    training_fig_outfile =  fig_outdir + 'SpatialCV_Training_{}_{}_{}_{}Channel_{}x{}{}.png'.format(typeName,species,version,nchannel,width,height,special_name)
    valid_fig_outfile =  fig_outdir + 'SpatialCV_Valid_{}_{}_{}_{}Channel_{}x{}{}.png'.format(typeName,species,version,nchannel,width,height,special_name)
    return training_fig_outfile, valid_fig_outfile

def save_estimation_map_figure(typeName, species,version,Area,YYYY,MM,nchannel,width,height,special_name ):
    fig_outdir = Estimation_Map_outdir + '{}/{}/Figures/figures-Estimation_Map/{}/'.format(species, version,YYYY)
    if not os.path.isdir(fig_outdir):
        os.makedirs(fig_outdir)
    estimation_map_fig_outfile =  fig_outdir + 'EstimationMap_{}_{}_{}_{}_{}{}_{}Channel_{}x{}{}.png'.format(typeName,species,version,Area,YYYY,MM,nchannel,width,height,special_name)
    return estimation_map_fig_outfile

def save_uncertainty_map_figure(typeName, species,version,Area,YYYY,MM,nchannel,width,height,special_name ):
    fig_outdir = Uncertainty_Map_outdir + '{}/{}/Figures/figures-Uncertainty_Map/{}/'.format(species, version,YYYY)
    if not os.path.isdir(fig_outdir):
        os.makedirs(fig_outdir)
    uncertainty_map_fig_outfile =  fig_outdir + 'UncertaintyMap_{}_{}_{}_{}_{}{}_{}Channel_{}x{}{}.png'.format(typeName,species,version,Area,YYYY,MM,nchannel,width,height,special_name)
    return uncertainty_map_fig_outfile


def load_Population_MapData(YYYY,MM):
    inputfiles = inputfiles_table(YYYY=YYYY,MM=MM)
    infile = inputfiles['Population']
    tempdata = np.load(infile)
    output = tempdata
    lat = np.linspace(10.005,69.995,6000)
    lon = np.linspace(-169.995,-40.005,13000)
    return output,lat,lon
    