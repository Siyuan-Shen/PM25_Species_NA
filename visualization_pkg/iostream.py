import os
from visualization_pkg.utils import *


def save_loss_accuracy_figure(typeName, species,version,nchannel,width,height,special_name ):
    fig_outdir = Loss_Accuracy_outdir + '{}/{}/Figures/figures-Loss_Accuracy/'.format(species, version)
    if not os.path.isdir(fig_outdir):
        os.makedirs(fig_outdir)
    
    fig_outfile =  fig_outdir + 'SpatialCV_{}_{}_{}_{}Channel_{}x{}{}.png'.format(typeName,species,version,nchannel,width,height,special_name)
    return fig_outfile