import toml
import numpy as np
cfg = toml.load('./config.toml')

#######################################################################################
# Uncertainty Settings
Uncertainty_outdir = cfg['Pathway']['Estimation-dir']['Uncertainty_outdir']

Uncertainty_Settings = cfg['Uncertainty-Settings']
Uncertainty_Switch = Uncertainty_Settings['Uncertainty_Switch']
Derive_distances_map_Switch    = Uncertainty_Settings['Derive_distances_map_Switch']
Derive_rRMSE_map_Switch        = Uncertainty_Settings['Derive_rRMSE_map_Switch']
Derive_absolute_Uncertainty_map_Switch   = Uncertainty_Settings['Derive_absolute_Uncertainty_map_Switch']
Uncertainty_visualization_Switch = Uncertainty_Settings['Uncertainty_visualization_Switch']

Uncertainty_Map_Estimation_Settings = Uncertainty_Settings['Map_Estimation_Settings']
Uncertainty_Estimation_years  = Uncertainty_Map_Estimation_Settings['Estimation_years']
Uncertainty_Estimation_months = Uncertainty_Map_Estimation_Settings['Estiamtion_months']
Buffer_radii_forUncertainty   = Uncertainty_Map_Estimation_Settings['Buffer_radii_forUncertainty']

Uncertainty_Visualization_Settings = Uncertainty_Settings['Visualization_Settings']
Uncertainty_Plot_Switch = Uncertainty_Visualization_Settings['Uncertainty_Plot_Switch']
Uncertainty_Plot_YEARS  = Uncertainty_Visualization_Settings['Uncertainty_Plot_YEARS']
Uncertainty_Plot_MONTHS = Uncertainty_Visualization_Settings['Uncertainty_Plot_MONTHS']
Uncertainty_Plot_Area   = Uncertainty_Visualization_Settings['Uncertainty_Plot_Area']
Uncertainty_Plot_Extent = Uncertainty_Visualization_Settings['Uncertainty_Plot_Extent']

def get_extent_lat_lon_map(lat_index,lon_index,SATLAT,SATLON):
    extent_lat_map = np.zeros((len(lat_index),len(lon_index)),dtype=np.float32)
    extent_lon_map = np.zeros((len(lat_index),len(lon_index)),dtype=np.float32)
    for iy in range(len(lon_index)):
        extent_lat_map[:,iy] = SATLAT[lat_index]
    for ix in range(len(lat_index)):
        extent_lon_map[ix,:] = SATLON[lon_index]
    return extent_lat_map,extent_lon_map