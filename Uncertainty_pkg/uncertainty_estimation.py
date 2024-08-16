import numpy as np
import netCDF4 as nc
import os 

from Uncertainty_pkg.data_func import *
from Uncertainty_pkg.iostream import *
from Uncertainty_pkg.utils import *

from Training_pkg.iostream import load_TrainingVariables
from Training_pkg.utils import *

from Estimation_pkg.iostream import load_estimation_map_data,load_Annual_estimation_map_data

from visualization_pkg.Assemble_Func import plot_save_uncertainty_map_figure
def Derive_Estimation_Uncertainty():
    MONTH = ['Annual','01','02','03','04','05','06','07','08','09','10','11','12']
    if Derive_distances_map_Switch:
        get_nearest_site_distance_for_each_pixel()
    
    if Derive_rRMSE_map_Switch:
        distances_map = load_pixels_nearest_sites_distances_map()
        rRMSE = load_BLCO_rRMSE()
        for imonth in range(len(MONTH)):
            rRMSE_Map = convert_distance_to_rRMSE_uncertainty(buffer_radii=Buffer_radii_forUncertainty,BLOO_rRMSE=rRMSE[imonth,:],map_distances=distances_map)
            save_rRMSE_uncertainty_Map(Map_rRMSE=rRMSE_Map,MM=MONTH[imonth])
    
    if Derive_absolute_Uncertainty_map_Switch:
        for iyear in range(len(Uncertainty_Estimation_years)):
            for imonth in range(len(Uncertainty_Estimation_months)):
                print('Derive Absolute Uncertainty - YEAR:{}, MONTH:{}'.format(Uncertainty_Estimation_years[iyear],Uncertainty_Estimation_months[imonth]))
                if Uncertainty_Estimation_months[imonth] == 'Annual':
                    Estimation_Map, lat, lon = Uncertainty_Estimation_months(YYYY=Uncertainty_Estimation_years[iyear],
                                                          SPECIES=species,version=version,special_name=special_name)
                else:
                    Estimation_Map,lat, lon = load_estimation_map_data(YYYY=Uncertainty_Estimation_years[iyear],MM=Uncertainty_Estimation_months[imonth],
                                                          SPECIES=species,version=version,special_name=special_name)
                rRMSE_Map,lat, lon      = load_rRMSE_map_data(MM=Uncertainty_Estimation_months[imonth],version=version,special_name=special_name)
                print('rRMSE type:{}, Estimation type:{}'.format(type(rRMSE_Map),type(Estimation_Map)))
                Absolute_Uncertainty_Map = rRMSE_Map * Estimation_Map
                save_absolute_uncertainty_data(final_data=Absolute_Uncertainty_Map,YYYY=Uncertainty_Estimation_years[iyear],
                                               MM=Uncertainty_Estimation_months[imonth])
    if Uncertainty_visualization_Switch:
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        typeNAME = Get_typeName(bias=bias,normalize_bias=normalize_bias,normalize_species=normalize_species,
                                absolute_species=absolute_species,log_species=log_species,species=species)
        plot_save_uncertainty_map_figure(typeName=typeNAME,width=width,height=height,species=species,version=version,Area=Uncertainty_Plot_Area,PLOT_YEARS=Uncertainty_Estimation_years,PLOT_MONTHS=Uncertainty_Estimation_months)
    return