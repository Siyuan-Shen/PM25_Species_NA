import toml
import numpy as np
import time

cfg = toml.load('./config.toml')

#######################################################################################
# outdir
Estimation_outdir = cfg['Pathway']['Estimation-dir']['Estimation_outdir']
NorthAmerica_PM25_version =  cfg['Pathway']['Estimation-dir']['NorthAmerica_PM25_version']
NorthAmerica_PM25_special_name  = cfg['Pathway']['Estimation-dir']['NorthAmerica_PM25_special_name']
BC_NorthAmerica_PM25_version =  cfg['Pathway']['Estimation-dir']['BC_NorthAmerica_PM25_version']
BC_NorthAmerica_PM25_special_name  = cfg['Pathway']['Estimation-dir']['BC_NorthAmerica_PM25_special_name']
#######################################################################################
# Estimation Switch

Estimation_Switch = cfg['Estimation-Settings']['Estimation_Switch']
Train_model_Switch = cfg['Estimation-Settings']['Train_model_Switch']
Map_estimation_Switch = cfg['Estimation-Settings']['Map_estimation_Switch']
Estimation_visualization_Switch = cfg['Estimation-Settings']['Estimation_visualization_Switch']
Estimation_PWMPM_Cal_Switch = cfg['Estimation-Settings']['Estimation_PWMPM_Cal_Switch']
Derive_combinedGeo_MapData_Switch = cfg['Estimation-Settings']['Derive_combinedGeo_MapData_Switch']

#######################################################################################
# Estiamtion Training Settings

Training_Settings = cfg['Estimation-Settings']['Training_Settings']
Training_beginyears = Training_Settings['beginyears']
Training_endyears   = Training_Settings['endyears']
Training_training_months = Training_Settings['training_months']

#######################################################################################
# Estimation-Settings Map_Estimation_Settings

Map_Estimation_Settings = cfg['Estimation-Settings']['Map_Estimation_Settings']
Estimation_years = Map_Estimation_Settings['Estimation_years']
Estiamtion_months = Map_Estimation_Settings['Estiamtion_months']
Estiamtion_trained_beginyears = Map_Estimation_Settings['Estiamtion_trained_beginyears']
Estiamtion_trained_endyears   = Map_Estimation_Settings['Estiamtion_trained_endyears']
Estiamtion_trained_months     = Map_Estimation_Settings['Estiamtion_trained_months']
Extent = Map_Estimation_Settings['Extent']
Estimation_ForcedSlopeUnity = Map_Estimation_Settings['Estimation_ForcedSlopeUnity']
#######################################################################################
# Visualization Settings
Visualization_Settings = cfg['Estimation-Settings']['Visualization_Settings']

Map_Plot_Switch                                       = Visualization_Settings['Map_Plot_Switch']       # Switch for plotting the map of estimated concentration. 
ForcedSlopeUnity_Map_Plot_Switch                      = Visualization_Settings['ForcedSlopeUnity_Map_Plot_Switch']
Map_Plot_YEARS                                        = Visualization_Settings['Map_Plot_YEARS']
Map_Plot_MONTHS                                       = Visualization_Settings['Map_Plot_MONTHS']
Map_Plot_Area                                         = Visualization_Settings['Map_Plot_Area']
Map_Plot_Extent                                       = Visualization_Settings['Map_Plot_Extent']

#######################################################################################
# Combine With Geophysical Settings
Coefficient_start_distance  = cfg['Estimation-Settings']['CombineWithGeophysical_Settings']['Coefficient_start_distance']

#######################################################################################
# Population-Weighted Mean PM Analysis
PWM_PM_Calculation_Settings = cfg['Estimation-Settings']['PWM-PM_Calculation_Settings']
Monthly_Analysis_Switch                               = PWM_PM_Calculation_Settings['Monthly_Analysis_Switch']
Annual_Analysis_Switch                                = PWM_PM_Calculation_Settings['Annual_Analysis_Switch']
NorthAmerica_Analysis_Switch                          = PWM_PM_Calculation_Settings['NorthAmerica_Analysis_Switch']
UnitedStates_Analysis_Switch                          = PWM_PM_Calculation_Settings['UnitedStates_Analysis_Switch']
Canada_Analysis_Switch                                = PWM_PM_Calculation_Settings['Canada_Analysis_Switch']
Analysis_YEARS                                        = PWM_PM_Calculation_Settings['Analysis_YEARS']
Analysis_MONTH                                        = PWM_PM_Calculation_Settings['Analysis_MONTH']


#######################################################################################################################
#######################################################################################################################
###################################################### INPUT VARIABLES ################################################
#######################################################################################################################
#######################################################################################################################

NorthAmerica_PM25_version           = NorthAmerica_PM25_version#'v1.5.0'#
NorthAmerica_PM25_special_name      = NorthAmerica_PM25_special_name #'_MSE_forBC_add_Population'#'_BenchMark'# #  
BC_NorthAmerica_PM25_version           = BC_NorthAmerica_PM25_version#'v1.5.0' #'v1.7.2'
BC_NorthAmerica_PM25_special_name      = BC_NorthAmerica_PM25_special_name #  '_MSE_forBC_add_Population'#'_BenchMark' #  

GeoPM25_AOD_ETA_version             = 'gPM25-20240604'
GCC_version                         = 'MERRASPEC-GCV11.GLandNested-20240607-RH35-199801-202312-wSA'

GeoPM25_AOD_ETA_input_indir         = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/GeoPM25_AOD_ETA_input/'
GCHP_GeoPM25_AOD_ETA_input_indir    = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/GCHP_GeoPM25_AOD_ETA_input/'
GeoSpecies_input_indir              = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/GeoSpecies/'
GEOS_Chem_input_indir               = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/GEOS-Chem_input/'
GCHP_input_indir                    = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/GCHP_input/'
Anthropogenic_Emissions_input_indir = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/Anthropogenic_Emissions_input/'
Offline_Emissions_input_indir       = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/Offline_Emissions_input/'
Meteorology_input_indir             = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/Meteorology_input/'
LandCover_input_indir               = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/LandCover_input/'
Geographical_Variables_input_indir  = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/Geographical_Variables_input/'
Global_CNN_PM25_input_indir         = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/GL_CNN_PM25/'
NA_CNN_PM25_input_indir             = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/NorthAmerica_CNN_PM25/{}/'.format(NorthAmerica_PM25_version)
GFED4_input_indir                   = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/GFED4_Emissions_input/'
Population_input_indir              = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/Population_input/'
Month_of_Year_input_indir           = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/Month_of_Year_input/'
OpenStreetMap_log_road_indir        = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/OpenStreetMap_log_road_map_data/'
OpenStreetMap_road_density_indir    = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/OpenStreetMap_RoadDensity_input/'
OpenStreetMap_nearest_dist_indir    = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/OpenStreetMap_RoadDensity_NearestDistances_forEachPixels_input/'

def inputfiles_table(YYYY, MM):
    inputfiles_dic = {
        #####################[Variables from Satellite and GCC] ###################
        'EtaAOD_Bias'        : GeoPM25_AOD_ETA_input_indir + '{}/{}/ttETAAODBIAS_001x001_NA_map_{}{}.npy'.format(GeoPM25_AOD_ETA_version,YYYY,YYYY,MM),
        'EtaCoastal'         : GeoPM25_AOD_ETA_input_indir + '{}/{}/ttETACOASTAL_001x001_NA_map_{}{}.npy'.format(GeoPM25_AOD_ETA_version,YYYY,YYYY,MM),
        'EtaMixing'          : GeoPM25_AOD_ETA_input_indir + '{}/{}/ttETAMIXING_001x001_NA_map_{}{}.npy'.format(GeoPM25_AOD_ETA_version,YYYY,YYYY,MM),
        'EtaSGAOD_Bias'      : GeoPM25_AOD_ETA_input_indir + '{}/{}/ttETASGAODBIAS_001x001_NA_map_{}{}.npy'.format(GeoPM25_AOD_ETA_version,YYYY,YYYY,MM),
        'EtaSGTOPO_Bias'     : GeoPM25_AOD_ETA_input_indir + '{}/{}/ttETASGTOPOBIAS_001x001_NA_map_{}{}.npy'.format(GeoPM25_AOD_ETA_version,YYYY,YYYY,MM),
        'AOD'                : GeoPM25_AOD_ETA_input_indir + '{}/{}/AOD_001x001_NA_map_{}{}.npy'.format(GeoPM25_AOD_ETA_version,YYYY,YYYY,MM),
        'ETA'                : GeoPM25_AOD_ETA_input_indir + '{}/{}/ETA_001x001_NA_map_{}{}.npy'.format(GeoPM25_AOD_ETA_version,YYYY,YYYY,MM),
        'GeoPM25'            : GeoPM25_AOD_ETA_input_indir + '{}/{}/geophysical_PM25_001x001_NA_map_{}{}.npy'.format(GeoPM25_AOD_ETA_version,YYYY,YYYY,MM),
        
        #####################[Variables from Satellite and GCHP] ###################
        'ETA_GCHP'              : GCHP_GeoPM25_AOD_ETA_input_indir + '{}/ETA_GCHP_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'AOD_GCHP'              : GCHP_GeoPM25_AOD_ETA_input_indir + '{}/AOD_GCHP_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GeoPM25_GCHP'          : GCHP_GeoPM25_AOD_ETA_input_indir + '{}/GeoPM25_GCHP_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'ttETAAODBIAS_GCHP'     : GCHP_GeoPM25_AOD_ETA_input_indir + '{}/ttETAAODBIAS_GCHP_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'ttETACOASTAL_GCHP'     : GCHP_GeoPM25_AOD_ETA_input_indir + '{}/ttETACOASTAL_GCHP_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'ttETAMIXING_GCHP'      : GCHP_GeoPM25_AOD_ETA_input_indir + '{}/ttETAMIXING_GCHP_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'ttETASGAODBIAS_GCHP'   : GCHP_GeoPM25_AOD_ETA_input_indir + '{}/ttETASGAODBIAS_GCHP_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'ttETASGTOPOBIAS_GCHP'  : GCHP_GeoPM25_AOD_ETA_input_indir + '{}/ttETASGTOPOBIAS_GCHP_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'ttETACONSTEXTTOPO_GCHP': GCHP_GeoPM25_AOD_ETA_input_indir + '{}/ttETACONSTEXTTOPO_GCHP_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'ttETAPROFILE_GCHP'     : GCHP_GeoPM25_AOD_ETA_input_indir + '{}/ttETAPROFILE_GCHP_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'ttETARHSENS_GCHP'      : GCHP_GeoPM25_AOD_ETA_input_indir + '{}/ttETARHSENS_GCHP_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),

        #####################[Variables from Geophysical Species] ###################
        'GeoNIT'             : GeoSpecies_input_indir + '{}/{}/Geophysical_NO3_001x001_NA_map_{}{}{}.npy'.format(YYYY,NorthAmerica_PM25_version,YYYY,MM,NorthAmerica_PM25_special_name),
        'GeoNH4'             : GeoSpecies_input_indir + '{}/{}/Geophysical_NH4_001x001_NA_map_{}{}{}.npy'.format(YYYY,NorthAmerica_PM25_version,YYYY,MM,NorthAmerica_PM25_special_name),
        'GeoSO4'             : GeoSpecies_input_indir + '{}/{}/Geophysical_SO4_001x001_NA_map_{}{}{}.npy'.format(YYYY,NorthAmerica_PM25_version,YYYY,MM,NorthAmerica_PM25_special_name),
        'GeoBC'              : GeoSpecies_input_indir + '{}/{}/Geophysical_BC_001x001_NA_map_{}{}{}.npy'.format(YYYY,BC_NorthAmerica_PM25_version,YYYY,MM,BC_NorthAmerica_PM25_special_name),
        'GeoOM'              : GeoSpecies_input_indir + '{}/{}/Geophysical_OM_001x001_NA_map_{}{}{}.npy'.format(YYYY,NorthAmerica_PM25_version,YYYY,MM,NorthAmerica_PM25_special_name),
        'GeoDUST'            : GeoSpecies_input_indir + '{}/{}/Geophysical_DUST_001x001_NA_map_{}{}{}.npy'.format(YYYY,NorthAmerica_PM25_version,YYYY,MM,NorthAmerica_PM25_special_name),
        'GeoSS'              : GeoSpecies_input_indir + '{}/{}/Geophysical_SS_001x001_NA_map_{}{}{}.npy'.format(YYYY,NorthAmerica_PM25_version,YYYY,MM,NorthAmerica_PM25_special_name),


        ##################### [Variables from GEOS-Chem] ###################
        'GC_PM25'            : GEOS_Chem_input_indir + '{}/{}/PM25_001x001_NA_map_{}{}.npy'.format(GCC_version,YYYY,YYYY,MM),
        'GC_NH4'             : GEOS_Chem_input_indir + '{}/{}/NH4_001x001_NA_map_{}{}.npy'.format(GCC_version,YYYY,YYYY,MM),
        'GC_SO4'             : GEOS_Chem_input_indir + '{}/{}/SO4_001x001_NA_map_{}{}.npy'.format(GCC_version,YYYY,YYYY,MM),
        'GC_NIT'             : GEOS_Chem_input_indir + '{}/{}/NIT_001x001_NA_map_{}{}.npy'.format(GCC_version,YYYY,YYYY,MM),
        'GC_SOA'             : GEOS_Chem_input_indir + '{}/{}/SOA_001x001_NA_map_{}{}.npy'.format(GCC_version,YYYY,YYYY,MM),
        'GC_OC'              : GEOS_Chem_input_indir + '{}/{}/OC_001x001_NA_map_{}{}.npy'.format(GCC_version,YYYY,YYYY,MM),
        'GC_OM'              : GEOS_Chem_input_indir + '{}/{}/OM_001x001_NA_map_{}{}.npy'.format(GCC_version,YYYY,YYYY,MM),
        'GC_BC'              : GEOS_Chem_input_indir + '{}/{}/BC_001x001_NA_map_{}{}.npy'.format(GCC_version,YYYY,YYYY,MM),
        'GC_DST'             : GEOS_Chem_input_indir + '{}/{}/DST_001x001_NA_map_{}{}.npy'.format(GCC_version,YYYY,YYYY,MM),
        'GC_SSLT'            : GEOS_Chem_input_indir + '{}/{}/SSLT_001x001_NA_map_{}{}.npy'.format(GCC_version,YYYY,YYYY,MM),


        ##################### [Variables from GEOS-Chem] ###################
        'GCHP_PM25'          : GCHP_input_indir + '{}/PM25_001x001_NA_GCHP_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GCHP_NH4'           : GCHP_input_indir + '{}/NH4_001x001_NA_GCHP_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GCHP_SO4'           : GCHP_input_indir + '{}/SO4_001x001_NA_GCHP_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GCHP_NIT'           : GCHP_input_indir + '{}/NIT_001x001_NA_GCHP_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GCHP_SOA'           : GCHP_input_indir + '{}/SOA_001x001_NA_GCHP_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GCHP_OC'            : GCHP_input_indir + '{}/OC_001x001_NA_GCHP_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GCHP_OM'            : GCHP_input_indir + '{}/OM_001x001_NA_GCHP_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GCHP_BC'            : GCHP_input_indir + '{}/BC_001x001_NA_GCHP_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GCHP_DST'           : GCHP_input_indir + '{}/DST_001x001_NA_GCHP_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GCHP_SSLT'          : GCHP_input_indir + '{}/SSLT_001x001_NA_GCHP_map_{}{}.npy'.format(YYYY,YYYY,MM),

         ##################### [Variables from Global CNN-Based OM25 estimation] ###################
        'GL_CNN_PM25'        : Global_CNN_PM25_input_indir + '/vManuscript-2023May/{}/GL-prediction-cnn-PM25_28Channel_ResNet1111_SigmoidMSELoss_alpha0d005_beta8d0_gamma3d0_lambda1-0d2_ForceSlopeFalse_{}{}_NA.npy'.format(YYYY,YYYY,MM),
        'NA_CNN_PM25'        : NA_CNN_PM25_input_indir     + '{}/NorthAmerica-prediction-cnn-PM25_NA_{}{}{}.npy'.format(YYYY,YYYY,MM,NorthAmerica_PM25_special_name),
        
        ##################### [Variables from CEDS Emissions] ###################
        'NH3_anthro_emi'     : Anthropogenic_Emissions_input_indir + '{}/NH3-em-anthro_CMIP_v2023-04_CEDS_Total_001x001_NA_{}{}.npy'.format(YYYY,YYYY,MM),
        'SO2_anthro_emi'     : Anthropogenic_Emissions_input_indir + '{}/SO2-em-anthro_CMIP_v2023-04_CEDS_Total_001x001_NA_{}{}.npy'.format(YYYY,YYYY,MM),
        'NO_anthro_emi'      : Anthropogenic_Emissions_input_indir + '{}/NO-em-anthro_CMIP_v2023-04_CEDS_Total_001x001_NA_{}{}.npy'.format(YYYY,YYYY,MM),
        'N2O_anthro_emi'     : Anthropogenic_Emissions_input_indir + '{}/N2O-em-anthro_CMIP_v2023-04_CEDS_Total_001x001_NA_{}{}.npy'.format(YYYY,YYYY,MM),
        'OC_anthro_emi'      : Anthropogenic_Emissions_input_indir + '{}/OC-em-anthro_CMIP_v2023-04_CEDS_Total_001x001_NA_{}{}.npy'.format(YYYY,YYYY,MM),
        'BC_anthro_emi'      : Anthropogenic_Emissions_input_indir + '{}/BC-em-anthro_CMIP_v2023-04_CEDS_Total_001x001_NA_{}{}.npy'.format(YYYY,YYYY,MM),
        'NMVOC_anthro_emi'   : Anthropogenic_Emissions_input_indir + '{}/NMVOC-em-anthro_CMIP_v2023-04_CEDS_Total_001x001_NA_{}{}.npy'.format(YYYY,YYYY,MM),

        ##################### [Variables from Offline Natural Emissions] ###################
        'DST_offline_emi'    : Offline_Emissions_input_indir + '{}/DST-em-EMI_Total_001x001_NA_{}{}.npy'.format(YYYY,YYYY,MM),
        'SSLT_offline_emi'   : Offline_Emissions_input_indir + '{}/SSLT-em-EMI_Total_001x001_NA_{}{}.npy'.format(YYYY,YYYY,MM),
        
        ##################### [Variables from GFED4 Dry Matter Emissions] ###################

        'Total_DM'           : GFED4_input_indir + '{}/GFED4-DM_TOTL_001x001_{}{}.npy'.format(YYYY, YYYY, MM),

        ##################### [Variables from Meteorology] ###################
        'PBLH'               : Meteorology_input_indir + '{}/PBLH_001x001_GL_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'PRECTOT'            : Meteorology_input_indir + '{}/PRECTOT_001x001_GL_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'RH'                 : Meteorology_input_indir + '{}/RH_001x001_GL_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'T2M'                : Meteorology_input_indir + '{}/T2M_001x001_GL_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'U10M'               : Meteorology_input_indir + '{}/U10M_001x001_GL_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'V10M'               : Meteorology_input_indir + '{}/V10M_001x001_GL_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'USTAR'              : Meteorology_input_indir + '{}/USTAR_001x001_GL_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GWETTOP'            : Meteorology_input_indir + '{}/GWETTOP_001x001_GL_map_{}{}.npy'.format(YYYY,YYYY,MM),
        ##################### [Variables from Land Cover] ###################
        'Crop_Nat_Vege_Mos'  : LandCover_input_indir + 'Cropland-Natural-Vegetation-Mosaics/Cropland-Natural-Vegetation-Mosaics-MCD12C1_LandCover_001x001_NA_{}.npy'.format(YYYY),
        'Permanent_Wetlands' : LandCover_input_indir + 'Permanent-Wetlands/Permanent-Wetlands-MCD12C1_LandCover_001x001_NA_{}.npy'.format(YYYY),
        'Croplands'          : LandCover_input_indir + 'Croplands/Croplands-MCD12C1_LandCover_001x001_NA_{}.npy'.format(YYYY),
        'Urban_Builtup_Lands': LandCover_input_indir + 'Urban-Builtup-Lands/Urban-Builtup-Lands-MCD12C1_LandCover_001x001_NA_{}.npy'.format(YYYY),
        
        ##################### [Open Street Map log data] ###################
        'log_major_roads'      : OpenStreetMap_log_road_indir + 'OpenStreetMap-major_roads-LogRoadMap_001x001.npy',
        'log_major_roads_dist' : OpenStreetMap_log_road_indir + 'OpenStreetMap-major_roads_NearestDistances-LogRoadMap_001x001.npy',
        'log_minor_roads'      : OpenStreetMap_log_road_indir + 'OpenStreetMap-minor_roads-LogRoadMap_001x001.npy',
        'log_minor_roads_dist' : OpenStreetMap_log_road_indir + 'OpenStreetMap-minor_roads_NearestDistances-LogRoadMap_001x001.npy',
        'log_motorway'         : OpenStreetMap_log_road_indir + 'OpenStreetMap-motorway-LogRoadMap_001x001.npy',
        'log_motorway_dist'    : OpenStreetMap_log_road_indir + 'OpenStreetMap-motorway_NearestDistances-LogRoadMap_001x001.npy',
        'log_primary'          : OpenStreetMap_log_road_indir + 'OpenStreetMap-primary-LogRoadMap_001x001.npy',
        'log_primary_dist'     : OpenStreetMap_log_road_indir + 'OpenStreetMap-primary_NearestDistances-LogRoadMap_001x001.npy',
        'log_secondary'        : OpenStreetMap_log_road_indir + 'OpenStreetMap-secondary-LogRoadMap_001x001.npy',
        'log_secondary_dist'   : OpenStreetMap_log_road_indir + 'OpenStreetMap-secondary_NearestDistances-LogRoadMap_001x001.npy',
        'log_trunk'            : OpenStreetMap_log_road_indir + 'OpenStreetMap-trunk-LogRoadMap_001x001.npy',
        'log_trunk_dist'       : OpenStreetMap_log_road_indir + 'OpenStreetMap-trunk_NearestDistances-LogRoadMap_001x001.npy',
        'log_unclassified'     : OpenStreetMap_log_road_indir + 'OpenStreetMap-unclassified-LogRoadMap_001x001.npy',
        'log_unclassified_dist': OpenStreetMap_log_road_indir + 'OpenStreetMap-unclassified_NearestDistances-LogRoadMap_001x001.npy',
        'log_residential'      : OpenStreetMap_log_road_indir + 'OpenStreetMap-residential-LogRoadMap_001x001.npy',
        'log_residential_dist' : OpenStreetMap_log_road_indir + 'OpenStreetMap-residential_NearestDistances-LogRoadMap_001x001.npy',

        ##################### [Open Street Map Road Density] ###################
        'major_roads'        : OpenStreetMap_road_density_indir + '{}/OpenStreetMap-NorthAmerica-major_roads-RoadDensityMap_{}.npy'.format(YYYY,YYYY),
        'minor_roads'        : OpenStreetMap_road_density_indir + '{}/OpenStreetMap-NorthAmerica-minor_roads-RoadDensityMap_{}.npy'.format(YYYY,YYYY),
        'motorway'           : OpenStreetMap_road_density_indir + '{}/OpenStreetMap-NorthAmerica-motorway-RoadDensityMap_{}.npy'.format(YYYY,YYYY),
        'primary'            : OpenStreetMap_road_density_indir + '{}/OpenStreetMap-NorthAmerica-primary-RoadDensityMap_{}.npy'.format(YYYY,YYYY),
        'secondary'          : OpenStreetMap_road_density_indir + '{}/OpenStreetMap-NorthAmerica-secondary-RoadDensityMap_{}.npy'.format(YYYY,YYYY),
        'trunk'              : OpenStreetMap_road_density_indir + '{}/OpenStreetMap-NorthAmerica-trunk-RoadDensityMap_{}.npy'.format(YYYY,YYYY),
        'unclassified'       : OpenStreetMap_road_density_indir + '{}/OpenStreetMap-NorthAmerica-unclassified-RoadDensityMap_{}.npy'.format(YYYY,YYYY),
        'residential'        : OpenStreetMap_road_density_indir + '{}/OpenStreetMap-NorthAmerica-residential-RoadDensityMap_{}.npy'.format(YYYY,YYYY),
        
        ##################### [Open Street Map Road Density nearest distances] ###################
        'major_roads_dist'   : OpenStreetMap_nearest_dist_indir + 'major_roads/OpenStreetMap-major_roads-NearestDistanceforEachPixel_001x001_NA_{}.npy'.format(YYYY),
        'minor_roads_dist'   : OpenStreetMap_nearest_dist_indir + 'minor_roads/OpenStreetMap-minor_roads-NearestDistanceforEachPixel_001x001_NA_{}.npy'.format(YYYY),
        'motorway_dist'      : OpenStreetMap_nearest_dist_indir + 'motorway_NearestDistances/OpenStreetMap-motorway_NearestDistances-NearestDistanceforEachPixel_001x001_NA_{}.npy'.format(YYYY),
        'primary_dist'       : OpenStreetMap_nearest_dist_indir + 'primary_NearestDistances/OpenStreetMap-primary_NearestDistances-NearestDistanceforEachPixel_001x001_NA_{}.npy'.format(YYYY),
        'secondary_dist'     : OpenStreetMap_nearest_dist_indir + 'secondary_NearestDistances/OpenStreetMap-secondary_NearestDistances-NearestDistanceforEachPixel_001x001_NA_{}.npy'.format(YYYY),
        'trunk_dist'         : OpenStreetMap_nearest_dist_indir + 'trunk_NearestDistances/OpenStreetMap-trunk_NearestDistances-NearestDistanceforEachPixel_001x001_NA_{}.npy'.format(YYYY),
        'unclassified_dist'  : OpenStreetMap_nearest_dist_indir + 'unclassified_NearestDistances/OpenStreetMap-unclassified_NearestDistances-NearestDistanceforEachPixel_001x001_NA_{}.npy'.format(YYYY),
        'residential_dist'   : OpenStreetMap_nearest_dist_indir + 'residential_NearestDistances/OpenStreetMap-residential_NearestDistances-NearestDistanceforEachPixel_001x001_NA_{}.npy'.format(YYYY),

        ##################### [Geographical Information] ###################
        'S1'                 : Geographical_Variables_input_indir + 'Spherical_Coordinates/Spherical_Coordinates_1.npy',
        'S2'                 : Geographical_Variables_input_indir + 'Spherical_Coordinates/Spherical_Coordinates_2.npy',
        'S3'                 : Geographical_Variables_input_indir + 'Spherical_Coordinates/Spherical_Coordinates_3.npy',
        'Lat'                : '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/tSATLAT_NA_MAP.npy',
        'Lon'                : '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/tSATLON_NA_MAP.npy',
        'elevation'          : Geographical_Variables_input_indir + 'elevation/elevartion_001x001_NA.npy',

        ###################### [Population Information] ####################
        'Population'         : Population_input_indir + 'WorldPopGrid-{}-0.01.npy'.format(YYYY),
        
        ###################### [Temporal Information] ####################

        'Month_of_Year'      : Month_of_Year_input_indir + '/Month_of_Year_{}.npy'.format(MM),
    }
    return inputfiles_dic
