import toml
import numpy as np
import time

cfg = toml.load('./config.toml')

#######################################################################################
# outdir
Estimation_outdir = cfg['Pathway']['Estimation-dir']['Estimation_outdir']
NorthAmerica_PM25_version =  cfg['Pathway']['Estimation-dir']['NorthAmerica_PM25_version']
NorthAmerica_PM25_special_name  = cfg['Pathway']['Estimation-dir']['NorthAmerica_PM25_special_name']
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
NA_CNN_PM25_input_indir             = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/NorthAmerica_CNN_PM25/v1.0.0/'
GFED4_input_indir                   = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/GFED4_Emissions_input/'
Population_input_indir              = '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/Population_input/'

def inputfiles_table(YYYY, MM):
    inputfiles_dic = {
        #####################[Variables from Satellite and GCC] ###################
        'EtaAOD_Bias'        : GeoPM25_AOD_ETA_input_indir + '{}/ttETAAODBIAS_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'EtaCoastal'         : GeoPM25_AOD_ETA_input_indir + '{}/ttETACOASTAL_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'EtaMixing'          : GeoPM25_AOD_ETA_input_indir + '{}/ttETAMIXING_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'EtaSGAOD_Bias'      : GeoPM25_AOD_ETA_input_indir + '{}/ttETASGAODBIAS_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'EtaSGTOPO_Bias'     : GeoPM25_AOD_ETA_input_indir + '{}/ttETASGTOPOBIAS_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'AOD'                : GeoPM25_AOD_ETA_input_indir + '{}/AOD_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'ETA'                : GeoPM25_AOD_ETA_input_indir + '{}/ETA_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GeoPM25'            : GeoPM25_AOD_ETA_input_indir + '{}/geophysical_PM25_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        
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
        'GeoNIT'             : GeoSpecies_input_indir + '{}/Geophysical_NO3_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GeoNH4'             : GeoSpecies_input_indir + '{}/Geophysical_NH4_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GeoSO4'             : GeoSpecies_input_indir + '{}/Geophysical_SO4_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GeoBC'              : GeoSpecies_input_indir + '{}/Geophysical_BC_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GeoOM'              : GeoSpecies_input_indir + '{}/Geophysical_OM_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GeoDUST'            : GeoSpecies_input_indir + '{}/Geophysical_DUST_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GeoSS'              : GeoSpecies_input_indir + '{}/Geophysical_SS_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),

        ##################### [Variables from GEOS-Chem] ###################
        'GC_PM25'            : GEOS_Chem_input_indir + '{}/PM25_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GC_NH4'             : GEOS_Chem_input_indir + '{}/NH4_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GC_SO4'             : GEOS_Chem_input_indir + '{}/SO4_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GC_NIT'             : GEOS_Chem_input_indir + '{}/NIT_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GC_SOA'             : GEOS_Chem_input_indir + '{}/SOA_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GC_OC'              : GEOS_Chem_input_indir + '{}/OC_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GC_OM'              : GEOS_Chem_input_indir + '{}/OM_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GC_BC'              : GEOS_Chem_input_indir + '{}/BC_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GC_DST'             : GEOS_Chem_input_indir + '{}/DST_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GC_SSLT'            : GEOS_Chem_input_indir + '{}/SSLT_001x001_NA_map_{}{}.npy'.format(YYYY,YYYY,MM),


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
        'NA_CNN_PM25'        : NA_CNN_PM25_input_indir     + '{}/NorthAmerica-prediction-cnn-PM25_NA_{}{}_ResNet_Basic1111_2.npy'.format(YYYY,YYYY,MM),
        
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

        ##################### [Variables from Land Cover] ###################
        'Crop_Nat_Vege_Mos'  : LandCover_input_indir + 'Cropland-Natural-Vegetation-Mosaics/Cropland-Natural-Vegetation-Mosaics-MCD12C1_LandCover_001x001_NA_{}.npy'.format(YYYY),
        'Permanent_Wetlands' : LandCover_input_indir + 'Permanent-Wetlands/Permanent-Wetlands-MCD12C1_LandCover_001x001_NA_{}.npy'.format(YYYY),
        'Croplands'          : LandCover_input_indir + 'Croplands/Croplands-MCD12C1_LandCover_001x001_NA_{}.npy'.format(YYYY),
        'Urban_Builtup_Lands': LandCover_input_indir + 'Urban-Builtup-Lands/Urban-Builtup-Lands-MCD12C1_LandCover_001x001_NA_{}.npy'.format(YYYY),

        ##################### [Geographical Information] ###################
        'S1'                 : Geographical_Variables_input_indir + 'Spherical_Coordinates/Spherical_Coordinates_1.npy',
        'S2'                 : Geographical_Variables_input_indir + 'Spherical_Coordinates/Spherical_Coordinates_2.npy',
        'S3'                 : Geographical_Variables_input_indir + 'Spherical_Coordinates/Spherical_Coordinates_3.npy',
        'Lat'                : '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/tSATLAT_NA_MAP.npy',
        'Lon'                : '/my-projects/Projects/PM25_Speices_DL_2023/data/input_variables_map/tSATLON_NA_MAP.npy',
        'elevation'          : Geographical_Variables_input_indir + 'elevation/elevartion_001x001_NA.npy',

        ###################### [Population Information] ####################
        'Population'         : Population_input_indir + 'WorldPopGrid-{}-0.01.npy'.format(YYYY),
    }
    return inputfiles_dic
