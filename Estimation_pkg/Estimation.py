import numpy as np
import time
import os
import gc
import netCDF4 as nc
from Estimation_pkg.utils import *
from Estimation_pkg.data_func import *
from Estimation_pkg.training_func import Train_Model_forEstimation
from Estimation_pkg.predict_func import map_predict,map_final_output
from Estimation_pkg.iostream import load_map_data, load_trained_model_forEstimation, save_final_map_data, load_estimation_map_data,save_combinedGeo_map_data

from Training_pkg.iostream import load_TrainingVariables
from Training_pkg.iostream import Learning_Object_Datasets
from Training_pkg.data_func import normalize_Func
from Training_pkg.utils import *

from Evaluation_pkg.utils import *

def Estimation_Func():
    typeName   = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=log_species, species=species)
    nchannel = Get_nchannels(channel_names=channel_names)
    if Train_model_Switch:
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        Train_Model_forEstimation(train_beginyears=Training_beginyears,train_endyears=Training_endyears,width=width,height=height,sitesnumber=sitesnumber,start_YYYY=start_YYYY,TrainingDatasets=TrainingDatasets)
        del width, height, sitesnumber,start_YYYY, TrainingDatasets 
        gc.collect()
    
    if Map_estimation_Switch:
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        Initial_Normalized_TrainingData, input_mean, input_std = normalize_Func(inputarray=TrainingDatasets)
        true_input, mean, std = Learning_Object_Datasets(bias=bias,Normalized_bias=normalize_bias,Normlized_Speices=normalize_species,Absolute_Species=absolute_species,Log_PM25=log_species,species=species)
    
        del  TrainingDatasets, Initial_Normalized_TrainingData,true_input
        gc.collect()
        MM = ['01','02','03','04','05','06','07','08','09','10','11','12']
        for imodel in range(len(Estiamtion_trained_beginyears)):
            model = load_trained_model_forEstimation(model_outdir=model_outdir,typeName=typeName,version=version,species=species, nchannel=nchannel,special_name=special_name,
                                                             beginyear=Estiamtion_trained_beginyears[imodel],endyear=Estiamtion_trained_endyears[imodel], width=width, height=height)
            for YEAR in Estimation_years[imodel]:
                for imonth in Estiamtion_months:
                    print('YEAR: {}, MONTH: {}'.format(YEAR,MM[imonth]))
                    map_input = load_map_data(channel_names=channel_names,YYYY=YEAR,MM=MM[imonth])
                    final_map_data = map_predict(inputmap=map_input,model=model,train_mean=input_mean,train_std=input_std,extent=Extent,width=width,nchannel=nchannel,YYYY=YEAR,MM=MM[imonth])
                    final_map_data = map_final_output(output=final_map_data,extent=Extent,YYYY=YEAR,MM=MM[imonth],SPECIES=species,bias=bias,
                                                      normalize_bias=normalize_bias,normalize_species=normalize_species,absolute_species=absolute_species,
                                                      log_species=log_species,mean=mean,std=std)
                    save_final_map_data(final_data=final_map_data,YYYY=YEAR,MM=MM[imonth],extent=Extent,SPECIES=species,version=version,special_name=special_name)
                    del map_input, final_map_data
                    gc.collect()
    if Derive_combinedGeo_MapData_Switch:
        coefficients = Get_coefficient_map()
        for imodel in range(len(Estiamtion_trained_beginyears)):
            for YEAR in Estimation_years[imodel]:
                for imonth in Estiamtion_months:
                    MM = ['01','02','03','04','05','06','07','08','09','10','11','12']
                    CNN_Species = load_estimation_map_data(YYYY=YEAR,MM=MM[imonth],SPECIES=species,version=version,
                                                           special_name=special_name)
                    Combined_species = Combine_CNN_GeophysicalSpecies(CNN_Species=CNN_Species,coefficient=coefficients,YYYY=YEAR,MM=MM[imonth])
                    save_combinedGeo_map_data(final_data=Combined_species,YYYY=YEAR,MM=MM[imonth],extent=Extent,
                                              SPECIES=species,version=version,special_name=special_name)
                    
    return