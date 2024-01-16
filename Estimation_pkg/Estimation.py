import numpy as np
import time
import os
import gc
import netCDF4 as nc
from Estimation_pkg.utils import *
from Estimation_pkg.training_func import Train_Model_forEstimation
from Estimation_pkg.predict_func import map_predict
from Estimation_pkg.iostream import load_map_data, load_trained_model_forEstimation, save_final_map_data

from Training_pkg.iostream import load_TrainingVariables
from Training_pkg.data_func import normalize_Func
from Training_pkg.utils import *

from Evaluation_pkg.utils import *

def Estimation_Func():
    typeName   = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=log_species, species=species)
    
    
    if Train_model_Switch:
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        Train_Model_forEstimation(train_beginyears=Training_beginyears,train_endyears=Training_endyears,width=width,height=height,sitesnumber=sitesnumber,start_YYYY=start_YYYY,TrainingDatasets=TrainingDatasets)
        del width, height, sitesnumber,start_YYYY, TrainingDatasets 
        gc.collect()
    
    if Map_estimation_Switch:
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
        Initial_Normalized_TrainingData, input_mean, input_std = normalize_Func(inputarray=TrainingDatasets)
        del  TrainingDatasets, Initial_Normalized_TrainingData
        gc.collect()

        for imodel in range(len(Estiamtion_trained_beginyears)):
            model = load_trained_model_forEstimation(model_outdir=model_outdir,typeName=typeName,version=version,species=species, nchannel=len(channel_names),special_name=special_name,
                                                             beginyear=Estiamtion_trained_beginyears[imodel],endyear=Estiamtion_trained_endyears[imodel], width=width, height=height)
            for YEAR in Estimation_years[imodel]:
                for imonth in Estiamtion_months:
                    MM = ['01','02','03','04','05','06','07','08','09','10','11','12']
                    map_input = load_map_data(channel_names=channel_names,YYYY=YEAR,MM=MM[imonth])
                    final_map_data = map_predict(inputmap=map_input,model=model,train_mean=input_mean,train_std=input_std,extent=Extent,width=width,nchannel=len(channel_names),YYYY=YEAR,MM=MM[imonth])
                    save_final_map_data(final_data=final_map_data,YYYY=YEAR,MM=MM[imonth],extent=Extent,SPECIES=species,version=version,special_name=special_name)
                    del map_input, final_map_data
                    gc.collect()
    return