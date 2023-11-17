import toml
import numpy as np
import time


cfg = toml.load('./config.toml')

############################ Spatial Cross-Validation ################################

Spatial_CrossValidation_Switch = cfg['Spatial-CrossValidation']['Spatial_CrossValidation_Switch'] # On/Off for Spatial Crosss Validation
Spatial_CV_LossAccuracy_plot_Switch = cfg['Spatial-CrossValidation']['Spatial_CV_LossAccuracy_plot_Switch']
regression_plot_switch   = cfg['Spatial-CrossValidation']['Visualization_Settings']['regression_plot_switch']
#######################################################################################
# training Settings
Spatial_Trainning_Settings = cfg['Spatial-CrossValidation']['Training_Settings']

kfold = Spatial_Trainning_Settings['kfold']
repeats = Spatial_Trainning_Settings['repeats']
beginyears = Spatial_Trainning_Settings['beginyears']
endyears = Spatial_Trainning_Settings['endyears']

#######################################################################################
# Forced Slope Unity Settings
ForcedSlopeUnityTable = cfg['Spatial-CrossValidation']['Forced-Slope-Unity']

ForcedSlopeUnity = ForcedSlopeUnityTable['ForcedSlopeUnity']
EachMonthForcedSlopeUnity = ForcedSlopeUnityTable['EachMonthForcedSlopeUnity']

#######################################################################################
# Training file Path
results_dir = cfg['Pathway']['Results-dir'] 

txt_outdir = results_dir['txt_outdir']



def Get_typeName(bias, normalize_bias, normalize_species, absolute_species, log_species, species):
    if bias == True:
        typeName = '{}-bias'.format(species)
    elif normalize_bias:
        typeName = 'Normalized-{}-bias'.format(species)
    elif normalize_species == True:
        typeName = 'Normaized-{}'.format(species)
    elif absolute_species == True:
        typeName = 'Absolute-{}'.format(species)
    elif log_species == True:
        typeName = 'Log-{}'.format(species)
    return  typeName

def initialize_multimodels_CV_Dic(kfold:int, repeats:int, beginyears:list,):

    CV_R2 = {}
    CV_slope = {}
    CV_RMSE = {}

    annual_CV_R2 = {}
    annual_CV_slope = {}
    annual_CV_RMSE = {}

    month_CV_R2 = {}
    month_CV_slope = {}
    month_CV_RMSE = {}

    training_CV_R2 = {}
    training_annual_CV_R2 = {}
    training_month_CV_R2 = {}

    geophysical_CV_R2 = {}
    geophysical_annual_CV_R2 = {}
    geophysical_month_CV_R2 = {}


    CV_R2['Alltime']    = np.zeros((kfold * repeats + 1), dtype=np.float32)
    CV_slope['Alltime'] = np.zeros((kfold * repeats + 1), dtype=np.float32)
    CV_RMSE['Alltime']  = np.zeros((kfold * repeats + 1), dtype=np.float32)

    annual_CV_R2['Alltime']    = np.zeros((kfold * repeats + 1), dtype=np.float32)
    annual_CV_slope['Alltime'] = np.zeros((kfold * repeats + 1), dtype=np.float32)
    annual_CV_RMSE['Alltime']  = np.zeros((kfold * repeats + 1), dtype=np.float32)

    month_CV_R2['Alltime']    = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
    month_CV_slope['Alltime'] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
    month_CV_RMSE['Alltime']  = np.zeros((12, kfold * repeats + 1), dtype = np.float32)

    training_CV_R2['Alltime']        = np.zeros((kfold * repeats + 1), dtype=np.float32)
    training_annual_CV_R2['Alltime'] = np.zeros((kfold * repeats + 1), dtype=np.float32)
    training_month_CV_R2['Alltime']  = np.zeros((12, kfold * repeats + 1), dtype = np.float32)

    geophysical_CV_R2['Alltime']        = np.zeros((kfold * repeats + 1), dtype=np.float32)
    geophysical_annual_CV_R2['Alltime'] = np.zeros((kfold * repeats + 1), dtype=np.float32)
    geophysical_month_CV_R2['Alltime']  = np.zeros((12, kfold * repeats + 1), dtype = np.float32)


    for imodel in range(len(beginyears)):

        CV_R2[str(beginyears[imodel])]    = np.zeros((kfold * repeats + 1), dtype=np.float32)
        CV_slope[str(beginyears[imodel])] = np.zeros((kfold * repeats + 1), dtype=np.float32)
        CV_RMSE[str(beginyears[imodel])]  = np.zeros((kfold * repeats + 1), dtype=np.float32)

        annual_CV_R2[str(beginyears[imodel])]    = np.zeros((kfold * repeats + 1), dtype=np.float32)
        annual_CV_slope[str(beginyears[imodel])] = np.zeros((kfold * repeats + 1), dtype=np.float32)
        annual_CV_RMSE[str(beginyears[imodel])]  = np.zeros((kfold * repeats + 1), dtype=np.float32)

        month_CV_R2[str(beginyears[imodel])]    = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
        month_CV_slope[str(beginyears[imodel])] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
        month_CV_RMSE[str(beginyears[imodel])]  = np.zeros((12, kfold * repeats + 1), dtype = np.float32)

        training_CV_R2[str(beginyears[imodel])]          = np.zeros((kfold * repeats + 1), dtype=np.float32)
        training_annual_CV_R2[str(beginyears[imodel])]   = np.zeros((kfold * repeats + 1), dtype=np.float32)
        training_month_CV_R2[str(beginyears[imodel])]    = np.zeros((12, kfold * repeats + 1), dtype = np.float32)

        geophysical_CV_R2[str(beginyears[imodel])]          = np.zeros((kfold * repeats + 1), dtype=np.float32)
        geophysical_annual_CV_R2[str(beginyears[imodel])]   = np.zeros((kfold * repeats + 1), dtype=np.float32)
        geophysical_month_CV_R2[str(beginyears[imodel])]    = np.zeros((12, kfold * repeats + 1), dtype = np.float32)


    return training_CV_R2, training_month_CV_R2, training_annual_CV_R2, geophysical_CV_R2,geophysical_annual_CV_R2,geophysical_month_CV_R2,CV_R2, CV_slope, CV_RMSE, annual_CV_R2, annual_CV_slope, annual_CV_RMSE, month_CV_R2, month_CV_slope, month_CV_RMSE

def initialize_AnnualDataRecording_Dic(beginyears):
    annual_final_test = {}
    annual_obs_test   = {}
    for imodel in range(len(beginyears)):
        annual_final_test[str(beginyears[imodel])] = np.array([],dtype=np.float64)
        annual_obs_test[str(beginyears[imodel])] = np.array([],dtype=np.float64)
        annual_final_test['Alltime'] = np.array([],dtype=np.float64)
        annual_obs_test['Alltime'] = np.array([],dtype=np.float64)
    return annual_final_test, annual_obs_test

def initialize_MonthlyDataRecording_Dic(beginyears):
    monthly_final_test = {}
    monthly_obs_test   = {}
    MONTH = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    
    for imodel in range(len(beginyears)):
        monthly_final_test[str(beginyears[imodel])] = {}
        monthly_obs_test[str(beginyears[imodel])] = {}
    monthly_final_test['Alltime'] = {}
    monthly_obs_test['Alltime'] = {}
    for imonth in range(len(MONTH)):
        for imodel in range(len(beginyears)):
            monthly_final_test[str(beginyears[imodel])][MONTH[imonth]] = np.array([],dtype=np.float64)
            monthly_obs_test[str(beginyears[imodel])][MONTH[imonth]] = np.array([],dtype=np.float64)
        monthly_final_test['Alltime'][MONTH[imonth]] = np.array([],dtype=np.float64)
        monthly_obs_test['Alltime'][MONTH[imonth]] = np.array([],dtype=np.float64)
    return monthly_final_test, monthly_obs_test


############# BLOO CV toolkits ####################


def GetBufferTrainingIndex(test_index:np.array,train_index:np.array,buffer:float):
    """_summary_

    Args:
        test_index (np.array): _description_
        train_index (np.array): _description_
        buffer (float): _description_
    """
    time_start = time.time()
    sitelat, sitelon = load_obs_loc()
    for isite in range(len(test_index)):
        train_index = find_sites_nearby(test_lat=sitelat[test_index[isite]],test_lon=sitelon[test_index[isite]],train_index=train_index,
                                        train_lat=sitelat,train_lon=sitelon,buffer_radius=buffer)
    time_end = time.time()
    print('Number of train index: ',len(train_index),'\nNumber of test index: ', len(test_index),'\nTime consume: ',str(np.round(time_end-time_start,4)),'s')
    return train_index