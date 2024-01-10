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
test_beginyear = Spatial_Trainning_Settings['test_beginyear']
test_endyear = Spatial_Trainning_Settings['test_endyear']
#######################################################################################
# Forced Slope Unity Settings
ForcedSlopeUnityTable = cfg['Spatial-CrossValidation']['Forced-Slope-Unity']

ForcedSlopeUnity = ForcedSlopeUnityTable['ForcedSlopeUnity']
EachMonthForcedSlopeUnity = ForcedSlopeUnityTable['EachMonthForcedSlopeUnity']

#######################################################################################
# Training file Path
results_dir = cfg['Pathway']['Results-dir'] 

txt_outdir = results_dir['txt_outdir']



################################## BLOO Cross-Validation ################################

BLOO_CrossValidation_Switch = cfg['BLOO-CrossValidation']['BLOO_CrossValidation_Switch']
Buffer_size = cfg['BLOO-CrossValidation']['Buffer_size']

#######################################################################################
# BLOO Training Settings

BLOO_TrainingSettings = cfg['BLOO-CrossValidation']['TrainingSettings']
BLOO_kfold   = BLOO_TrainingSettings['kfold']
BLOO_repeats = BLOO_TrainingSettings['repeats']
BLOO_beginyears = BLOO_TrainingSettings['beginyears']
BLOO_endyears   = BLOO_TrainingSettings['endyears']
BLOO_test_beginyear = BLOO_TrainingSettings['test_beginyear']
BLOO_test_endyear   = BLOO_TrainingSettings['test_endyear']



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


def initialize_AVD_DataRecording(beginyear:int,endyear:int):
    """This is used to return data recording dict. dict = { area: {Year : {Month : np.array() }}}

    Args:
        Areas (list): _description_
        Area_beginyears (dict): _description_
        endyear (int): _description_

    Returns:
        _type_: _description_
    """
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']
    
    final_data_recording = {}
    obs_data_recording = {}
    geo_data_recording = {}
    testing_population_data_recording  = {}
    training_final_data_recording = {}
    training_obs_data_recording = {}
    training_dataForSlope_recording = {}

    
    for iyear in range(endyear-beginyear+1): 
            print(str(beginyear+iyear))   
            final_data_recording[str(beginyear+iyear)] = {}
            obs_data_recording[str(beginyear+iyear)] = {}
            geo_data_recording[str(beginyear+iyear)] = {}
            testing_population_data_recording[str(beginyear+iyear)] = {}
            training_final_data_recording[str(beginyear+iyear)] = {}
            training_obs_data_recording[str(beginyear+iyear)] = {}
            training_dataForSlope_recording[str(beginyear+iyear)] = {}

            for imonth in MONTH:
                final_data_recording[str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
                obs_data_recording[str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
                geo_data_recording[str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
                testing_population_data_recording[str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
                training_final_data_recording[str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
                training_obs_data_recording[str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
                training_dataForSlope_recording[str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
             
    return final_data_recording, obs_data_recording, geo_data_recording, testing_population_data_recording, training_final_data_recording, training_obs_data_recording, training_dataForSlope_recording

def initialize_AVD_CV_dict(test_beginyear:int,test_endyear:int):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']
    test_CV_R2   = {}
    train_CV_R2  = {}
    geo_CV_R2    = {}
    RMSE_CV_R2   = {}
    NRMSE_CV_R2  = {}
    slope_CV_R2  = {}
    #PWAModel     = {}
    #PWAMonitors  = {}
    
    for iyear in range(test_endyear-test_beginyear+1):
            test_CV_R2[str(test_beginyear+iyear)]  = {}
            train_CV_R2[str(test_beginyear+iyear)]  = {}
            geo_CV_R2[str(test_beginyear+iyear)]  = {}
            RMSE_CV_R2[str(test_beginyear+iyear)]  = {}
            NRMSE_CV_R2[str(test_beginyear+iyear)]  = {}
            slope_CV_R2[str(test_beginyear+iyear)]  = {}
            #PWAModel[str(test_beginyear+iyear)]  = {}
            #PWAMonitors[str(test_beginyear+iyear)]  = {}
            
            for imonth in MONTH:
                test_CV_R2[str(test_beginyear+iyear)][imonth]  = -1.0
                train_CV_R2[str(test_beginyear+iyear)][imonth]  = -1.0
                geo_CV_R2[str(test_beginyear+iyear)][imonth]  = -1.0
                RMSE_CV_R2[str(test_beginyear+iyear)][imonth]  = -1.0
                NRMSE_CV_R2[str(test_beginyear+iyear)][imonth]  = -1.0
                slope_CV_R2[str(test_beginyear+iyear)][imonth]  = -1.0
                #PWAModel[str(test_beginyear+iyear)][imonth]  = -1.0
                #PWAMonitors[str(test_beginyear+iyear)][imonth]  = -1.0

    return test_CV_R2, train_CV_R2, geo_CV_R2, RMSE_CV_R2, NRMSE_CV_R2, slope_CV_R2, slope_CV_R2# PWAModel, PWAMonitors


def initialize_AVD_CV_Alltime_dict():
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']
    test_CV_R2_Alltime   = {'Alltime':{}}
    train_CV_R2_Alltime  = {'Alltime':{}}
    geo_CV_R2_Alltime    = {'Alltime':{}}
    RMSE_CV_R2_Alltime   = {'Alltime':{}}
    NRMSE_CV_R2_Alltime   = {'Alltime':{}}
    slope_CV_R2_Alltime  = {'Alltime':{}}
    #PWAModel_Alltime     = {}
    #PWAMonitors_Alltime  = {}
    
    for imonth in MONTH:
            ## np.zeros((3),dtype=np.float64) - 0 - mean, 1 - min, 2 - max
            test_CV_R2_Alltime['Alltime'][imonth]  = np.zeros((3),dtype=np.float64)
            train_CV_R2_Alltime['Alltime'][imonth] = np.zeros((3),dtype=np.float64)
            geo_CV_R2_Alltime['Alltime'][imonth]   = np.zeros((3),dtype=np.float64)
            RMSE_CV_R2_Alltime['Alltime'][imonth]  = np.zeros((3),dtype=np.float64)
            NRMSE_CV_R2_Alltime['Alltime'][imonth]  = np.zeros((3),dtype=np.float64)
            slope_CV_R2_Alltime['Alltime'][imonth] = np.zeros((3),dtype=np.float64)
            #PWAModel_Alltime['Alltime'][imonth]    = np.zeros((3),dtype=np.float64)
            #PWAMonitors_Alltime['Alltime'][imonth] = np.zeros((3),dtype=np.float64)
    return test_CV_R2_Alltime, train_CV_R2_Alltime, geo_CV_R2_Alltime, RMSE_CV_R2_Alltime, NRMSE_CV_R2_Alltime, slope_CV_R2_Alltime#, PWAModel_Alltime, PWAMonitors_Alltime

def get_annual_longterm_array(beginyear, endyear, final_data_recording,obs_data_recording):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    final_longterm_data = np.zeros(final_data_recording[str(beginyear)]['Jan'].shape, dtype=np.float64)
    obs_longterm_data   = np.zeros(final_data_recording[str(beginyear)]['Jan'].shape, dtype=np.float64)
    count = 0
    for iyear in range(endyear-beginyear+1):
        for imonth in range(len(MONTH)):
            final_longterm_data += final_data_recording[str(beginyear+iyear)][MONTH[imonth]]
            obs_longterm_data   += obs_data_recording[str(beginyear+iyear)][MONTH[imonth]]
            count += 1
    final_longterm_data = final_longterm_data/count
    obs_longterm_data   = obs_longterm_data/count
    return final_longterm_data, obs_longterm_data

def get_monthly_longterm_array(beginyear, imonth, endyear, final_data_recording,obs_data_recording):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    final_longterm_data = np.zeros(final_data_recording[str(beginyear)]['Jan'].shape, dtype=np.float64)
    obs_longterm_data   = np.zeros(final_data_recording[str(beginyear)]['Jan'].shape, dtype=np.float64)
    count = 0
    for iyear in range(endyear-beginyear+1):
    
        final_longterm_data += final_data_recording[str(beginyear+iyear)][MONTH[imonth]]
        obs_longterm_data   += obs_data_recording[str(beginyear+iyear)][MONTH[imonth]]
        count += 1
    final_longterm_data = final_longterm_data/count
    obs_longterm_data   = obs_longterm_data/count
    return final_longterm_data, obs_longterm_data

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


def GetBufferTrainingIndex(test_index:np.array,train_index:np.array,buffer:float,sitelat:np.array, sitelon:np.array):
    """_summary_

    Args:
        test_index (np.array): _description_
        train_index (np.array): _description_
        buffer (float): _description_
    """
    time_start = time.time()
    for isite in range(len(test_index)):
        train_index = find_sites_nearby(test_lat=sitelat[test_index[isite]],test_lon=sitelon[test_index[isite]],train_index=train_index,
                                        train_lat=sitelat,train_lon=sitelon,buffer_radius=buffer)
    time_end = time.time()
    print('Number of train index: ',len(train_index),'\nNumber of test index: ', len(test_index),'\nTime consume: ',str(np.round(time_end-time_start,4)),'s')
    return train_index


def find_sites_nearby(test_lat: np.float32, test_lon: np.float32,train_index:np.array,
                      train_lat: np.array, train_lon: np.array, buffer_radius: np.float32):
    """This function is used to get the sites index within the buffe area and exclue them from the training index. 

    Args:
        test_lat (np.float32): Test site latitude.
        test_lon (np.float32): Test site longitude.
        train_index (np.array): Training index(remain). This function should be in a loop,
        and all input training index already exclude other sites within the buffer zone near other testing site.
        train_lat (np.array): The initial sites lat array.
        train_lon (np.array): The initial sites lon array.
        buffer_radius (np.float32): The buffer radius.

    Returns:
        np.array : The train index exclude the sites within the input test sites surronding buffer zone.
    """
    lat_min = max(-69.95, (test_lat - 0.009 * buffer_radius))
    lat_max = min(59.95, (test_lat + 0.009 * buffer_radius))
    lon_min = max(-179.95, (test_lon - 0.026 * buffer_radius))
    lon_max = min(179.95, (test_lon + 0.026 * buffer_radius))
    # Find the sites within the square first
    lat_index = np.intersect1d(np.where(train_lat>lat_min),np.where(train_lat<lat_max))
    lon_index = np.intersect1d(np.where(train_lon>lon_min),np.where(train_lon<lon_max))
    sites_nearby_index = np.intersect1d(lat_index,lon_index)
           
    sites_lat_nearby = train_lat[sites_nearby_index]
    sites_lon_nearby = train_lon[sites_nearby_index]

    # Find the sites within the buffer zones
    sites_within_radius_index = np.array([],dtype=int)
    for isite in range(len(sites_nearby_index)):
        distance = calculate_distance(test_lat,test_lon,train_lat[sites_nearby_index[isite]],train_lon[sites_nearby_index[isite]])
        if distance < buffer_radius:
            sites_within_radius_index = np.append(sites_within_radius_index,sites_nearby_index[isite])
    sites_within_index,X_index,Y_index = np.intersect1d(train_index,sites_within_radius_index,return_indices=True)
    train_index = np.delete(train_index,X_index)
    return train_index 

def calculate_distance(pixel_lat:np.float32,pixel_lon:np.float32,site_lat:np.float32,site_lon:np.float32,r=6371.01):
    site_pos1 = pixel_lat * np.pi / 180.0
    site_pos2 = pixel_lon * np.pi / 180.0
    other_sites_pos1_array = site_lat * np.pi / 180.0
    other_sites_pos2_array = site_lon * np.pi / 180.0
    dist = r * np.arccos(np.sin(site_pos1)*np.sin(other_sites_pos1_array)+np.cos(site_pos1)*np.cos(other_sites_pos1_array)*np.cos(site_pos2-other_sites_pos2_array))
    return dist