import numpy as np
from Training_pkg.Statistic_Func import linear_regression,regress2, Cal_RMSE, Cal_NRMSE
from Evaluation_pkg.utils import *
    

def GetXIndex(index,beginyear:int, endyear:int, sitenumber:int):
    X_index = np.zeros((12 * (endyear - beginyear + 1) * len(index)), dtype=int)
    for i in range(12 * (endyear - beginyear + 1)):
        X_index[i * len(index):(i + 1) * len(index)] = i * sitenumber + index
    return X_index

def GetYIndex(index,beginyear:int, endyear:int,sitenumber:int):
    # Y is for observations
    Y_index = np.zeros((12 * (endyear - beginyear + 1) * len(index)), dtype=int)
    for i in range(12 * (endyear - beginyear + 1)):
        Y_index[i * len(index):(i + 1) * len(index)] = ((beginyear - 1998)*12 + i) * sitenumber + index
    return Y_index   

def Get_XY_indices(train_index,test_index, beginyear, endyear, sitesnumber ):
    X_Training_index = GetXIndex(index=train_index,beginyear=beginyear,endyear=endyear,sitenumber=sitesnumber)
    X_Testing_index  = GetXIndex(index=test_index ,beginyear=beginyear,endyear=endyear,sitenumber=sitesnumber)
    Y_Training_index = GetYIndex(index=train_index,beginyear=beginyear,endyear=endyear,sitenumber=sitesnumber)
    Y_Testing_index  = GetYIndex(index=test_index ,beginyear=beginyear,endyear=endyear,sitenumber=sitesnumber)
    
    return X_Training_index, X_Testing_index, Y_Training_index, Y_Testing_index

def Get_XY_arraies(Normalized_TrainingData, true_input, X_Training_index, X_Testing_index, Y_Training_index, Y_Testing_index):
    print('length of Normalized_TrainingData: {}, length of true_input : {}, \nlength of X_Training_index: {}, length of Y_Training_index: {},\
          \n length of X_Testing_index: {}, length of Y_Testing_index: {}'.format(len(Normalized_TrainingData), len(true_input),\
                                                                                   len(X_Training_index), len(Y_Training_index),\
                                                                                    len(X_Testing_index), len(true_input)))
    X_train, y_train  = Normalized_TrainingData[X_Training_index, :, :, :], true_input[Y_Training_index]
    X_test,  y_test   = Normalized_TrainingData[X_Testing_index, :, :, :], true_input[Y_Testing_index]
    return X_train, X_test, y_train, y_test

def Get_final_output(Validation_Prediction, geophysical_species,bias,normalize_bias,normalize_species,absolute_species,log_species,mean,std,Y_Testing_index ):
    """This function is used to convert the model estimation to absolute PM species concentration and to compare with the 
    observed PM species.

    Args:
        Validation_Prediction (_type_): _description_
        geophysical_species (_type_): _description_
        SPECIES_OBS (_type_): _description_
        bias (_type_): _description_
        normalize_species (_type_): _description_
        absolute_species (_type_): _description_
        log_species (_type_): _description_
        Y_Testing_index (_type_): _description_

    Returns:
        _type_: _description_
    """
    if bias == True:
        final_data = Validation_Prediction + geophysical_species[Y_Testing_index]
    elif normalize_bias == True:
        final_data = Validation_Prediction * std + mean + geophysical_species[Y_Testing_index]
    elif normalize_species == True:
        final_data = Validation_Prediction * std + mean
    elif absolute_species == True:
        final_data = Validation_Prediction
    elif log_species == True:
        final_data = np.exp(Validation_Prediction) - 1
    return final_data



def ForcedSlopeUnity_Func(train_final_data,train_obs_data,test_final_data,train_area_index,test_area_index,endyear,beginyear,EachMonth:bool):
    if EachMonth:
        for i in range(12 * (endyear - beginyear + 1)):
            temp_train_final_data = train_final_data[i*len(train_area_index):(i+1)*len(train_area_index)]
            temp_train_obs_data   = train_obs_data[i*len(train_area_index):(i+1)*len(train_area_index)]
            temp_regression_dic = regress2(_x=temp_train_obs_data,_y=temp_train_final_data,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
            temp_offset,temp_slope = temp_regression_dic['intercept'], temp_regression_dic['slope']
            test_final_data[i*len(test_area_index):(i+1)*len(test_area_index)] = (test_final_data[i*len(test_area_index):(i+1)*len(test_area_index)] - temp_offset)/temp_slope
    else:
        month_train_obs_average = np.zeros((len(train_area_index)))
        month_train_average = np.zeros((len(train_area_index)))
        monthly_test_month = np.array(range(endyear - beginyear + 1)) * 12
        for imonth in range(12):
            for isite in range(len(train_area_index)):
                month_train_obs_average[isite] = np.mean(train_final_data[isite + (imonth + monthly_test_month) * len(train_area_index)])
                month_train_average[isite] = np.mean(train_final_data[isite + (imonth + monthly_test_month) * len(train_area_index)])
            temp_regression_dic = regress2(_x=month_train_obs_average,_y=month_train_average,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
            temp_offset,temp_slope = temp_regression_dic['intercept'], temp_regression_dic['slope']

            for iyear in range(endyear-beginyear+1):
                test_final_data[(iyear*12+imonth)*len(test_area_index):(iyear*12+imonth+1)*len(test_area_index)] -= temp_offset
                test_final_data[(iyear*12+imonth)*len(test_area_index):(iyear*12+imonth+1)*len(test_area_index)] /= temp_slope
    return test_final_data



def calculate_Statistics_results(test_beginyear,test_endyear:int,final_data_recording, obs_data_recording, geo_data_recording, training_final_data_recording, training_obs_data_recording):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    test_CV_R2, train_CV_R2, geo_CV_R2, RMSE_CV_R2, NRMSE_CV_R2, slope_CV_R2, slope_CV_R2 = initialize_AVD_CV_dict(test_beginyear=test_beginyear,test_endyear=test_endyear)
    
    for iyear in range(test_endyear-test_beginyear+1):
            for imonth in MONTH:
                print('Area: {}, Year: {}, Month: {}'.format('NA', test_beginyear+iyear, imonth))
                test_CV_R2[str(test_beginyear+iyear)][imonth] = linear_regression(final_data_recording[str(test_beginyear+iyear)][imonth], obs_data_recording[str(test_beginyear+iyear)][imonth])
                train_CV_R2[str(test_beginyear+iyear)][imonth] = linear_regression(training_final_data_recording[str(test_beginyear+iyear)][imonth], training_obs_data_recording[str(test_beginyear+iyear)][imonth])
                geo_CV_R2[str(test_beginyear+iyear)][imonth] = linear_regression(geo_data_recording[str(test_beginyear+iyear)][imonth], obs_data_recording[str(test_beginyear+iyear)][imonth])
                RMSE_CV_R2[str(test_beginyear+iyear)][imonth] = Cal_RMSE(final_data_recording[str(test_beginyear+iyear)][imonth], obs_data_recording[str(test_beginyear+iyear)][imonth])
                NRMSE_CV_R2[str(test_beginyear+iyear)][imonth] = Cal_NRMSE(final_data_recording[str(test_beginyear+iyear)][imonth], obs_data_recording[str(test_beginyear+iyear)][imonth])
                regression_Dic = regress2(_x= obs_data_recording[str(test_beginyear+iyear)][imonth],_y=final_data_recording[str(test_beginyear+iyear)][imonth],_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
                intercept,slope = regression_Dic['intercept'], regression_Dic['slope']
                slope_CV_R2[str(test_beginyear+iyear)][imonth]= slope
                
                if imonth == 'Jan':
                    final_data_recording[str(test_beginyear+iyear)]['Annual'] = final_data_recording[str(test_beginyear+iyear)][imonth]
                    obs_data_recording[str(test_beginyear+iyear)]['Annual'] = obs_data_recording[str(test_beginyear+iyear)][imonth]
                    geo_data_recording[str(test_beginyear+iyear)]['Annual'] = geo_data_recording[str(test_beginyear+iyear)][imonth]
                    training_final_data_recording[str(test_beginyear+iyear)]['Annual'] = training_final_data_recording[str(test_beginyear+iyear)][imonth]
                    training_obs_data_recording[str(test_beginyear+iyear)]['Annual'] = training_obs_data_recording[str(test_beginyear+iyear)][imonth]
                else:
                    final_data_recording[str(test_beginyear+iyear)]['Annual'] += final_data_recording[str(test_beginyear+iyear)][imonth]
                    obs_data_recording[str(test_beginyear+iyear)]['Annual'] += obs_data_recording[str(test_beginyear+iyear)][imonth]
                    geo_data_recording[str(test_beginyear+iyear)]['Annual'] += geo_data_recording[str(test_beginyear+iyear)][imonth]
                    training_final_data_recording[str(test_beginyear+iyear)]['Annual'] += training_final_data_recording[str(test_beginyear+iyear)][imonth]
                    training_obs_data_recording[str(test_beginyear+iyear)]['Annual'] += training_obs_data_recording[str(test_beginyear+iyear)][imonth]
                    
            final_data_recording[str(test_beginyear+iyear)]['Annual'] = final_data_recording[str(test_beginyear+iyear)]['Annual']/12.0
            obs_data_recording[str(test_beginyear+iyear)]['Annual'] = obs_data_recording[str(test_beginyear+iyear)]['Annual']/12.0
            geo_data_recording[str(test_beginyear+iyear)]['Annual'] = geo_data_recording[str(test_beginyear+iyear)]['Annual']/12.0
            training_final_data_recording[str(test_beginyear+iyear)]['Annual'] = training_final_data_recording[str(test_beginyear+iyear)]['Annual']/12.0
            training_obs_data_recording[str(test_beginyear+iyear)]['Annual'] = training_obs_data_recording[str(test_beginyear+iyear)]['Annual']/12.0
            
            print('Area: {}, Year: {}, Month: {}'.format('NA', test_beginyear+iyear, 'Annual'))
            test_CV_R2[str(test_beginyear+iyear)]['Annual'] = linear_regression(final_data_recording[str(test_beginyear+iyear)]['Annual'], obs_data_recording[str(test_beginyear+iyear)]['Annual'])
            train_CV_R2[str(test_beginyear+iyear)]['Annual'] = linear_regression(training_final_data_recording[str(test_beginyear+iyear)]['Annual'], training_obs_data_recording[str(test_beginyear+iyear)]['Annual'])
            geo_CV_R2[str(test_beginyear+iyear)]['Annual'] = linear_regression(geo_data_recording[str(test_beginyear+iyear)]['Annual'], obs_data_recording[str(test_beginyear+iyear)]['Annual'])
            RMSE_CV_R2[str(test_beginyear+iyear)]['Annual'] = Cal_RMSE(final_data_recording[str(test_beginyear+iyear)]['Annual'], obs_data_recording[str(test_beginyear+iyear)]['Annual'])
            NRMSE_CV_R2[str(test_beginyear+iyear)]['Annual'] = Cal_NRMSE(final_data_recording[str(test_beginyear+iyear)]['Annual'], obs_data_recording[str(test_beginyear+iyear)]['Annual'])
            
            regression_Dic = regress2(_x= obs_data_recording[str(test_beginyear+iyear)]['Annual'],_y=final_data_recording[str(test_beginyear+iyear)]['Annual'],_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
            intercept,slope = regression_Dic['intercept'], regression_Dic['slope']
            slope_CV_R2[str(test_beginyear+iyear)]['Annual'] = slope
            

    return test_CV_R2, train_CV_R2, geo_CV_R2, RMSE_CV_R2, NRMSE_CV_R2, slope_CV_R2


def calculate_Alltime_Statistics_results(test_beginyear:dict,test_endyear:int,test_CV_R2, train_CV_R2, geo_CV_R2, RMSE_CV_R2, NRMSE_CV_R2,slope_CV_R2):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec','Annual']
    test_CV_R2_Alltime, train_CV_R2_Alltime, geo_CV_R2_Alltime, RMSE_CV_R2_Alltime, NRMSE_CV_R2_Alltime, slope_CV_R2_Alltime = initialize_AVD_CV_Alltime_dict()
    
    for imonth in MONTH:
            temp_test_CV_R2_Alltime   = np.array([],dtype=np.float64)
            temp_train_CV_R2_Alltime  = np.array([],dtype=np.float64)
            temp_geo_CV_R2_Alltime    = np.array([],dtype=np.float64)
            temp_RMSE_CV_R2_Alltime   = np.array([],dtype=np.float64)
            temp_NRMSE_CV_R2_Alltime  = np.array([],dtype=np.float64)
            temp_slope_CV_R2_Alltime  = np.array([],dtype=np.float64)
            #temp_PWAModel_Alltime     = np.array([],dtype=np.float64)
            #temp_PWAMonitors_Alltime  = np.array([],dtype=np.float64)
            for iyear in range(test_endyear-test_beginyear+1):
                print('Area: {}, Year: {}, Month: {}'.format('NA', test_beginyear+iyear, imonth))
                temp_test_CV_R2_Alltime  = np.append(temp_test_CV_R2_Alltime, test_CV_R2[str(test_beginyear+iyear)][imonth])
                temp_train_CV_R2_Alltime = np.append(temp_train_CV_R2_Alltime, train_CV_R2[str(test_beginyear+iyear)][imonth])
                temp_geo_CV_R2_Alltime   = np.append(temp_geo_CV_R2_Alltime, geo_CV_R2[str(test_beginyear+iyear)][imonth])
                temp_RMSE_CV_R2_Alltime  = np.append(temp_RMSE_CV_R2_Alltime, RMSE_CV_R2[str(test_beginyear+iyear)][imonth])
                temp_NRMSE_CV_R2_Alltime = np.append(temp_NRMSE_CV_R2_Alltime, NRMSE_CV_R2[str(test_beginyear+iyear)][imonth])
                temp_slope_CV_R2_Alltime = np.append(temp_slope_CV_R2_Alltime, slope_CV_R2[str(test_beginyear+iyear)][imonth])
                #temp_PWAModel_Alltime    = np.append(temp_PWAModel_Alltime, PWAModel[str(test_beginyear+iyear)][imonth])
                #temp_PWAMonitors_Alltime = np.append(temp_PWAMonitors_Alltime, PWAMonitors[str(test_beginyear+iyear)][imonth])
            
            test_CV_R2_Alltime['Alltime'][imonth]     = get_mean_min_max_statistic(temp_test_CV_R2_Alltime)
            train_CV_R2_Alltime['Alltime'][imonth]    = get_mean_min_max_statistic(temp_train_CV_R2_Alltime)
            geo_CV_R2_Alltime['Alltime'][imonth]      = get_mean_min_max_statistic(temp_geo_CV_R2_Alltime)
            RMSE_CV_R2_Alltime['Alltime'][imonth]     = get_mean_min_max_statistic(temp_RMSE_CV_R2_Alltime)
            NRMSE_CV_R2_Alltime['Alltime'][imonth]    = get_mean_min_max_statistic(temp_NRMSE_CV_R2_Alltime)
            slope_CV_R2_Alltime['Alltime'][imonth]    = get_mean_min_max_statistic(temp_slope_CV_R2_Alltime)
            #PWAModel_Alltime['Alltime'][imonth]       = get_mean_min_max_statistic(temp_PWAModel_Alltime)
            #PWAMonitors_Alltime['Alltime'][imonth]    = get_mean_min_max_statistic(temp_PWAMonitors_Alltime)

    return test_CV_R2_Alltime, train_CV_R2_Alltime, geo_CV_R2_Alltime,RMSE_CV_R2_Alltime, NRMSE_CV_R2_Alltime, slope_CV_R2_Alltime

def get_mean_min_max_statistic(temp_CV):
    temp_array = np.zeros((3),dtype=np.float64)
    temp_array[0] = np.mean(temp_CV)
    temp_array[1] = np.min(temp_CV)
    temp_array[2] = np.max(temp_CV)
    return temp_array

def CalculateAnnualR2(test_index,final_data,test_obs_data,beginyear,endyear):
    '''
    This funciton is used to calculate the Annual R2, slope and RMSE
    return:
    annual_R2,annual_final_data,annual_mean_obs,slope, RMSE
    '''
    annual_mean_obs = np.zeros((len(test_index)))
    annual_final_data = np.zeros((len(test_index)))
    test_month = np.array(range((endyear - beginyear + 1) * 12))
    for isite in range(len(test_index)):
        annual_mean_obs[isite] = np.mean(test_obs_data[isite + test_month * len(test_index)])
        annual_final_data[isite] = np.mean(final_data[isite + test_month * len(test_index)])
    print(' ################### Annual R2: #######################')
    annual_R2 = linear_regression(annual_mean_obs, annual_final_data)
    regression_Dic = regress2(_x=annual_mean_obs,_y=annual_final_data,_method_type_1='ordinary least square',_method_type_2='reduced major axis',
    )
    intercept,slope = regression_Dic['intercept'], regression_Dic['slope']
    #b0, b1 = linear_slope(plot_obs_pm25,
    #                      plot_pre_pm25)
    intercept = round(intercept, 2)
    slope = round(slope, 2)
    RMSE = Cal_RMSE(annual_mean_obs, annual_final_data)
    return annual_R2,annual_final_data,annual_mean_obs,slope,RMSE

def CalculateMonthR2(test_index,final_data,test_obs_data,beginyear:int,endyear:int,monthly_final_test_imodel,monthly_obs_test_imodel):
    '''
    This funciton is used to calculate the monthly R2, slope and RMSE
    return:
    month_R2, month_slope, month_RMSE
    '''
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12']
    month_obs = np.zeros((len(test_index)))
    month_predict = np.zeros((len(test_index)))

    monthly_test_month = np.array(range(endyear - beginyear + 1)) * 12
    month_R2 = np.zeros(12,dtype = np.float64)
    month_slope = np.zeros(12,dtype = np.float64)
    month_RMSE = np.zeros(12,dtype = np.float64)
   

    for imonth in range(12):
        for isite in range(len(test_index)):
            month_obs[isite] = np.mean(test_obs_data[isite + (imonth + monthly_test_month) * len(test_index)])
            month_predict[isite] = np.mean(final_data[isite + (imonth + monthly_test_month) * len(test_index)])
        monthly_final_test_imodel[MONTH[imonth]] = np.append(monthly_final_test_imodel[MONTH[imonth]], month_predict)
        monthly_obs_test_imodel[MONTH[imonth]]   = np.append(monthly_obs_test_imodel[MONTH[imonth]], month_obs)
        print('-------------------- Month: {} --------------------------'.format(MONTH[imonth]))
        month_R2[imonth] = linear_regression(month_obs, month_predict)
        regression_Dic = regress2(_x=month_obs,_y=month_predict,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
        intercept,slope = regression_Dic['intercept'], regression_Dic['slope']
        month_slope[imonth] = round(slope, 2)
        month_RMSE[imonth] = Cal_RMSE(month_obs, month_predict)
        
    return month_R2, month_slope, month_RMSE, monthly_final_test_imodel, monthly_obs_test_imodel

