import numpy as np
from Training_pkg.Statistic_Func import linear_regression,regress2, Cal_RMSE

    

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
