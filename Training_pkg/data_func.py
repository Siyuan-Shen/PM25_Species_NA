import numpy as np

def normalize_Func(inputarray:np.array):
    normalized_inputarray = inputarray.copy()
    input_mean = np.mean(inputarray,axis=0)
    input_std  = np.std(inputarray,axis=0)
    input_std[np.where(input_std==0)] = 1.0
    normalized_inputarray -= input_mean
    normalized_inputarray /= input_std
    return normalized_inputarray,input_mean,input_std



def get_trainingdata_within_sart_end_YEAR(initial_array,training_start_YYYY, training_end_YYYY, start_YYYY, sitesnumber):
    final_array = initial_array[(training_start_YYYY-start_YYYY)*12*sitesnumber:(training_end_YYYY-start_YYYY+1)*12*sitesnumber,:,:,:]
    return final_array