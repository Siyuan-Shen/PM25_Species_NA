import torch
import numpy as np
import torch
import torch.nn as nn
import os
import gc
from sklearn.model_selection import RepeatedKFold
import random
import csv

from Training_pkg.iostream import load_TrainingVariables, load_geophysical_biases_data, load_geophysical_species_data, load_monthly_obs_data, Learning_Object_Datasets
from Training_pkg.utils import *
from Training_pkg.Model_Func import train, predict
from Training_pkg.data_func import normalize_Func, get_trainingdata_within_sart_end_YEAR
from Training_pkg.Statistic_Func import regress2, linear_regression, Cal_RMSE
from Training_pkg.Net_Construction import *

from Evaluation_pkg.utils import *
from Evaluation_pkg.data_func import GetXIndex,GetYIndex,Get_XY_indices, Get_XY_arraies, Get_final_output, ForcedSlopeUnity_Func, CalculateAnnualR2, CalculateMonthR2, calculate_Statistics_results
from Evaluation_pkg.iostream import load_coMonitor_Population,save_trained_model, output_text, save_loss_accuracy, save_data_recording, load_data_recording, AVD_output_text

from visualization_pkg.Assemble_Func import plot_save_loss_accuracy_figure


def Get_SecondStage_GeoSpecies(first_stage_final_data,first_stage_training_data,width):
    second_stage_training_matrix = np.full((len(first_stage_final_data),len(TwoStage_Variables),width,width),1.0)
    for itag in range(len(TwoStage_Variables)):
        for iw in range(width):
            for ih in range(width):
                second_stage_training_matrix[:,itag,iw,ih] = first_stage_final_data
    for itag in range(len(TwoStage_Variables)):
        ratio = first_stage_training_data[:,channel_names.index(TwoStage_Variables[itag]),:,:]/first_stage_training_data[:,channel_names.index('GC_PM25'),:,:]
        second_stage_training_matrix[:,itag,:,:] = second_stage_training_matrix[:,itag,:,:]*ratio
    return second_stage_training_matrix

