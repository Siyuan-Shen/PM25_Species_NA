import torch
import torch.nn as nn
from Training_pkg.Loss_Func import *
from Training_pkg.Net_Construction import *
import toml

cfg = toml.load('./config.toml')

#######################################################################################
# Observation Path
obs_dir = cfg['Pathway']['observations-dir']

geophysical_species_data_dir = obs_dir['geophysical_species_data_dir']
geophysical_biases_data_dir  = obs_dir['geophysical_biases_data_dir']
ground_observation_data_dir  = obs_dir['ground_observation_data_dir']
#######################################################################################
# Training file Path
Training_dir = cfg['Pathway']['TrainingModule-dir']

training_infile = Training_dir['training_infile']
model_outdir = Training_dir['model_outdir']

#######################################################################################
Config_outdir = cfg['Pathway']['Config-outdir']['Config_outdir']
#######################################################################################
# identity settings
identity = cfg['Training-Settings']['identity']

special_name = identity['special_name']
version = identity['version']

#######################################################################################
# Hyperparameters settings
HyperParameters = cfg['Training-Settings']['hyper-parameters']

channel_names = HyperParameters['channel_names']
epoch = HyperParameters['epoch']
batchsize = HyperParameters['batchsize']
lr0 = HyperParameters['learning_rate']

#######################################################################################
# Learning Objectives Settings
learning_objective = cfg['Training-Settings']['learning-objective']

species = learning_objective['species']

bias = learning_objective['bias']
normalize_bias = learning_objective['normalize_bias']
normalize_species = learning_objective['normalize_species']
absolute_species = learning_objective['absolute_species']
log_species = learning_objective['log_species']

#######################################################################################
# Loss Function Settings
Loss_Func = cfg['Training-Settings']['Loss-Functions']

Loss_type = Loss_Func['Loss_type']




def loss_func_lookup_table():
    loss_dic = {
        'MSE' : nn.MSELoss,
    }
    return loss_dic

def resnet_block_lookup_table(blocktype):
    if blocktype == 'BasicBlock':
        return BasicBlock
    elif blocktype == 'Bottleneck':
        return Bottleneck
    else:
        print(' Wrong Key Word! BasicBlock or Bottleneck only! ')
        return None
    
def Net_Structure_lookup_table(nchannel,block,blocks_num,):
    Net_Structure_table = {
        'normal_cnn' : Net(nchannel=nchannel),
        'Resnet'     : ResNet(nchannel=nchannel,block=block,)

    }
    return