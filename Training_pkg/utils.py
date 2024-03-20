import torch
import torch.nn as nn

import toml

cfg = toml.load('./config.toml')

#######################################################################################
# Observation Path
obs_dir = cfg['Pathway']['observations-dir']

geophysical_species_data_dir    = obs_dir['geophysical_species_data_dir']
geophysical_biases_data_dir     = obs_dir['geophysical_biases_data_dir']
ground_observation_data_dir     = obs_dir['ground_observation_data_dir']
geophysical_species_data_infile = obs_dir['geophysical_species_data_infile']
geophysical_biases_data_infile  = obs_dir['geophysical_biases_data_infile']
ground_observation_data_infile  = obs_dir['ground_observation_data_infile']
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

#######################################################################################
# Net Structure Settings

net_structure_settings = cfg['Training-Settings']['net_structure_settings']

ResNet_setting      = net_structure_settings['ResNet']['Settings']
ResNet_Blocks       = net_structure_settings['ResNet']['Blocks']
ResNet_blocks_num   = net_structure_settings['ResNet']['blocks_num']

LateFusion_setting      = net_structure_settings['LateFusion']['Settings']
LateFusion_Blocks       = net_structure_settings['LateFusion']['Blocks']
LateFusion_blocks_num   = net_structure_settings['LateFusion']['blocks_num']
LateFusion_initial_channels     = net_structure_settings['LateFusion']['initial_channels']
LateFusion_latefusion_channels  = net_structure_settings['LateFusion']['LateFusion_channels']


#######################################################################################
# learning rate settings
lr_settings = cfg['Training-Settings']['learning_rate']
lr0 = lr_settings['learning_rate0']


### Strategy
ExponentialLR = lr_settings['ExponentialLR']['Settings']
ExponentialLR_gamma = lr_settings['ExponentialLR']['gamma']

CosineAnnealingLR = lr_settings['CosineAnnealingLR']['Settings']
CosineAnnealingLR_T_max = lr_settings['CosineAnnealingLR']['T_max']
CosineAnnealingLR_eta_min = lr_settings['CosineAnnealingLR']['eta_min']

CosineAnnealingRestartsLR = lr_settings['CosineAnnealingRestartsLR']['Settings']
CosineAnnealingRestartsLR_T_0 = lr_settings['CosineAnnealingRestartsLR']['T_0']
CosineAnnealingRestartsLR_T_mult = lr_settings['CosineAnnealingRestartsLR']['T_mult']
CosineAnnealingRestartsLR_eta_min = lr_settings['CosineAnnealingRestartsLR']['eta_min']

#######################################################################################
# activation func settings
activation_func_settings = cfg['Training_Settings']['activation_func']
activation_func_name = activation_func_settings['activation_func_name']
ReLU_ACF = activation_func_settings['ReLU']['Settings']
Tanh_ACF = activation_func_settings['Tanh']['Settings']
GeLU_ACF = activation_func_settings['GeLU']['Settings']
Sigmoid_ACF = activation_func_settings['Sigmoid']['Settings']

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

#######################################################################################
# CombineWithGeophysical Settings
CombineWithGeophysical = cfg['Training-Settings']['CombineWithGeophysical']

combine_with_GeophysicalSpeceis_Switch                = CombineWithGeophysical['combine_with_GeophysicalSpeceis_Switch']
cutoff_size                                           = CombineWithGeophysical['cutoff_size']



training_infile = training_infile.format(species,species)
geophysical_biases_data_infile  = geophysical_biases_data_infile.format(species)
geophysical_species_data_infile = geophysical_species_data_infile.format(species)
ground_observation_data_infile  = ground_observation_data_infile.format(species)

def activation_function_table():
    if ReLU_ACF == True:
        return nn.ReLU()
    elif Tanh_ACF == True:
        return nn.Tanh()
    elif GeLU_ACF == True:
        return nn.GELU()
    elif Sigmoid_ACF == True:
        return nn.Sigmoid()
    

def lr_strategy_lookup_table(optimizer):
    if ExponentialLR:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=ExponentialLR_gamma)
    elif CosineAnnealingLR:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=CosineAnnealingLR_T_max,eta_min=CosineAnnealingLR_eta_min)
    elif CosineAnnealingRestartsLR:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=CosineAnnealingRestartsLR_T_0,T_mult=CosineAnnealingRestartsLR_T_mult,eta_min=CosineAnnealingLR_eta_min)



def find_latfusion_index():
    initial_channel_index = []
    for i in range(len(LateFusion_initial_channels)):
        initial_channel_index.append(channel_names.index(LateFusion_initial_channels[i]))
    
    latefusion_channel_index = []
    for i in range(len(LateFusion_latefusion_channels)):
        latefusion_channel_index.append(channel_names.index(LateFusion_latefusion_channels[i]))
    
    return initial_channel_index, latefusion_channel_index
    
