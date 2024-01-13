import torch
import numpy as np
import os
import csv

def save_trained_model_forEstimation(cnn_model, model_outdir, typeName, version, species, nchannel, special_name,beginyear, endyear, width, height):
    outdir = model_outdir + '{}/{}/Results/Estimation-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    model_outfile = outdir +  'Estimation_{}_{}_{}x{}_{}-{}_{}Channel{}.pt'.format(typeName, species, width,height, beginyear, endyear, nchannel,special_name)
    torch.save(cnn_model, model_outfile)