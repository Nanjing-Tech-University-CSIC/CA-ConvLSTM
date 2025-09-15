import torch
import numpy as np


def SCS_SST_reverse_minmaxscaler(norm_data):
    max = 32.86
    min = 9.76

    return norm_data * (max - min) + min

def SCS_SSH_reverse_minmaxscaler(norm_data):
    max = 0.7264
    min = -0.6781

    return norm_data * (max - min) + min
def SST_minmaxscaler(data):
    # highLat  max = 305.53998   min = 274.19998
    min = 11.1053
    max = 33.3644

    return (data - min)/(max-min)


def SST_reverse_minmaxscaler(norm_data):
    min = 11.1053
    max = 33.3644

    return norm_data * (max - min) + min
