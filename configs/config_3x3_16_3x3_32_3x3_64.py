# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
root_dir = os.path.join(os.getcwd(), '.')
print(root_dir)

class Config:

    batch_size = 16
    time_steps = 10
    channel = 1
    # width = 32
    # height = 32
    epochs = 1
    lr_init = 0.001
    early_stop = False
    early_stop_patience = 15
    # MultiStep学习率衰减
    milestones = [15, 35]
    gamma = 0.9


    mae_thresh = None
    mape_thresh = 0.


config = Config()
