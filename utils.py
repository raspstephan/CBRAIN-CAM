
import subprocess
import os
import math
import json
import logging
import numpy as np
from PIL import Image
from datetime import datetime

from folderDefs import *

def logdir():
    str_time = datetime.now().strftime('%Y%m%d-%H%M')#'20161020-1441'#
    return trainingLogDir + str_time

def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    if config.load_path:
        if config.load_path.startswith(config.log_dir):
            print(1)
            config.model_dir = config.load_path
        else:
            if config.load_path.startswith(config.dataset):
                print(2)
                config.model_name = config.load_path
            else:
                print(3)
                config.model_name = "{}_{}".format(config.dataset, config.load_path)
    else:
        print(4)
        config.model_name = "{}_{}_{}".format(config.dataset, get_time(), ','.join(config.hidden.split(',')))
        config.model_name = config.model_name + '_dropout_' + str(config.dropout_rate)
        config.model_name = config.model_name + '_activation_' + str(config.act)
    
#    config.log_dir   = config.log_dir + '/' + config.varname + '/' + config.hidden
#    config.data_dir  = config.data_dir + '/' + config.varname + '/' + config.hidden

    if not hasattr(config, 'model_dir'):
        config.model_dir = os.path.join(config.log_dir, config.model_name)
    config.data_path = os.path.join(config.data_dir, config.dataset)

    for path in [config.log_dir, config.data_dir, config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
    print('config.load_path', config.load_path)
    print('config.model_name', config.model_name)
    print('config.log_dir', config.log_dir)
    print('config.data_dir', config.data_dir)
    print('config.model_dir', config.model_dir)

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def load_config(config, subset):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[read] MODEL dir: %s" % config.model_dir)
    print("[read] PARAM path: %s" % param_path)

    with open(param_path, 'r') as fp:
        json_dict = json.load(fp)
    for k,v in json_dict.items():
        if subset is None or k in subset:
            setattr(config, k, v)
    return config

def rank(array):
    return len(array.shape)

