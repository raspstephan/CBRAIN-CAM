import numpy as np
import tensorflow as tf
print('tensorflow', tf.__version__)

from trainer import Trainer
from config import get_config
from utils import prepare_dirs_and_logger, save_config, load_config
from colorama import Fore, Back, Style

from  dataLoad import *

from folderDefs import *
import subprocess, threading, time

validationProcess = "python main.py --is_train=false --epoch=1 --use_gpu=false --load_path={}"
devnull = open(os.devnull, 'wb')

def main(config):
    'Preparing dirs and files'

    prepare_dirs_and_logger(config)

    rng = np.random.RandomState(config.random_seed)
    tf.set_random_seed(config.random_seed)

    if config.is_train:
        data_path = config.data_path
        batch_size = config.batch_size
        do_shuffle = True
    else:
        if config.load_path:
            # automatically reloads the correct arguments for the network
            config = load_config(config, ['dataset', 'hidden', 'keep_dropout_rate', 'act', 'addon', 'normalize', 'convo', 'input_names'])
            print(Fore.RED, 'config\n', config)
            print(Style.RESET_ALL)
        setattr(config, 'batch_size', 1024)
        data_path = config.data_path
        batch_size = config.batch_size
        do_shuffle = False
    save_config(config)
    with DataLoader(trainingDataDir, config) as data_loader:
        with tf.device("/cpu:0"):
            data_loader.prepareQueue()
        #data_loader = get_loader(data_path, config.batch_size, config.input_scale_size, config.data_format, config.split)
        trainer = Trainer(config, data_loader)

        if config.is_train:
            save_config(config)
            print('batches=', data_loader.NumBatchTrain)
            threadValid = None
            isTraining = True
            if config.run_validation:
                def loopvalidation():
                    time.sleep(5)
                    while isTraining:
                        for i in range(trainer.saveEverySec):
                            if isTraining:
                               time.sleep(1)
                        validationProcesslocal = validationProcess# + ' --dataset=' + config.dataset
                        processArg = validationProcesslocal.format(config.model_name).split()
                        print(Fore.RED, processArg)
                        print(Style.RESET_ALL)
                        subprocess.run(processArg, stdout=devnull)#, stderr=devnull)
                threadValid = threading.Thread(target=loopvalidation)
                threadValid.start()
            trainer.train()
            isTraining = False
        else:
            if not config.load_path:
                raise Exception("[!] You should specify `load_path` to load a pretrained model")
            print('batches=', data_loader.NumBatchValid)
            trainer.validate()


if __name__ == "__main__":
    config, unparsed = get_config()
    print(Fore.GREEN, 'config\n', config)
    print(Fore.RED, 'unparsed\n', unparsed)
    print(Style.RESET_ALL)
    if unparsed:
        assert(False)

    main(config)
