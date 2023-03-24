#!/bin/bash

#python3 main.py --exp_path=camelyon17_test_fedavgm.yml --gpu=1 
#python3 main.py --exp_path=camelyon17_sign_flipping_fedavgm.yml --gpu=2
#python3 main.py --exp_path=camelyon17_ana_fedavgm.yml --gpu=3

python3 main.py --exp_path=camelyon17_none_fedavgm.yml --logfile=train_camelyon.log --gpu=3,4,5
python3 main.py --exp_path=camelyon17_none_fedavgm.yml --logfile=train_ana_camelyon.log --gpu=3,4,5
python3 main.py --exp_path=camelyon17_none_fedavgm.yml --logfile=train_sfa_camelyon.log --gpu=3,4,5
