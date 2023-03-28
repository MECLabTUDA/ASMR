#!/bin/bash

#python3 main.py --exp_path=camelyon17_none_fedavgm.yml --logfile=camelyon_none_avgm.log --gpu=2,3
#python3 main.py --exp_path=camelyon17_none_fedavg.yml --logfile=camelyon_none_avg.log --gpu=2,3

python3 main.py --exp_path=camelyon17_ana_fedavgm.yml --logfile=camelyon_ana_avgm_new.log --gpu=2,3,4
python3 main.py --exp_path=camelyon17_ana_fedavg.yml --logfile=camelyon_ana_avg_new.log --gpu=2,3,4

python3 main.py --exp_path=camelyon17_sfa_fedavgm.yml --logfile=camelyon_sfa_avgm_new.log --gpu=2,3,4
python3 main.py --exp_path=camelyon17_sfa_fedavg.yml --logfile=camelyon_sfa_avg_new.log --gpu=2,3,4

