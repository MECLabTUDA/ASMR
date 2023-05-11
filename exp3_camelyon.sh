#!/bin/bash

### Study - Experiment3 - ANA
python3 main.py --gpu=0,1 --exp_path=experiment3/camelyon17_ana_fedavgm_seed1.yml --logfile=final/exp3/k2/camelyon_ana_fedavgm_seed1.log
python3 main.py --gpu=0,1 --exp_path=experiment3/camelyon17_ana_fedavgm_seed2.yml --logfile=final/exp3/k2/camelyon_ana_fedavgm_seed2.log
python3 main.py --gpu=0,1 --exp_path=experiment3/camelyon17_ana_fedavgm_seed3.yml --logfile=final/exp3/k2/camelyon_ana_fedavgm_seed3.log
python3 main.py --gpu=0,1 --exp_path=experiment3/camelyon17_ana_fedavgm_seed4.yml --logfile=final/exp3/k2/camelyon_ana_fedavgm_seed4.log

### Study - Experiment3 - SFA
python3 main.py --gpu=0,1 --exp_path=experiment3/camelyon17_sfa_fedavgm_seed1.yml --logfile=final/exp3/k2/camelyon_sfa_fedavgm_seed1.log
python3 main.py --gpu=0,1 --exp_path=experiment3/camelyon17_sfa_fedavgm_seed2.yml --logfile=final/exp3/k2/camelyon_sfa_fedavgm_seed2.log
python3 main.py --gpu=0,1 --exp_path=experiment3/camelyon17_sfa_fedavgm_seed3.yml --logfile=final/exp3/k2/camelyon_sfa_fedavgm_seed3.log
python3 main.py --gpu=0,1 --exp_path=experiment3/camelyon17_sfa_fedavgm_seed4.yml --logfile=final/exp3/k2/camelyon_sfa_fedavgm_seed4.log

