#!/bin/bash

### Camelyon17

##Train model for spectral anomaly detection
#python3 train_camelyon_test.py

##Train Camelyon without attacks
#python3 main.py --exp_path=camelyon17_none_fedavgm.yml --logfile=camelyon_none_avgm.log --gpu=0,1
#python3 main.py --exp_path=camelyon17_none_fedavg.yml --logfile=camelyon_none_avg.log --gpu=0,1

##Train with ana attack without defense
#python3 main.py --exp_path=camelyon17_ana_fedavg.yml --logfile=camelyon_ana_avg.log --gpu=0,1
#python3 main.py --exp_path=camelyon17_ana_fedavgm.yml --logfile=camelyon_ana_avgm.log --gpu=0,1

##Train with sfa attack without defense
#python3 main.py --exp_path=camelyon17_sfa_fedavg.yml --logfile=camelyon_sfa_avg.log --gpu=0,1
#python3 main.py --exp_path=camelyon17_sfa_fedavgm.yml --logfile=camelyon_sfa_avgm.log --gpu=0,1

##Train with ana attack with Krum defense
#python3 main.py --exp_path=camelyon17_ana_krum_fedavgm.yml --logfile=camelyon_ana_krum_avgm.log --gpu=0
python3 main.py --exp_path=camelyon17_ana_mirko_fedavg.yml --logfile=camelyon_ana_mirko_avgm_new.log --gpu=0,1

##Train with sfa attack with Krum defense
#python3 main.py --exp_path=camelyon17_sfa_krum_fedavgm.yml --logfile=camelyon_sfa_krum_avgm.log --gpu=0,1
#python3 main.py --exp_path=camelyon17_sfa_mirko_fedavgm.yml --logfile=camelyon_sfa_mirko_avgm.log --gpu=0,1

#Train with ana attack with own defense

#Train with sfa attack with own defense

