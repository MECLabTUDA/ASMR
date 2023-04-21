#!/bin/bash

### Camelyon17

##Train model for spectral anomaly detection
#python3 train_camelyon_test.py

##Train Camelyon without attacks
#python3 main.py --exp_path=camelyon17_none_fedavgm.yml --logfile=camelyon_none_avgm_le1.log --gpu=0,1
#python3 main.py --exp_path=camelyon17_none_fedavg.yml --logfile=camelyon_none_avg_le1.log --gpu=0,1

##Train with ana attack without defense
#python3 main.py --exp_path=camelyon17_ana_fedavg.yml --logfile=camelyon_ana_avg.log --gpu=0,1
#python3 main.py --exp_path=camelyon17_ana_fedavgm.yml --logfile=camelyon_ana_avgm_le1_dp75.log --gpu=0,1

##Train with sfa attack without defense
#python3 main.py --exp_path=camelyon17_sfa_fedavg.yml --logfile=camelyon_sfa_avg.log --gpu=0,1
#python3 main.py --exp_path=camelyon17_sfa_fedavgm.yml --logfile=camelyon_sfa_avgm_le1.log --gpu=0,1

##Train with Krum defense
#python3 main.py --exp_path=camelyon17_ana_krum_fedavgm.yml --logfile=camelyon_ana_krum_avgm_le1_dyn.log --gpu=0,1
#python3 main.py --exp_path=camelyon17_sfa_krum_fedavgm.yml --logfile=camelyon_sfa_krum_avgm_le5_dyn.log --gpu=0,1

##Train with mirko defense
#python3 main.py --exp_path=camelyon17_ana_mirko_fedavgm.yml --logfile=camelyon_ana_mirko_avgm_le1.log --gpu=0,1
#python3 main.py --exp_path=camelyon17_sfa_mirko_fedavgm.yml --logfile=camelyon_sfa_mirko_avgm_le1_dyn_long.log --gpu=0,1

#Train with clustering defense
#python3 main.py --exp_path=camelyon17_ana_clustering_fedavgm.yml --logfile=camelyon_ana_clustering_avgm_le1.log --gpu=0,1
#python3 main.py --exp_path=camelyon17_sfa_clustering_fedavgm.yml --logfile=camelyon_sfa_clustering_avgm_le5_dyn.log --gpu=0,1

#Train with dnc defense
#python3 main.py --exp_path=camelyon17_ana_dnc_fedavgm.yml --logfile=camelyon_ana_dnc_avgm_le1.log --gpu=0,1
#python3 main.py --exp_path=camelyon17_sfa_dnc_fedavgm.yml --logfile=camelyon_sfa_dnc_avgm_le5_dyn.log --gpu=0,1

#python3 main.py --exp_path=camelyon17_ana_clustering_fedavg.yml --logfile=camelyon_ana_clustering_avgm_new.log --gpu=0,1
#python3 main.py --exp_path=camelyon17_ana_krum_fedavgm.yml --logfile=camelyon_ana_krum_avgm_new.log --gpu=0,1

python3 main.py --exp_path=crc_none_fedavg.yml --logfile=test_crc.log --gpu=0,1,2,3

#python3 main.py --exp_path=camelyon17_ana_dnc_fedavg.yml --logfile=camelyon_ana_dnc_avgm_new.log --gpu=0,1
