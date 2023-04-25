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
#python3 main.py --exp_path=camelyon17_fedavgm_seed1.yml --logfile=camelyon_sfa_avgm_le1.log --gpu=0,1

##Train with Krum defense
#/gris/gris-f/homestud/mikonsta/master-thesis/FedPath/logs/final/experiment1_seeds
#python3 main.py --exp_path=camelyon17_ana_krum_fedavgm.yml --logfile=camelyon_ana_krum_avgm_le1_dyn.log --gpu=0,1
#python3 main.py --exp_path=camelyon17_sfa_krum_fedavgm.yml --logfile=camelyon_sfa_krum_avgm_le5_dyn.log --gpu=0,1

python3 main.py --exp_path=krum/camelyon17_ana_krum_fedavgm_seed1.yml --logfile=final/experiment1_seeds/krum/camelyon_ana_krum_avgm_le1_seed1.log --gpu=0,1
python3 main.py --exp_path=krum/camelyon17_ana_krum_fedavgm_seed2.yml --logfile=final/experiment1_seeds/krum/camelyon_ana_krum_avgm_le1_seed2.log --gpu=0,1
python3 main.py --exp_path=krum/camelyon17_ana_krum_fedavgm_seed3.yml --logfile=final/experiment1_seeds/krum/camelyon_ana_krum_avgm_le1_seed3.log --gpu=0,1
python3 main.py --exp_path=krum/camelyon17_ana_krum_fedavgm_seed4.yml --logfile=final/experiment1_seeds/krum/camelyon_ana_krum_avgm_le1_seed4.log --gpu=0,1
python3 main.py --exp_path=krum/camelyon17_ana_krum_fedavgm_seed5.yml --logfile=final/experiment1_seeds/krum/camelyon_ana_krum_avgm_le1_seed5.log --gpu=0,1
python3 main.py --exp_path=krum/camelyon17_ana_krum_fedavgm_seed6.yml --logfile=final/experiment1_seeds/krum/camelyon_ana_krum_avgm_le1_seed6.log --gpu=0,1
python3 main.py --exp_path=krum/camelyon17_ana_krum_fedavgm_seed7.yml --logfile=final/experiment1_seeds/krum/camelyon_ana_krum_avgm_le1_seed7.log --gpu=0,1
python3 main.py --exp_path=krum/camelyon17_ana_krum_fedavgm_seed8.yml --logfile=final/experiment1_seeds/krum/camelyon_ana_krum_avgm_le1_seed8.log --gpu=0,1
python3 main.py --exp_path=krum/camelyon17_ana_krum_fedavgm_seed9.yml --logfile=final/experiment1_seeds/krum/camelyon_ana_krum_avgm_le1_seed9.log --gpu=0,1
python3 main.py --exp_path=krum/camelyon17_ana_krum_fedavgm_seed10.yml --logfile=final/experiment1_seeds/krum/camelyon_ana_krum_avgm_le1_seed10.log --gpu=0,1

##Train with mirko defense
#python3 main.py --exp_path=camelyon17_ana_mirko_fedavgm.yml --logfile=camelyon_ana_mirko_avgm_le1.log --gpu=0,1
#python3 main.py --exp_path=camelyon17_sfa_mirko_fedavgm.yml --logfile=camelyon_sfa_mirko_avgm_le1_dyn_long.log --gpu=0,1

python3 main.py --exp_path=mirko/camelyon17_ana_mirko_fedavgm_seed1.yml --logfile=final/experiment1_seeds/mirko/camelyon_ana_mirko_avgm_le1_seed1.log --gpu=0,1
python3 main.py --exp_path=mirko/camelyon17_ana_mirko_fedavgm_seed2.yml --logfile=final/experiment1_seeds/mirko/camelyon_ana_mirko_avgm_le1_seed2.log --gpu=0,1
python3 main.py --exp_path=mirko/camelyon17_ana_mirko_fedavgm_seed3.yml --logfile=final/experiment1_seeds/mirko/camelyon_ana_mirko_avgm_le1_seed3.log --gpu=0,1
python3 main.py --exp_path=mirko/camelyon17_ana_mirko_fedavgm_seed4.yml --logfile=final/experiment1_seeds/mirko/camelyon_ana_mirko_avgm_le1_seed4.log --gpu=0,1
python3 main.py --exp_path=mirko/camelyon17_ana_mirko_fedavgm_seed5.yml --logfile=final/experiment1_seeds/mirko/camelyon_ana_mirko_avgm_le1_seed5.log --gpu=0,1
python3 main.py --exp_path=mirko/camelyon17_ana_mirko_fedavgm_seed6.yml --logfile=final/experiment1_seeds/mirko/camelyon_ana_mirko_avgm_le1_seed6.log --gpu=0,1
python3 main.py --exp_path=mirko/camelyon17_ana_mirko_fedavgm_seed7.yml --logfile=final/experiment1_seeds/mirko/camelyon_ana_mirko_avgm_le1_seed7.log --gpu=0,1
python3 main.py --exp_path=mirko/camelyon17_ana_mirko_fedavgm_seed8.yml --logfile=final/experiment1_seeds/mirko/camelyon_ana_mirko_avgm_le1_seed8.log --gpu=0,1
python3 main.py --exp_path=mirko/camelyon17_ana_mirko_fedavgm_seed9.yml --logfile=final/experiment1_seeds/mirko/camelyon_ana_mirko_avgm_le1_seed9.log --gpu=0,1
python3 main.py --exp_path=mirko/camelyon17_ana_mirko_fedavgm_seed10.yml --logfile=final/experiment1_seeds/mirko/camelyon_ana_mirko_avgm_le1_seed10.log --gpu=0,1


#Train with clustering defense
#python3 main.py --exp_path=camelyon17_ana_clustering_fedavgm.yml --logfile=camelyon_ana_clustering_avgm_le1.log --gpu=0,1
#python3 main.py --exp_path=camelyon17_sfa_clustering_fedavgm.yml --logfile=camelyon_sfa_clustering_avgm_le5_dyn.log --gpu=0,1

python3 main.py --exp_path=clustering/camelyon17_ana_clustering_fedavgm_seed1.yml --logfile=final/experiment1_seeds/clustering/camelyon_ana_clustering_avgm_le1_seed1.log --gpu=0,1
python3 main.py --exp_path=clustering/camelyon17_ana_clustering_fedavgm_seed2.yml --logfile=final/experiment1_seeds/clustering/camelyon_ana_clustering_avgm_le1_seed2.log --gpu=0,1
python3 main.py --exp_path=clustering/camelyon17_ana_clustering_fedavgm_seed3.yml --logfile=final/experiment1_seeds/clustering/camelyon_ana_clustering_avgm_le1_seed3.log --gpu=0,1
python3 main.py --exp_path=clustering/camelyon17_ana_clustering_fedavgm_seed4.yml --logfile=final/experiment1_seeds/clustering/camelyon_ana_clustering_avgm_le1_seed4.log --gpu=0,1
python3 main.py --exp_path=clustering/camelyon17_ana_clustering_fedavgm_seed5.yml --logfile=final/experiment1_seeds/clustering/camelyon_ana_clustering_avgm_le1_seed5.log --gpu=0,1
python3 main.py --exp_path=clustering/camelyon17_ana_clustering_fedavgm_seed6.yml --logfile=final/experiment1_seeds/clustering/camelyon_ana_clustering_avgm_le1_seed6.log --gpu=0,1
python3 main.py --exp_path=clustering/camelyon17_ana_clustering_fedavgm_seed7.yml --logfile=final/experiment1_seeds/clustering/camelyon_ana_clustering_avgm_le1_seed7.log --gpu=0,1
python3 main.py --exp_path=clustering/camelyon17_ana_clustering_fedavgm_seed8.yml --logfile=final/experiment1_seeds/clustering/camelyon_ana_clustering_avgm_le1_seed8.log --gpu=0,1
python3 main.py --exp_path=clustering/camelyon17_ana_clustering_fedavgm_seed9.yml --logfile=final/experiment1_seeds/clustering/camelyon_ana_clustering_avgm_le1_seed9.log --gpu=0,1
python3 main.py --exp_path=clustering/camelyon17_ana_clustering_fedavgm_seed10.yml --logfile=final/experiment1_seeds/clustering/camelyon_ana_clustering_avgm_le1_seed10.log --gpu=0,1


#Train with dnc defense
#python3 main.py --exp_path=camelyon17_ana_dnc_fedavgm.yml --logfile=camelyon_ana_dnc_avgm_le1.log --gpu=0,1
#python3 main.py --exp_path=camelyon17_sfa_dnc_fedavgm.yml --logfile=camelyon_sfa_dnc_avgm_le5_dyn.log --gpu=0,1

python3 main.py --exp_path=dnc/camelyon17_ana_dnc_fedavgm_seed1.yml --logfile=final/experiment1_seeds/dnc/camelyon_ana_dnc_avgm_le1_seed1.log --gpu=0,1
python3 main.py --exp_path=dnc/camelyon17_ana_dnc_fedavgm_seed2.yml --logfile=final/experiment1_seeds/dnc/camelyon_ana_dnc_avgm_le1_seed2.log --gpu=0,1
python3 main.py --exp_path=dnc/camelyon17_ana_dnc_fedavgm_seed3.yml --logfile=final/experiment1_seeds/dnc/camelyon_ana_dnc_avgm_le1_seed3.log --gpu=0,1
python3 main.py --exp_path=dnc/camelyon17_ana_dnc_fedavgm_seed4.yml --logfile=final/experiment1_seeds/dnc/camelyon_ana_dnc_avgm_le1_seed4.log --gpu=0,1
python3 main.py --exp_path=dnc/camelyon17_ana_dnc_fedavgm_seed5.yml --logfile=final/experiment1_seeds/dnc/camelyon_ana_dnc_avgm_le1_seed5.log --gpu=0,1
python3 main.py --exp_path=dnc/camelyon17_ana_dnc_fedavgm_seed6.yml --logfile=final/experiment1_seeds/dnc/camelyon_ana_dnc_avgm_le1_seed6.log --gpu=0,1
python3 main.py --exp_path=dnc/camelyon17_ana_dnc_fedavgm_seed7.yml --logfile=final/experiment1_seeds/dnc/camelyon_ana_dnc_avgm_le1_seed7.log --gpu=0,1
python3 main.py --exp_path=dnc/camelyon17_ana_dnc_fedavgm_seed8.yml --logfile=final/experiment1_seeds/dnc/camelyon_ana_dnc_avgm_le1_seed8.log --gpu=0,1
python3 main.py --exp_path=dnc/camelyon17_ana_dnc_fedavgm_seed9.yml --logfile=final/experiment1_seeds/dnc/camelyon_ana_dnc_avgm_le1_seed9.log --gpu=0,1
python3 main.py --exp_path=dnc/camelyon17_ana_dnc_fedavgm_seed10.yml --logfile=final/experiment1_seeds/dnc/camelyon_ana_dnc_avgm_le1_seed10.log --gpu=0,1


#python3 main.py --exp_path=camelyon17_ana_clustering_fedavg.yml --logfile=camelyon_ana_clustering_avgm_new.log --gpu=0,1
#python3 main.py --exp_path=camelyon17_ana_krum_fedavgm.yml --logfile=slurm_new.log --gpu=0,1,2,3
#python3 main.py --exp_path=camelyon17_ana_krum_fedavgm.yml --logfile=camelyon_ana_krum_avgm_new.log --gpu=0,1,2,3

#python3 main.py --exp_path=crc_none_fedavg.yml --logfile=test_crc.log --gpu=0,1,2,3

#python3 main.py --exp_path=camelyon17_ana_dnc_fedavg.yml --logfile=camelyon_ana_dnc_avgm_new.log --gpu=0,1
