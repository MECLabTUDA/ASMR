#!/bin/bash

### Clustering - ANA
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/clustering/camelyon17_ana_fedavgm_seed1.yml --logfile=final/exp2/clustering/camelyon/ana/camelyon_fedavgm_seed1.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/clustering/camelyon17_ana_fedavgm_seed2.yml --logfile=final/exp2/clustering/camelyon/ana/camelyon_fedavgm_seed2.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/clustering/camelyon17_ana_fedavgm_seed3.yml --logfile=final/exp2/clustering/camelyon/ana/camelyon_fedavgm_seed3.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/clustering/camelyon17_ana_fedavgm_seed4.yml --logfile=final/exp2/clustering/camelyon/ana/camelyon_fedavgm_seed4.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/clustering/camelyon17_ana_fedavgm_seed5.yml --logfile=final/exp2/clustering/camelyon/ana/camelyon_fedavgm_seed5.log

### Clustering - SFA
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/clustering/camelyon17_sfa_fedavgm_seed1.yml --logfile=final/exp2/clustering/camelyon/sfa/camelyon_fedavgm_seed1.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/clustering/camelyon17_sfa_fedavgm_seed2.yml --logfile=final/exp2/clustering/camelyon/sfa/camelyon_fedavgm_seed2.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/clustering/camelyon17_sfa_fedavgm_seed3.yml --logfile=final/exp2/clustering/camelyon/sfa/camelyon_fedavgm_seed3.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/clustering/camelyon17_sfa_fedavgm_seed4.yml --logfile=final/exp2/clustering/camelyon/sfa/camelyon_fedavgm_seed4.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/clustering/camelyon17_sfa_fedavgm_seed5.yml --logfile=final/exp2/clustering/camelyon/sfa/camelyon_fedavgm_seed5.log

### Krum - ANA
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/krum/camelyon17_ana_fedavgm_seed1.yml --logfile=final/exp2/krum/camelyon/ana/camelyon_fedavgm_seed1.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/krum/camelyon17_ana_fedavgm_seed2.yml --logfile=final/exp2/krum/camelyon/ana/camelyon_fedavgm_seed2.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/krum/camelyon17_ana_fedavgm_seed3.yml --logfile=final/exp2/krum/camelyon/ana/camelyon_fedavgm_seed3.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/krum/camelyon17_ana_fedavgm_seed4.yml --logfile=final/exp2/krum/camelyon/ana/camelyon_fedavgm_seed4.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/krum/camelyon17_ana_fedavgm_seed5.yml --logfile=final/exp2/krum/camelyon/ana/camelyon_fedavgm_seed5.log

### Krum - SFA
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/krum/camelyon17_sfa_fedavgm_seed1.yml --logfile=final/exp2/krum/camelyon/sfa/camelyon_fedavgm_seed1.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/krum/camelyon17_sfa_fedavgm_seed2.yml --logfile=final/exp2/krum/camelyon/sfa/camelyon_fedavgm_seed2.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/krum/camelyon17_sfa_fedavgm_seed3.yml --logfile=final/exp2/krum/camelyon/sfa/camelyon_fedavgm_seed3.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/krum/camelyon17_sfa_fedavgm_seed4.yml --logfile=final/exp2/krum/camelyon/sfa/camelyon_fedavgm_seed4.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/krum/camelyon17_sfa_fedavgm_seed5.yml --logfile=final/exp2/krum/camelyon/sfa/camelyon_fedavgm_seed5.log

### DnC - ANA
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/dnc/camelyon17_ana_fedavgm_seed1.yml --logfile=final/exp2/dnc/camelyon/ana/camelyon_fedavgm_seed1.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/dnc/camelyon17_ana_fedavgm_seed2.yml --logfile=final/exp2/dnc/camelyon/ana/camelyon_fedavgm_seed2.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/dnc/camelyon17_ana_fedavgm_seed3.yml --logfile=final/exp2/dnc/camelyon/ana/camelyon_fedavgm_seed3.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/dnc/camelyon17_ana_fedavgm_seed4.yml --logfile=final/exp2/dnc/camelyon/ana/camelyon_fedavgm_seed4.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/dnc/camelyon17_ana_fedavgm_seed5.yml --logfile=final/exp2/dnc/camelyon/ana/camelyon_fedavgm_seed5.log

### DnC - SFA
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/krum/camelyon17_sfa_fedavgm_seed1.yml --logfile=final/exp2/krum/camelyon/sfa/camelyon_fedavgm_seed1.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/krum/camelyon17_sfa_fedavgm_seed2.yml --logfile=final/exp2/krum/camelyon/sfa/camelyon_fedavgm_seed2.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/krum/camelyon17_sfa_fedavgm_seed3.yml --logfile=final/exp2/krum/camelyon/sfa/camelyon_fedavgm_seed3.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/krum/camelyon17_sfa_fedavgm_seed4.yml --logfile=final/exp2/krum/camelyon/sfa/camelyon_fedavgm_seed4.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/krum/camelyon17_sfa_fedavgm_seed5.yml --logfile=final/exp2/krum/camelyon/sfa/camelyon_fedavgm_seed5.log

### Mirko - ANA
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/mirko/camelyon17_ana_fedavgm_seed1.yml --logfile=final/exp2/mirko/camelyon/ana/camelyon_fedavgm_seed1.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/mirko/camelyon17_ana_fedavgm_seed2.yml --logfile=final/exp2/mirko/camelyon/ana/camelyon_fedavgm_seed2.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/mirko/camelyon17_ana_fedavgm_seed3.yml --logfile=final/exp2/mirko/camelyon/ana/camelyon_fedavgm_seed3.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/mirko/camelyon17_ana_fedavgm_seed4.yml --logfile=final/exp2/mirko/camelyon/ana/camelyon_fedavgm_seed4.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/mirko/camelyon17_ana_fedavgm_seed5.yml --logfile=final/exp2/mirko/camelyon/ana/camelyon_fedavgm_seed5.log

### Mirko - SFA
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/krum/camelyon17_sfa_fedavgm_seed1.yml --logfile=final/exp2/mirko/camelyon/sfa/camelyon_fedavgm_seed1.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/krum/camelyon17_sfa_fedavgm_seed2.yml --logfile=final/exp2/mirko/camelyon/sfa/camelyon_fedavgm_seed2.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/krum/camelyon17_sfa_fedavgm_seed3.yml --logfile=final/exp2/mirko/camelyon/sfa/camelyon_fedavgm_seed3.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/krum/camelyon17_sfa_fedavgm_seed4.yml --logfile=final/exp2/mirko/camelyon/sfa/camelyon_fedavgm_seed4.log
python3 main.py --gpu=0,1 --exp_path=experiment2/camelyon/krum/camelyon17_sfa_fedavgm_seed5.yml --logfile=final/exp2/mirko/camelyon/sfa/camelyon_fedavgm_seed5.log
