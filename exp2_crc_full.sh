#!/bin/bash -l

#SBATCH --mail-user=mirko.konstantin@gris.informatik.tu-darmstadt.de
#SBATCH -J CRC_exp2_full
#SBATCH -n 20
#SBATCH -c 8
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:4
#SBATCH -t 240:00:10

#SBATCH -o /gris/gris-f/homestud/mikonsta/master-thesis/FedPath/errlog/%m_%M.log
#SBATCH -e /gris/gris-f/homestud/mikonsta/master-thesis/FedPath/errlog/%j_%J.err

### CRC - ANA - No Defense
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/no_defense/crc_ana_fedavgm_seed1.yml --logfile=final/exp2/pure/crc/no_attack/crc_ana_fedavgm_seed1.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/no_defense/crc_ana_fedavgm_seed2.yml --logfile=final/exp2/pure/crc/no_attack/crc_ana_fedavgm_seed2.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/no_defense/crc_ana_fedavgm_seed3.yml --logfile=final/exp2/pure/crc/no_attack/crc_ana_fedavgm_seed3.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/no_defense/crc_ana_fedavgm_seed4.yml --logfile=final/exp2/pure/crc/no_attack/crc_ana_fedavgm_seed4.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/no_defense/crc_ana_fedavgm_seed5.yml --logfile=final/exp2/pure/crc/no_attack/crc_ana_fedavgm_seed5.log

## CRC - ANA - DnC
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/dnc/crc_ana_fedavgm_seed1.yml --logfile=final/exp2/dnc/crc/ana/crc_fedavgm_seed1.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/dnc/crc_ana_fedavgm_seed2.yml --logfile=final/exp2/dnc/crc/ana/crc_fedavgm_seed2.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/dnc/crc_ana_fedavgm_seed3.yml --logfile=final/exp2/dnc/crc/ana/crc_fedavgm_seed3.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/dnc/crc_ana_fedavgm_seed4.yml --logfile=final/exp2/dnc/crc/ana/crc_fedavgm_seed4.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/dnc/crc_ana_fedavgm_seed5.yml --logfile=final/exp2/dnc/crc/ana/crc_fedavgm_seed5.log

## CRC - SFA - DnC
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/dnc/crc_sfa_fedavgm_seed1.yml --logfile=final/exp2/dnc/crc/sfa/crc_fedavgm_seed1.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/dnc/crc_sfa_fedavgm_seed2.yml --logfile=final/exp2/dnc/crc/sfa/crc_fedavgm_seed2.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/dnc/crc_sfa_fedavgm_seed3.yml --logfile=final/exp2/dnc/crc/sfa/crc_fedavgm_seed3.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/dnc/crc_sfa_fedavgm_seed4.yml --logfile=final/exp2/dnc/crc/sfa/crc_fedavgm_seed4.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/dnc/crc_sfa_fedavgm_seed5.yml --logfile=final/exp2/dnc/crc/sfa/crc_fedavgm_seed5.log

## CRC - ANA - Clustering
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/clustering/crc_ana_fedavgm_seed1.yml --logfile=final/exp2/clustering/crc/ana/crc_fedavgm_seed1.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/clustering/crc_ana_fedavgm_seed2.yml --logfile=final/exp2/clustering/crc/ana/crc_fedavgm_seed2.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/clustering/crc_ana_fedavgm_seed3.yml --logfile=final/exp2/clustering/crc/ana/crc_fedavgm_seed3.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/clustering/crc_ana_fedavgm_seed4.yml --logfile=final/exp2/clustering/crc/ana/crc_fedavgm_seed4.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/clustering/crc_ana_fedavgm_seed5.yml --logfile=final/exp2/clustering/crc/ana/crc_fedavgm_seed5.log

## CRC - SFA - Clustering
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/clustering/crc_sfa_fedavgm_seed1.yml --logfile=final/exp2/clustering/crc/sfa/crc_fedavgm_seed1.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/clustering/crc_sfa_fedavgm_seed2.yml --logfile=final/exp2/clustering/crc/sfa/crc_fedavgm_seed2.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/clustering/crc_sfa_fedavgm_seed3.yml --logfile=final/exp2/clustering/crc/sfa/crc_fedavgm_seed3.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/clustering/crc_sfa_fedavgm_seed4.yml --logfile=final/exp2/clustering/crc/sfa/crc_fedavgm_seed4.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/clustering/crc_sfa_fedavgm_seed5.yml --logfile=final/exp2/clustering/crc/sfa/crc_fedavgm_seed5.log

## CRC - ANA - Mirko
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/mirko/crc_ana_fedavgm_seed1.yml --logfile=final/exp2/mirko/crc/ana/crc_fedavgm_seed1.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/mirko/crc_ana_fedavgm_seed2.yml --logfile=final/exp2/mirko/crc/ana/crc_fedavgm_seed2.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/mirko/crc_ana_fedavgm_seed3.yml --logfile=final/exp2/mirko/crc/ana/crc_fedavgm_seed3.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/mirko/crc_ana_fedavgm_seed4.yml --logfile=final/exp2/mirko/crc/ana/crc_fedavgm_seed4.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/mirko/crc_ana_fedavgm_seed5.yml --logfile=final/exp2/mirko/crc/ana/crc_fedavgm_seed5.log

## CRC - SFA - Mirko
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/mirko/crc_sfa_fedavgm_seed1.yml --logfile=final/exp2/mirko/crc/sfa/crc_fedavgm_seed1.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/mirko/crc_sfa_fedavgm_seed2.yml --logfile=final/exp2/mirko/crc/sfa/crc_fedavgm_seed2.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/mirko/crc_sfa_fedavgm_seed3.yml --logfile=final/exp2/mirko/crc/sfa/crc_fedavgm_seed3.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/mirko/crc_sfa_fedavgm_seed4.yml --logfile=final/exp2/mirko/crc/sfa/crc_fedavgm_seed4.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/mirko/crc_sfa_fedavgm_seed5.yml --logfile=final/exp2/mirko/crc/sfa/crc_fedavgm_seed5.log

## CRC - ANA - Krum
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/krum/crc_ana_fedavgm_seed1.yml --logfile=final/exp2/krum/crc/ana/crc_fedavgm_seed1.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/krum/crc_ana_fedavgm_seed2.yml --logfile=final/exp2/krum/crc/ana/crc_fedavgm_seed2.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/krum/crc_ana_fedavgm_seed3.yml --logfile=final/exp2/krum/crc/ana/crc_fedavgm_seed3.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/krum/crc_ana_fedavgm_seed4.yml --logfile=final/exp2/krum/crc/ana/crc_fedavgm_seed4.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/krum/crc_ana_fedavgm_seed5.yml --logfile=final/exp2/krum/crc/ana/crc_fedavgm_seed5.log

## CRC - SFA - Krum
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/krum/crc_sfa_fedavgm_seed1.yml --logfile=final/exp2/krum/crc/sfa/crc_fedavgm_seed1.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/krum/crc_sfa_fedavgm_seed2.yml --logfile=final/exp2/krum/crc/sfa/crc_fedavgm_seed2.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/krum/crc_sfa_fedavgm_seed3.yml --logfile=final/exp2/krum/crc/sfa/crc_fedavgm_seed3.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/krum/crc_sfa_fedavgm_seed4.yml --logfile=final/exp2/krum/crc/sfa/crc_fedavgm_seed4.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/krum/crc_sfa_fedavgm_seed5.yml --logfile=final/exp2/krum/crc/sfa/crc_fedavgm_seed5.log
