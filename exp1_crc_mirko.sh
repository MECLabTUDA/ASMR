#!/bin/bash -l

#SBATCH --mail-user=mirko.konstantin@gris.informatik.tu-darmstadt.de
#SBATCH -J CRC_mirko
#SBATCH -n 20
#SBATCH -c 8
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:4
#SBATCH -t 40:00:10

#SBATCH -o /gris/gris-f/homestud/mikonsta/master-thesis/FedPath/errlog/%m_%M.log
#SBATCH -e /gris/gris-f/homestud/mikonsta/master-thesis/FedPath/errlog/%j_%J.err

## CRC - ANA - Mirko
#python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_ana_fedavgm_seed1.yml --logfile=final/exp1/mirko/crc/ana/crc_fedavgm_seed1.log
#python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_ana_fedavgm_seed2.yml --logfile=final/exp1/mirko/crc/ana/crc_fedavgm_seed2.log
#python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_ana_fedavgm_seed3.yml --logfile=final/exp1/mirko/crc/ana/crc_fedavgm_seed3.log
#python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_ana_fedavgm_seed4.yml --logfile=final/exp1/mirko/crc/ana/crc_fedavgm_seed4.log
#python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_ana_fedavgm_seed5.yml --logfile=final/exp1/mirko/crc/ana/crc_fedavgm_seed5.log
#python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_ana_fedavgm_seed6.yml --logfile=final/exp1/mirko/crc/ana/crc_fedavgm_seed6.log
#python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_ana_fedavgm_seed7.yml --logfile=final/exp1/mirko/crc/ana/crc_fedavgm_seed7.log
#python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_ana_fedavgm_seed8.yml --logfile=final/exp1/mirko/crc/ana/crc_fedavgm_seed8.log
#python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_ana_fedavgm_seed9.yml --logfile=final/exp1/mirko/crc/ana/crc_fedavgm_seed9.log
#python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_ana_fedavgm_seed10.yml --logfile=final/exp1/mirko/crc/ana/crc_fedavgm_seed10.log

## CRC - SFA - Mirko
python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_sfa_fedavgm_seed1.yml --logfile=final/exp1/mirko/crc/sfa/crc_fedavgm_seed1.log
#python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_sfa_fedavgm_seed2.yml --logfile=final/exp1/mirko/crc/sfa/crc_fedavgm_seed2.log
#python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_sfa_fedavgm_seed3.yml --logfile=final/exp1/mirko/crc/sfa/crc_fedavgm_seed3.log
#python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_sfa_fedavgm_seed4.yml --logfile=final/exp1/mirko/crc/sfa/crc_fedavgm_seed4.log
#python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_sfa_fedavgm_seed5.yml --logfile=final/exp1/mirko/crc/sfa/crc_fedavgm_seed5.log
#python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_sfa_fedavgm_seed6.yml --logfile=final/exp1/mirko/crc/sfa/crc_fedavgm_seed6.log
#python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_sfa_fedavgm_seed7.yml --logfile=final/exp1/mirko/crc/sfa/crc_fedavgm_seed7.log
#python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_sfa_fedavgm_seed8.yml --logfile=final/exp1/mirko/crc/sfa/crc_fedavgm_seed8.log
#python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_sfa_fedavgm_seed9.yml --logfile=final/exp1/mirko/crc/sfa/crc_fedavgm_seed9.log
#python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_sfa_fedavgm_seed10.yml --logfile=final/exp1/mirko/crc/sfa/crc_fedavgm_seed10.log
