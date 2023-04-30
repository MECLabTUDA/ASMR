#!/bin/bash -l

#SBATCH --mail-user=mirko.konstantin@gris.informatik.tu-darmstadt.de
#SBATCH -J fl_crc_test_ana
#SBATCH -n 20
#SBATCH -c 8
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:4
#SBATCH -t 40:00:10

#SBATCH -o /gris/gris-f/homestud/mikonsta/master-thesis/FedPath/errlog/%m_%M.log
#SBATCH -e /gris/gris-f/homestud/mikonsta/master-thesis/FedPath/errlog/%j_%J.err

#python3 main.py --exp_path=experiment1/crc/ana_seeds/crc_ana_fedavgm_40.yml --logfile=test/crc_ana_40.log --gpu=0,1,2,3
python3 main.py --exp_path=experiment1/crc/ana_seeds/crc_ana_fedavgm_60.yml --logfile=test/crc_ana_60.log --gpu=0,1,2,3
python3 main.py --exp_path=experiment1/crc/ana_seeds/crc_ana_fedavgm_75.yml --logfile=test/crc_ana_75.log --gpu=0,1,2,3
python3 main.py --exp_path=experiment1/crc/ana_seeds/crc_ana_fedavgm_85.yml --logfile=test/crc_ana_75.log --gpu=0,1,2,3
