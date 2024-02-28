#!/bin/bash -l

#SBATCH --mail-user=mirko.konstantin@gris.informatik.tu-darmstadt.de
#SBATCH -J CRC_mirko
#SBATCH -n 20
#SBATCH -c 8
#SBATCH --mem-per-cpu=6000
#SBATCH --gres=gpu:4
#SBATCH -t 90:00:10

#SBATCH -o /gris/gris-f/homestud/mikonsta/master-thesis/FedPath/errlog/%m_%M.log
#SBATCH -e /gris/gris-f/homestud/mikonsta/master-thesis/FedPath/errlog/%j_%J.err

## CRC - ANA - Mirko - exp1
python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_ana_fedavgm_seed1.yml --logfile=final/exp1/mirko/crc/ana/crc_fedavgm_seed1.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_ana_fedavgm_seed3.yml --logfile=final/exp1/mirko/crc/ana/crc_fedavgm_seed3.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_ana_fedavgm_seed5.yml --logfile=final/exp1/mirko/crc/ana/crc_fedavgm_seed5.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_ana_fedavgm_seed7.yml --logfile=final/exp1/mirko/crc/ana/crc_fedavgm_seed7.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_ana_fedavgm_seed9.yml --logfile=final/exp1/mirko/crc/ana/crc_fedavgm_seed9.log

## CRC - SFA - Mirko - exp1
python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_sfa_fedavgm_seed1.yml --logfile=final/exp1/mirko/crc/sfa/crc_fedavgm_seed1.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_sfa_fedavgm_seed3.yml --logfile=final/exp1/mirko/crc/sfa/crc_fedavgm_seed3.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_sfa_fedavgm_seed5.yml --logfile=final/exp1/mirko/crc/sfa/crc_fedavgm_seed5.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_sfa_fedavgm_seed7.yml --logfile=final/exp1/mirko/crc/sfa/crc_fedavgm_seed7.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/mirko/crc_sfa_fedavgm_seed9.yml --logfile=final/exp1/mirko/crc/sfa/crc_fedavgm_seed9.log


## CRC - ANA - Mirko - exp2
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/mirko/crc_ana_fedavgm_seed1.yml --logfile=final/exp2/mirko/crc/ana/crc_fedavgm_seed1.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/mirko/crc_ana_fedavgm_seed2.yml --logfile=final/exp2/mirko/crc/ana/crc_fedavgm_seed2.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/mirko/crc_ana_fedavgm_seed3.yml --logfile=final/exp2/mirko/crc/ana/crc_fedavgm_seed3.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/mirko/crc_ana_fedavgm_seed4.yml --logfile=final/exp2/mirko/crc/ana/crc_fedavgm_seed4.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/mirko/crc_ana_fedavgm_seed5.yml --logfile=final/exp2/mirko/crc/ana/crc_fedavgm_seed5.log

## CRC - SFA - Mirko - exp2
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/mirko/crc_sfa_fedavgm_seed1.yml --logfile=final/exp2/mirko/crc/sfa/crc_fedavgm_seed1.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/mirko/crc_sfa_fedavgm_seed2.yml --logfile=final/exp2/mirko/crc/sfa/crc_fedavgm_seed2.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/mirko/crc_sfa_fedavgm_seed3.yml --logfile=final/exp2/mirko/crc/sfa/crc_fedavgm_seed3.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/mirko/crc_sfa_fedavgm_seed4.yml --logfile=final/exp2/mirko/crc/sfa/crc_fedavgm_seed4.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/mirko/crc_sfa_fedavgm_seed5.yml --logfile=final/exp2/mirko/crc/sfa/crc_fedavgm_seed5.log

### CRC - ANA - No Defense - exp1
python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/no_defense/crc_ana_fedavgm_seed1.yml --logfile=final/exp1/pure/crc/no_attack/crc_ana_fedavgm_seed1.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/no_defense/crc_ana_fedavgm_seed3.yml --logfile=final/exp1/pure/crc/no_attack/crc_ana_fedavgm_seed3.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/no_defense/crc_ana_fedavgm_seed5.yml --logfile=final/exp1/pure/crc/no_attack/crc_ana_fedavgm_seed5.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/no_defense/crc_ana_fedavgm_seed7.yml --logfile=final/exp1/pure/crc/no_attack/crc_ana_fedavgm_seed7.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment1/crc/no_defense/crc_ana_fedavgm_seed9.yml --logfile=final/exp1/pure/crc/no_attack/crc_ana_fedavgm_seed9.log



### CRC - ANA - No Defense - exp2
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/no_defense/crc_ana_fedavgm_seed1.yml --logfile=final/exp2/pure/crc/no_attack/crc_ana_fedavgm_seed1.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/no_defense/crc_ana_fedavgm_seed2.yml --logfile=final/exp2/pure/crc/no_attack/crc_ana_fedavgm_seed2.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/no_defense/crc_ana_fedavgm_seed3.yml --logfile=final/exp2/pure/crc/no_attack/crc_ana_fedavgm_seed3.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/no_defense/crc_ana_fedavgm_seed4.yml --logfile=final/exp2/pure/crc/no_attack/crc_ana_fedavgm_seed4.log
python3 main.py --gpu=0,1,2,3 --exp_path=experiment2/crc/no_defense/crc_ana_fedavgm_seed5.yml --logfile=final/exp2/pure/crc/no_attack/crc_ana_fedavgm_seed5.log
