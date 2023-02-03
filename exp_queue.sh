#!/bin/bash

python main.py --exp_path=./configs/camelyon17_base.yml --gpu=0,1,2,3
python main.py --exp_path=./configs/camelyon17_base_fedavg.yml --gpu=0,1,2,3