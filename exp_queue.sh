#!/bin/bash

python main.py --exp_path=camelyon17_base.yml --gpu=0,3,4,5,6,7
python main.py --exp_path=camelyon17_base_fedavg.yml --gpu=0,1,2,3