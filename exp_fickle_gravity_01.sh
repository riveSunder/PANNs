#! /bin/bash

source ~/.venv/epanns/bin/activate

cd ~/Desktop/earl/hebbian

python3 hebbian/train.py -w 40 -p 92 -e 8 -g 100 -n LunarLanderContinuous-v2 -c 0.0 0.01 0.1 1.0 -s 43110
python3 hebbian/train.py -w 40 -p 92 -e 8 -g 100 -n LunarLanderContinuous-v2 -c 0.0 0.01 0.1 1.0 -s 43770 
python3 hebbian/train.py -w 40 -p 92 -e 8 -g 100 -n LunarLanderContinuous-v2 -c 0.0 0.01 0.1 1.0 -s 13371355

#python3 hebbian/train.py -w 46 -p 92 -e 8 -g 200 -n LunarLanderContinuous-v2 -c 0.0 0.01 0.1 -s 5 
#python3 hebbian/train.py -w 46 -p 92 -e 8 -g 200 -n LunarLanderContinuous-v2 -c 0.0 0.01 0.1 -s 8
#python3 hebbian/train.py -w 46 -p 92 -e 8 -g 200 -n LunarLanderContinuous-v2 -c 0.0 0.01 0.1 -s 16
