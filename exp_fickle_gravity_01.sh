#! /bin/bash

source ~/.venv/epanns/bin/activate

cd ~/Desktop/earl/hebbian

python3 hebbian/train.py -w 43 -p 92 -e 8 -g 100 -n LunarLanderContinuous-v2 -c 0.1 0.01 0.0 1.0 -s 43770 
python3 hebbian/train.py -w 43 -p 92 -e 8 -g 100 -n LunarLanderContinuous-v2 -c 0.0 0.01 0.1 1.0 -s 43110
python3 hebbian/train.py -w 43 -p 92 -e 8 -g 100 -n LunarLanderContinuous-v2 -c 0.0 0.01 0.1 1.0 -s 13371355

python3 hebbian/train.py -w 43 -p 92 -e 8 -g 100 -n LunarLanderContinuous-v2 -c 0.0 0.01 0.1 1.0 -s 15 
python3 hebbian/train.py -w 43 -p 92 -e 8 -g 100 -n LunarLanderContinuous-v2 -c 0.0 0.01 0.1 1.0 -s 18

python3 hebbian/train.py -w 43 -p 92 -e 8 -g 100 -n LunarLanderContinuous-v2 -c 0.0 0.01 0.1 1.0 -s 116
