#! /bin/bash

source ~/.venv/epanns/bin/activate

cd ~/Desktop/earl/hebbian

python3 hebbian/train.py -w 46 -p 92 -e 4 -g 200 -n LunarLanderContinuous-v2 -c 0.0 0.01 0.1 1.0 -s 13 
python3 hebbian/train.py -w 46 -p 92 -e 4 -g 200 -n LunarLanderContinuous-v2 -c 0.0 0.01 0.1 1.0 -s 42
python3 hebbian/train.py -w 46 -p 92 -e 4 -g 200 -n LunarLanderContinuous-v2 -c 0.0 0.01 0.1 1.0 -s 501337
python3 hebbian/train.py -w 46 -p 92 -e 4 -g 200 -n LunarLanderContinuous-v2 -c 0.0 0.01 0.1 1.0 -s 139 
python3 hebbian/train.py -w 46 -p 92 -e 4 -g 200 -n LunarLanderContinuous-v2 -c 0.0 0.01 0.1 1.0 -s 43110
