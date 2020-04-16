#! /bin/bash

source ~/.venv/epanns/bin/activate

cd ~/Desktop/earl/hebbian

python3 hebbian/train.py -w 46 -p 92 -e 8 -g 300 -n LunarLanderContinuous-v2 -c 0.0 1.0 -s 13
python3 hebbian/train.py -w 46 -p 92 -e 8 -g 300 -n LunarLanderContinuous-v2 -c 0.0 1.0 -s 42
python3 hebbian/train.py -w 46 -p 92 -e 8 -g 300 -n LunarLanderContinuous-v2 -c 0.0 1.0 -s 1337

python3 hebbian/train.py -w 46 -p 92 -e 8 -g 300 -n LunarLanderContinuous-v2 -c 0.0 1.0 -s 5 
python3 hebbian/train.py -w 46 -p 92 -e 8 -g 300 -n LunarLanderContinuous-v2 -c 0.0 1.0 -s 8
python3 hebbian/train.py -w 46 -p 92 -e 8 -g 300 -n LunarLanderContinuous-v2 -c 0.0 1.0 -s 16
