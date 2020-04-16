import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym 

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import copy
import time
import os
import argparse

#mpi paralellization stuff
from mpi4py import MPI
comm = MPI.COMM_WORLD
import subprocess
import sys

import pybullet
import pybullet_envs

from policies import DirectedHebbianGraph, HebbianMLP, DHGPopulation

from train import print_stats_fit

import Box2D
from Box2D import b2Vec2

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description=\
            "parameters for enjoying trained EPANNS")
    parser.add_argument("-n", "--env_name", type=str,\
            help="name of environment",\
            default="LunarLanderContinuous-v2")
    parser.add_argument("-m", "--model_filepath", nargs="+",type=str,\
            help="model to load", default=["./models/exp559445/model_epann2_clamp10_sd_43110_gen99.h5"])
    parser.add_argument("-s", "--steps", type=int,\
            help="number of steps in gravity range", default = 7)
    parser.add_argument("-x", "--max_gravity_x", type=float,\
            help="magnitude of min and max gravity perturbation in x", default=0.40)
    parser.add_argument("-y", "--max_gravity_y", type=float,\
            help="magnitude of min and max gravity perturbation in y", default=7.0)
    parser.add_argument("-e", "--epds", type=int,\
            help="num episodes to get fitness for each agent each generation",\
            default=8)

    args = parser.parse_args()

    env_name = args.env_name
    env = gym.make(env_name)
    epds = args.epds
    model_filepaths = args.model_filepath

    # base gravity
    g_x = 0.0
    # lunar gravity
    g_y = -1.625

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.sample().shape[0]

    agent = HebbianMLP(input_dim=obs_dim, output_dim=act_dim)

    res_mat = np.zeros((args.steps+1, args.steps+1))
    xx = 0
    yy = 0
    
    for model_filepath in model_filepaths:
        
        agent.load_state_dict(torch.load(model_filepath))

        xx = 0

        for gravity_x in np.linspace(- args.max_gravity_x, args.max_gravity_x, args.steps):
            yy = 0
            for gravity_y in np.linspace(-args.max_gravity_y, args.max_gravity_y, args.steps):

                env.env.world.gravity = b2Vec2(\
                        g_x + gravity_x,\
                        - g_y + gravity_y)


                fitness = []
                total_steps = 0
                for epd in range(epds):
                    obs = env.reset()
                    done = False
                    accumulated_reward = 0
                    while not done:
                        action = agent.forward(\
                            torch.Tensor(obs).reshape(1,obs.shape[0]))

                        action = nn.Tanh()(action)
                        if action.shape[1] > 1:
                            action = action.squeeze()

                        obs, rew, done, info = env.step(action.detach().numpy())
                        if 1:
                            env.render()
                            time.sleep(0.01)

                        total_steps += 1
                        accumulated_reward += rew
                    fitness.append(accumulated_reward)


                print(xx,yy)
                res_mat[xx,yy] = np.mean(fitness)
                print("gravity x: {:.2e} y: {:.2e} mean fitness {:.2e}".format(\
                        gravity_x, gravity_y, res_mat[xx,yy]))
                print(agent.W, agent.dopa)
                #print(fitness)

                print_stats_fit(0, fitness, total_steps, gravity_x, -4.9 + gravity_y, clamp=None)

                yy += 1
            xx += 1

        plt.figure()
        plt.imshow(res_mat)
        plt.colorbar()
        #plt.savefig("./temp.png")
        plt.show()
