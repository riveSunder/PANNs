import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym 
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


def normalize_rewards(rewards):
    return rewards #(rewards - torch.mean(rewards)) / torch.std(rewards)

def get_advantage(t_rew, t_done, gamma=0.9):

    t_adv = torch.zeros_like(t_rew)
    t_adv[-1] = t_rew[-1]
    
    for idx in range (t_rew.shape[0]-2,-1,-1):
        
        t_adv[idx] = (1 - t_done[idx]) * gamma * t_adv[idx+1]\
                + t_rew[idx]

    #t_adv -= torch.mean(t_adv)

    # normalize advantage? 
    t_adv = normalize_rewards(t_adv)

    return t_adv

def print_stats(epoch, t_rew, t_done, total_env_interacts):

    epd_rewards = []
    epd_sum = 0.0
    for ii in range(t_rew.shape[0]):
        if t_done[ii]: 
            epd_rewards.append(epd_sum)
            epd_sum = 0.0
        else:
            epd_sum += t_rew[ii]
    
    mean_rew = torch.sum(t_rew)/(torch.sum(t_done)+1)
    mean_ep_len = t_rew.shape[0]/ (torch.sum(t_done)+1)
    std_rew = np.std(epd_rewards)
    max_rew = np.max(epd_rewards)
    min_rew = np.min(epd_rewards)

    print("""
    _________________________________
    | epoch:                {}       
    | total_env_interacts:  {:.2e}|
    | mean_ep_len:          {:.2e}|
    | mean_rew:             {:.2e}|
    | std_rew:              {:.2e}|
    | max_rew:              {:.2e}|
    | min_rew:              {:.2e}|
    |_______________________________|
    """.format(epoch, total_env_interacts, mean_ep_len, mean_rew, std_rew,\
            max_rew, min_rew))

    return mean_rew, mean_ep_len, std_rew, max_rew, min_rew

def print_stats_fit(epoch, fitness, total_env_interacts):

    mean_rew = np.mean(fitness) 
    std_rew = np.std(fitness)
    max_rew = np.max(fitness)
    min_rew = np.min(np.min(fitness))

    print("""
    _________________________________
    | generation:             {}       
    | total_env_interacts:  {:.2e}|
    | mean_rew:             {:.2e}|
    | std_rew:              {:.2e}|
    | max_rew:              {:.2e}|
    | min_rew:              {:.2e}|
    |_______________________________|
    """.format(epoch, total_env_interacts, mean_rew, std_rew,\
            max_rew, min_rew))


def train_evo(args):

    # set up hyperparameters/parameters
    env_name = args.env_name
    epds = args.epds
    epochs = args.gens
    seeds = args.seeds
    population_size = args.pop_size
    clamp_values = args.clamp_values
    performance_threshold = args.threshold

    save_every = 50

    exp_id = str(hash(time.time()))[0:6]

    exp_dir = "exp{}".format(exp_id)

    gravity = env_name == "LunarLanderContinuous-v2"
    for my_seed in seeds:
        torch.manual_seed(my_seed)
        np.random.seed(my_seed)

        for clamp_value in clamp_values:

            print("making env {}".format(env_name ))

            env = gym.make(env_name)

            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.sample().shape[0]

            agent_fn = HebbianMLP

            agents = DHGPopulation(input_dim=obs_dim, output_dim=act_dim, \
                    agent_fn=agent_fn, clamp_value=clamp_value, \
                    population_size=population_size)

            exp_name = "epann2_clamp{}_sd_{}".format(str(int(clamp_value*100)),my_seed)


            results = {"epoch": [],\
                "total_env_interacts": [],\
                "wall_time": [],\
                "mean_rew": [],\
                "std_rew": [],\
                "max_rew": [],\
                "min_rew": [],\
                "env_name": env_name,\
                "epds": epds,
                "pop_size": population_size,\
                "clamp_value": clamp_value,\
                "performance_threshold": performance_threshold\
                }

            agents.train(env, exp_dir=exp_dir, exp_name=exp_name,\
                    generations=epochs, epds=epds, results=results,\
                    performance_threshold=performance_threshold,\
                    gravity=gravity)

def train_backprop():

    # set up hyperparameters/parameters
    env_name = "InvertedPendulumBulletEnv-v0"
    hid_dims = [16,16]
    clamp_value = 0.0
    steps_per_epoch = 10000
    epochs = 3000
    sigma = 0.10
    gamma = 0.90
    batch_size = 10000
    save_every = 1000
    lr = 1e-3

    # define env, obs and action spaces
    print("making env", env_name)
    env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.sample().shape[0]
    # initialize agent(s)

    #agent = DirectedHebbianGraph(input_dim=obs_dim, output_dim=act_dim,\
    #        hid_dims=hid_dims)
    agent = HebbianMLP(input_dim=obs_dim, output_dim=act_dim,\
            hid_dims=hid_dims)
    optimizer = torch.optim.SGD(agent.parameters(), lr=lr)
    agent.clamp_value = 1e-6 # first validate vpg w/o Hebbian traces

    # train
    total_env_interacts = 0

    #exp_id = "exp_" + exp_time + "env_" +\
    #        env_name + "_s" + str(my_seed)
    results = {"epoch": [],\
            "total_env_interacts": [],\
            "wall_time": [],\
            "mean_rew": [],\
            "std_rew": [],\
            "max_rew": [],\
            "min_rew": [],\
            "env_name": env_name\
            }

    for my_seed in [0,1,2]:

        exp_id = "test_vpg" + str(my_seed)
        torch.manual_seed(my_seed)
        np.random.seed(my_seed)

        t0 = time.time()
        for epoch in range(epochs):
            # set up training buffer tensors
            t_rew = torch.Tensor()
            t_done = torch.Tensor()
            t_act = torch.Tensor()
            t_mu = torch.Tensor()
            step = 0
            done = True
            while step < steps_per_epoch:
                
                if done:
                    done = False
                    obs = env.reset()
                    agent.reset_hebbians()
                
                mu = agent.forward(torch.Tensor(obs)\
                        .reshape(1,obs.shape[0]))
                mu = nn.Tanh()(mu)

                #mu = torch.matmul(torch.Tensor(obs)\
                #        .reshape(1,obs.shape[0]),agent.x2y)[:,0:-1]

                # TODO: update to use multinomial and learn std dev as well
                #action = mu + sigma * torch.randn(1,mu.shape[0]) 
                action = torch.normal(mean=mu, std=sigma)

                obs, rew, done, info = env.step(action.detach().numpy())

                t_rew = torch.cat((t_rew, torch.Tensor(np.array(rew))\
                        .reshape(1,1)),dim=0)
                t_done = torch.cat((t_done, torch.Tensor(np.array(1.0*done))\
                        .reshape(1,1)), dim=0)
                t_act = torch.cat((t_act, torch.Tensor(action)\
                        .reshape(1,action.shape[0])), dim=0)
                t_mu = torch.cat((t_mu, torch.Tensor(mu)\
                        .reshape(1,mu.shape[0])), dim=0)

                step += 1

            t_adv = get_advantage(t_rew, t_done, gamma=gamma)
            total_env_interacts += step

            for batch_start in range(0,t_adv.shape[0],batch_size):

                agent.zero_grad()
                batch_end = batch_start + batch_size

                # REINFORCE type loss function 
                pseudo_loss = -torch.mean(t_adv[batch_start:batch_end]\
                        * 1./2. *  (t_mu[batch_start:batch_end] \
                        - t_act[batch_start:batch_end])**2 )

                pseudo_loss.backward(retain_graph=True)
                optimizer.step()


            mean_rew, mean_ep_len, std_rew, max_rew, min_rew = \
                print_stats(epoch, t_rew, t_done, total_env_interacts)
            
            print(torch.mean(agent.x2h0.grad))
            results["epoch"] = epoch
            results["total_env_interacts"] = total_env_interacts
            results["wall_time"] = time.time() - t0
            results["mean_rew"] = mean_rew
            results["std_rew"] = std_rew 
            results["max_rew"] = max_rew
            results["min_rew"] = min_rew


            if epoch % save_every == 0:
                np.save("./results/{}.npy"\
                        .format(exp_id),results)
                torch.save(agent.state_dict(),"./models/{}_epoch_{}.h5"\
                        .format(exp_id, epoch)) 
def mpi_fork(n):
  """Re-launches the current script with workers
  Returns "parent" for original parent, "child" for MPI children
  (from https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease, 
  which is in turn from https://github.com/garymcintire/mpi_util/)
  """
  if n<=1:
    return "child"

  if os.getenv("IN_MPI") is None:
    env = os.environ.copy()
    env.update(
      MKL_NUM_THREADS="1",
      OMP_NUM_THREADS="1",
      IN_MPI="1"
    )
    print( ["mpirun", "-np", str(n), sys.executable] + sys.argv)
    subprocess.check_call(["mpirun", "-np", str(n), sys.executable] \
        +['-u']+ sys.argv, env=env)
    return "parent"
  else:
    global nWorker, rank
    nWorker = comm.Get_size()
    rank = comm.Get_rank()
    #print('assigning the rank and nworkers', nWorker, rank)
    return "child"

def mantle(args):


    env_name = args.env_name
    epds = args.epds
    epochs = args.gens
    seeds = args.seeds
    population_size = args.pop_size
    clamp_values = args.clamp_values

    agent_fn = HebbianMLP

    performance_threshold = args.threshold

    save_every = 50

    exp_id = str(hash(time.time()))[0:6]

    exp_dir = "exp{}".format(exp_id)

    gravity = env_name == "LunarLanderContinuous-v2"
    for my_seed in seeds:

        torch.manual_seed(my_seed)
        np.random.seed(my_seed)

        for clamp_value in [clamp_values[0]]:

            print("making env {}".format(env_name ))

            env = gym.make(env_name)

            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.sample().shape[0]


            agent = DHGPopulation(input_dim=obs_dim, output_dim=act_dim, \
                    agent_fn=agent_fn, clamp_value=clamp_value, \
                    population_size=population_size)

            exp_name = "epann2_clamp{}_sd_{}".format(str(int(clamp_value*100)),my_seed)

            results = {"epoch": [],\
                "total_env_interacts": [],\
                "wall_time": [],\
                "mean_rew": [],\
                "std_rew": [],\
                "max_rew": [],\
                "min_rew": [],\
                "env_name": env_name,\
                "epds": epds,
                "pop_size": population_size,\
                "clamp_value": clamp_value,\
                "performance_threshold": performance_threshold\
                }

            t0 = time.time()
            for generation in range(epochs): 

                bb = 0
                fitness = []
                total_steps =0

                t1 = time.time()

                # delegate rollouts to arm processes

                while bb <= population_size: # - nWorker:
                    
                    # TODO: workers will also need to know clamp_value

                    pop_left = population_size - bb
                    for cc in range(1, min(nWorker, 1+pop_left)):
                        comm.send(agent.population[bb+cc-1], dest=cc)
                    
                    for cc in range(1, min(nWorker, 1+pop_left)):
                        #comm.send(agent.population[bb+cc-1], dest=cc)
                        fit = comm.recv(source=cc)
                        fitness.extend(fit[0])
                        total_steps += fit[1]

                    bb += cc

                agent.update_pop(fitness)


                print("gen {} mean fitness {:.3f}/ max {:.3f} , time elapsed/per gen {:.2f}/{:.2f}".\
                        format(generation, np.mean(fitness), np.max(fitness),\
                        time.time()-t0, (time.time() - t0)/(generation+1)))

    
def arm(args):

    env_name = args.env_name

    fickle_gravity = "LunarLander" in env_name

    env_name = args.env_name
    epds = args.epds
    clamp_values = args.clamp_values

    agent_fn = HebbianMLP

    population_size = 1

    print("making env {}".format(env_name ))
    env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.sample().shape[0]

    clamp_value = clamp_values[0]

    agent = DHGPopulation(input_dim=obs_dim, output_dim=act_dim, \
            agent_fn=agent_fn, clamp_value=clamp_value, \
            population_size=population_size)

    while True:

        my_policy = comm.recv(source=0)

        if my_policy == 0:
            print("worker {} shutting down".format(rank))
            break

        agent.population = [my_policy]

        fitness = agent.get_fitness(env, epds=epds, gravity=fickle_gravity)

        comm.send(fitness, dest=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=\
            "experimental parameters for EPANNS")
    parser.add_argument("-p", "--pop_size", type=int,\
            help="training population size", default=64)
    parser.add_argument("-e", "--epds", type=int,\
            help="num episodes to get fitness for each agent each generation",\
            default=8)
    parser.add_argument("-g", "--gens", type=int,\
            help="number of generations/epochs to train for",\
            default=50)
    parser.add_argument("-s", "--seeds", type=list,\
            help="random seeds",\
            default=[13,42,1337])
    parser.add_argument("-n", "--env_name", type=str,\
            help="name of environment",\
            default="InvertedPendulumSwingupBulletEnv-v0")
    parser.add_argument("-c", "--clamp_values", type=list,\
            help="clamp values that limit Hebbian trace weighting",
            default=[0.5, 0.0])
    parser.add_argument("-t", "--threshold", type=float,\
            help="performance threshold for stopping",\
            default=float("Inf"))
    parser.add_argument("-v", "--evo", type=bool,\
            help="Train evo (default,True) or vanilla policy gradient (False)",\
            default=True)
    parser.add_argument("-w", "--num_workers", type=int,\
            help="number of cpu threads to use",\
            default=4)

    args = parser.parse_args()

    if mpi_fork(args.num_workers+1) == "parent":
        os._exit(0)

    if rank == 0:
        mantle(args)
    else: 
        arm(args)
