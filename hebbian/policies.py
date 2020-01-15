import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym 
import matplotlib.pyplot as plt
import copy
import time
import os

import pybullet
import pybullet_envs


class DHGPopulation():

    def __init__(self, input_dim, output_dim, agent_fn, population_size=96,\
            clamp_value=1e-3):

        self.population_size = population_size
        self.population = []
        for idx in range(self.population_size):
            self.population.append(agent_fn(input_dim=input_dim,\
                    output_dim = output_dim, requires_grad=False))
            self.population[idx].clamp_value = clamp_value 

        self.best_gen = -float("Inf")
        self.best_agent = -float("Inf")

        self.mean = torch.zeros(self.population[0].total_params)
        self.variance = 1e-1 * torch.ones(self.population[0].total_params)


    def get_fitness(self, env, epds=6):

        fitness = []
        complexity = []
        total_steps = 0

        for agent_idx in range(len(self.population)):
            #obs = flatten_obs(env.reset())
            accumulated_reward = 0.0
            for epd in range(epds):

                obs = env.reset()

                self.population[agent_idx].reset_hebbians()
                done=False
                while not done:
                    action = self.population[agent_idx].forward(\
                            torch.Tensor(obs).reshape(1,obs.shape[0]))
                    action = nn.Tanh()(action)
                    obs, rew, done, info = env.step(action.detach().numpy())
                    total_steps += 1
                    accumulated_reward += rew

            fitness.append(accumulated_reward/(epds))

        return fitness, total_steps

    def update_pop(self, fitness):
        
        sort_indices = list(np.argsort(fitness))
        sort_indices.reverse()

        sorted_fitness = np.array(fitness)[sort_indices]

        keep = int(np.ceil(0.125*self.population_size))
        if sorted_fitness[0] >= self.best_agent:
            # keep best agent
            print("new best elite agent: {} v {}".\
                    format(sorted_fitness[0], self.best_agent))
            self.elite_agent = self.population[sort_indices[0]]
            self.best_agent = sorted_fitness[0]

        if np.mean(sorted_fitness[:keep]) >= self.best_gen:
            # keep best elite population
            print("new best elite population: {} v {}".\
                    format(np.mean(sorted_fitness[:keep]), self.best_gen))
            self.best_gen = np.mean(sorted_fitness[:keep])

        self.elite_pop = []
        self.elite_pop.append(self.elite_agent)
        for oo in range(keep):
            
            self.elite_pop.append(self.population[sort_indices[oo]])
        self.population[0:keep] = self.elite_pop
        self.population.insert(0,self.elite_agent)
        
        # determine parameter means and variance
        variance = 1e0 * torch.ones(self.elite_pop[0].total_params)
        mean = torch.zeros(self.elite_pop[0].total_params)
        pop_mean = torch.zeros(self.elite_pop[0].total_params)

        dim0 = 0

        # compute cma
        # compute pop mean

        for agent_idx in range(len(self.population)):

            pop_params = np.array([])
            agent = self.population[agent_idx]
            for params in [agent.x2y, agent.x2h0, agent.x2h1,\
                    agent.h02y, agent.h02h1, agent.h12y]:
                pop_params = np.append(pop_params, \
                        params.detach().numpy().ravel())
            pop_mean += torch.tensor(pop_params,requires_grad=False)

        pop_mean = pop_mean / len(self.population)
        pop_mean = pop_mean.reshape(1,pop_mean.shape[0])
        
        

        elite_params = torch.Tensor()

        for agent_idx in range(keep):

            agent_params = np.array([])
            agent = self.population[agent_idx]
            for params in [agent.x2y, agent.x2h0, agent.x2h1,\
                    agent.h02y, agent.h02h1, agent.h12y]:
                agent_params = np.append(agent_params, \
                        params.detach().numpy().ravel())

            elite_params = torch.cat((elite_params,torch.Tensor(agent_params)\
                    .reshape(1,agent_params.shape[0])))
            mean += torch.tensor(agent_params,requires_grad=False)

        
        covariance = torch.matmul( (elite_params - pop_mean).T,\
                elite_params-pop_mean)

        covariance += 1e-10 * torch.eye(elite_params.shape[1])
        covariance = torch.clamp(covariance,-1e2,1e2)
        print(torch.mean(covariance))

        alpha = 1e-1
        mean = mean / keep
        self.mean = mean #alpha *  mean  

        for agent_idx in range(keep, self.population_size):
            self.population[agent_idx].init_parameters(mean=mean,\
                    variance=covariance)
        self.population.append(self.elite_agent)

    def train(self, env, generations=1000):

        for generation in range(generations):
            fitness, total_steps = self.get_fitness(env)
            self.update_pop(fitness)

            print("generation {} max/mean fitness: {}/{}"\
                    .format(generation, np.max(fitness), np.mean(fitness)))

class HebbianMLP(nn.Module):
    """
    Multi-Layer Perceptron with Hebbian memory

    (x)-->h0-->h1-->(y)

    """
    def __init__(self, input_dim, output_dimm, hid_dims=[16,16], \
            requires_grad=True, no_traces=False):
        super(DirectedHebbianGraph, self).__init__()


        self.input_dim = input_dim
        # +1 for the Hebbian trace control variable W
        self.output_dim = output_dim + 1
        self.hid_dims = hid_dims
        self.total_params = input_dim * hid_dims[0]\
                + hid_dims[0] * hid_dims[1]\
                + hid_dims[1] * self.output_dim

        self.requires_grad = requires_grad
        self.no_traces = no_traces

        self.alpha = 0.3
        self.clamp_value = .30
        self.act = torch.nn.ReLU()
        if self.requires_grad:
            self.init_parameters()
        else:
            mean = torch.zeros(self.total_params)
            variance = 1e0 * torch.ones(self.total_params)
            variance = torch.eye(self.total_params)
            self.init_parameters(mean=mean, variance=variance)

    def forward(self, x):

        h0_x = torch.matmul(x, self.x2h0 + self.W * self.heb_x2h0)
        h0 = self.act(h0_x)
        
        h1_0 = torch.matmul(h0, self.h02h1 + self.W * self.heb_h02h1)
        h1 = self.act( h1_0)

        y_1 = torch.matmul(h1, self.h12y + self.W * self.heb_h12y)
        y = y_1

        # any final activation function is external to the model for now

        if self.no_traces:
            self.W = 0.0
        else: 
            self.W = torch.mean(nn.Tanh()(y[:,-2:-1]))

            self.heb_h12y = torch.clamp(self.heb_h12y\
                + self.alpha * torch.matmul(h1.T, y_1),\
                -self.clamp_value, self.clamp_value)

            self.heb_h02h1 = torch.clamp(self.heb_h02h1\
                + self.alpha * torch.matmul(h0.T, h1_0),\
                -self.clamp_value, self.clamp_value)

            self.heb_x2h0 = torch.clamp(self.heb_x2h0\
                + self.alpha * torch.matmul(x.T, h0_x),\
                -self.clamp_value, self.clamp_value)

        y = y[:,:-1]

        return y

    def init_parameters(self, mean=None, variance=None):
        """
        initialize parameters from a mean and covariance matrix
        """
        if self.requires_grad:

            self.x2h0 = nn.Parameter(torch.randn(self.input_dim,\
                    self.hid_dims[0])\
                    * np.sqrt(2./self.input_dim),\
                    requires_grad = self.requires_grad)

            self.h02h1 = nn.Parameter(torch.randn(self.hid_dims[0],\
                    self.hid_dims[1])\
                    * np.sqrt(2./self.hid_dims[0]),\
                    requires_grad = self.requires_grad)

            self.h12y = nn.Parameter(torch.randn(self.hid_dims[1],\
                    self.output_dim)\
                    * np.sqrt(2./self.hid_dims[1]),\
                    requires_grad = self.requires_grad)
        else:
            assert variance is not None, "specify variance"

            # for NES
            #params = torch.normal(mean, variance)
            # for cma
            #m = torch.distributions.multivariate_normal.MultivariateNormal
            #dist = m(mean, variance)
            params = np.random.multivariate_normal(mean.detach().numpy(),\
                    variance.detach().numpy())

            params = torch.Tensor(params)

            dim_x2h0 = self.input_dim * self.hid_dims[0]
            self.x2h0 = nn.Parameter(params[0:dim_x2h0],\
                    requires_grad=self.requires_grad)\
                    .reshape(self.input_dim, self.hid_dims[0])

            dim_h02h1 = self.hid_dims[0] * self.hid_dims[1] + dim_x2h0
            self.h02h1 = nn.Parameter(params[dim_x2h    0:dim_h02h1],\
                    requires_grad=self.requires_grad)\
                    .reshape(self.hid_dims[0], self.hid_dims[1])

            dim_h12y = self.hid_dims[1] * self.output_dim +  dim_h02h
            self.h12y = nn.Parameter(params[dim_h02h1:dim_h12y],\
                    requires_grad=self.requires_grad)\
                    .reshape(self.hid_dims[1], self.output_dim)

        self.reset_hebbians()

    def reset_hebbians(self):
        # Hebbian initializations
        self.heb_x2h0 = torch.zeros(self.input_dim, self.hid_dims[0],\
                requires_grad=False)

        self.heb_h02h1 = torch.zeros(self.hid_dims[0], sddelf.hid_dims[1],\
                requires_grad=False)

        self.heb_h12y = torch.zeros(self.hid_dims[1], self.output_dim,\
                requires_grad=False)

        self.W = torch.zeros(1,1)

class DirectedHebbianGraph(nn.Module):

    """ A directed acyclic graph with Hebbian memory
        Currently limited to two hidden layers
           ______________ 
          /________      |
         /         v     v
        (x)-->h0-->h1-->(y)
         \___ ^|_________^
    """


    def __init__(self,input_dim, output_dim, hid_dims=[16,16], \
            requires_grad=True):
        super(DirectedHebbianGraph, self).__init__()

        self.input_dim = input_dim
        # +1 for the Hebbian trace control variable W
        self.output_dim = output_dim + 1
        self.hid_dims = hid_dims
        self.total_params = input_dim * self.output_dim\
                + input_dim * hid_dims[0]\
                + input_dim * hid_dims[1]\
                + hid_dims[0] * hid_dims[1]\
                + hid_dims[0] * self.output_dim\
                + hid_dims[1] * self.output_dim

        self.requires_grad = requires_grad

        self.alpha = 0.3
        self.clamp_value = .30
        self.act = torch.nn.ReLU()
        if self.requires_grad:
            self.init_parameters()
        else:
            mean = torch.zeros(self.total_params)
            variance = 1e0 * torch.ones(self.total_params)
            variance = torch.eye(self.total_params)
            self.init_parameters(mean=mean, variance=variance)


    def forward(self, x):

        h0_x = torch.matmul(x, self.x2h0 + self.W * self.heb_x2h0)
        self.heb_x2h0 = torch.clamp(self.heb_x2h0\
                + self.alpha * torch.matmul(x.T, h0_x),\
                -self.clamp_value, self.clamp_value)
        h0 = self.act(h0_x)
        
        h1_x = torch.matmul(x, self.x2h1 + self.W * self.heb_x2h1)
        self.heb_x2h1 = torch.clamp(self.heb_x2h1\
                + self.alpha * torch.matmul(x.T, h1_x),\
                -self.clamp_value, self.clamp_value)
        h1_0 = torch.matmul(h0, self.h02h1 + self.W * self.heb_h02h1)
        self.heb_h02h1 = torch.clamp(self.heb_h02h1\
                + self.alpha * torch.matmul(h0.T, h1_0),\
                -self.clamp_value, self.clamp_value)
        h1 = self.act(h1_x + h1_0)

        y_x = torch.matmul(x, self.x2y+ self.W * self.heb_x2y)
        self.heb_x2y = torch.clamp(self.heb_x2y\
                + self.alpha * torch.matmul(x.T, y_x),\
                -self.clamp_value, self.clamp_value)
        y_0 = torch.matmul(h0, self.h02y + self.W * self.heb_h02y)
        self.heb_h02y = torch.clamp(self.heb_h02y\
                + self.alpha * torch.matmul(h0.T, y_0),\
                -self.clamp_value, self.clamp_value)
        y_1 = torch.matmul(h1, self.h12y + self.W * self.heb_h12y)
        self.heb_h12y = torch.clamp(self.heb_h12y\
                + self.alpha * torch.matmul(h1.T, y_1),\
                -self.clamp_value, self.clamp_value)

        y = y_x + y_0 + y_1

        # any final activation function is external to the model for now

        self.W = torch.mean(nn.Tanh()(y[:,-2:-1]))
        y = y[:,:-1]

        return y

    def init_parameters(self, mean=None, variance=None):
        """
        initialize parameters from a mean and covariance matrix
        """
        if self.requires_grad:

            self.x2h0 = nn.Parameter(torch.randn(self.input_dim,\
                    self.hid_dims[0])\
                    * np.sqrt(2./self.input_dim),\
                    requires_grad = self.requires_grad)

            self.x2h1 = nn.Parameter(torch.randn(self.input_dim, \
                    self.hid_dims[1])\
                    * np.sqrt(2./self.input_dim),\
                    requires_grad = self.requires_grad)

            self.h02h1 = nn.Parameter(torch.randn(self.hid_dims[0],\
                    self.hid_dims[1])\
                    * np.sqrt(2./self.hid_dims[0]),\
                    requires_grad = self.requires_grad)

            self.x2y = nn.Parameter(torch.randn(self.input_dim,\
                    self.output_dim)\
                    * np.sqrt(2./self.input_dim),\
                    requires_grad = self.requires_grad)

            self.h02y = nn.Parameter(torch.randn(self.hid_dims[0],\
                    self.output_dim)\
                    * np.sqrt(2./self.hid_dims[0]),\
                    requires_grad = self.requires_grad)

            self.h12y = nn.Parameter(torch.randn(self.hid_dims[1],\
                    self.output_dim)\
                    * np.sqrt(2./self.hid_dims[1]),\
                    requires_grad = self.requires_grad)
        else:
            assert variance is not None, "specify variance"

            # for NES
            #params = torch.normal(mean, variance)
            # for cma
            #m = torch.distributions.multivariate_normal.MultivariateNormal
            #dist = m(mean, variance)
            params = np.random.multivariate_normal(mean.detach().numpy(),\
                    variance.detach().numpy())

            params = torch.Tensor(params)

            dim_x2h0 = self.input_dim * self.hid_dims[0]
            self.x2h0 = nn.Parameter(params[0:dim_x2h0],\
                    requires_grad=self.requires_grad)\
                    .reshape(self.input_dim, self.hid_dims[0])

            dim_x2h1 = self.input_dim * self.hid_dims[1] + dim_x2h0
            self.x2h1 = nn.Parameter(params[dim_x2h0:dim_x2h1],\
                    requires_grad=self.requires_grad)\
                    .reshape(self.input_dim, self.hid_dims[1])

            dim_h02h1 = self.hid_dims[0] * self.hid_dims[1] + dim_x2h1
            self.h02h1 = nn.Parameter(params[dim_x2h1:dim_h02h1],\
                    requires_grad=self.requires_grad)\
                    .reshape(self.hid_dims[0], self.hid_dims[1])

            dim_x2y = self.input_dim * self.output_dim + dim_h02h1
            self.x2y = nn.Parameter(params[dim_h02h1:dim_x2y],\
                    requires_grad=self.requires_grad)\
                    .reshape(self.input_dim, self.output_dim)

            dim_h02y = self.hid_dims[0] * self.output_dim + dim_x2y
            self.h02y = nn.Parameter(params[dim_x2y:dim_h02y],\
                    requires_grad=self.requires_grad)\
                    .reshape(self.hid_dims[0], self.output_dim)

            dim_h12y = self.hid_dims[1] * self.output_dim +  dim_h02y
            self.h12y = nn.Parameter(params[dim_h02y:dim_h12y],\
                    requires_grad=self.requires_grad)\
                    .reshape(self.hid_dims[1], self.output_dim)

        self.reset_hebbians()

    def reset_hebbians(self):
        # Hebbian initializations
        self.heb_x2h0 = torch.zeros(self.input_dim, self.hid_dims[0],\
                requires_grad=False)

        self.heb_h02h1 = torch.zeros(self.hid_dims[0], self.hid_dims[1],\
                requires_grad=False)
        self.heb_x2h1 = torch.zeros(self.input_dim, self.hid_dims[1],\
                requires_grad=False)

        self.heb_x2y = torch.zeros(self.input_dim, self.output_dim,\
                requires_grad=False)
        self.heb_h02y = torch.zeros(self.hid_dims[0], self.output_dim,\
                requires_grad=False)
        self.heb_h12y = torch.zeros(self.hid_dims[1], self.output_dim,\
                requires_grad=False)

        self.W = torch.zeros(1,1)


if __name__ == "__main__":

    x = torch.randn(128, 8)
    y = torch.randn(128, 1)

    model = DirectedHebbianGraph(input_dim=x.shape[1], output_dim=y.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.clamp_value = 1e-5
    print(model)

    print(model.parameters)

    for step in range(20):

        model.zero_grad()

        y_pred = model(x)

        loss = torch.mean(torch.pow(y_pred-y,2))

        loss.backward(retain_graph=True)
        optimizer.step()

        if step % 10 == 0:
            print("loss at step {} = {:.3f}".format(step,loss))


    env = gym.make("InvertedPendulumBulletEnv-v0")
    act_dim = 1
    obs_dim = 5

    agent_fn = HebbianMLP

    agents = DHGPopulation(input_dim=obs_dim, output_dim=act_dim, \
            agent_fn=agent_fn)

    agents.train(env)

    import pdb; pdb.set_trace()

    agents = DHGPopulation(input_dim=obs_dim, output_dim=act_dim, \
            agent_fn=agent_fn,clamp_value=1e-9)

    agents.train(env)

    import pdb; pdb.set_trace()

