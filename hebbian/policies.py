import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym 
import matplotlib.pyplot as plt
import copy
import time
import os


class DirectedHebbianGraph(nn.Module):

    """ A directed acyclic graph with Hebbian memory
        Currently limited to two hidden layers
           ______________ 
          /________      |
         /         v     v
        (x)-->h0-->h1-->(y)
         \___ ^|_________^
    """


    def __init__(self,input_dim, output_dim, hid_dims=[32,32], requires_grad=True):
        super(DirectedHebbianGraph, self).__init__()

        self.input_dim = input_dim
        # +1 for the Hebbian trace control variable W
        self.output_dim = output_dim + 1
        self.hid_dims = hid_dims
        self.requires_grad = requires_grad

        self.alpha = 0.3
        self.clamp_value = .30
        self.act = torch.nn.LeakyReLU()
        self.init_parameters()

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

        y_x = torch.matmul(x, self.x2y)
        self.heb_x2y = torch.clamp(self.heb_x2y\
                + self.alpha * torch.matmul(x.T, y_x),\
                -self.clamp_value, self.clamp_value)
        y_0 = torch.matmul(h0, self.h02y)
        self.heb_h02y = torch.clamp(self.heb_h02y\
                + self.alpha * torch.matmul(h0.T, y_0),\
                -self.clamp_value, self.clamp_value)
        y_1 = torch.matmul(h1, self.h12y)
        self.heb_h12y = torch.clamp(self.heb_h12y\
                + self.alpha * torch.matmul(h1.T, y_1),\
                -self.clamp_value, self.clamp_value)

        y = y_x + y_0 + y_1

        # any final activation function is external to the model for now

        self.W = torch.mean(nn.Tanh()(y[:,-2:-1]))
        y = y[:,:-1]

        return y

    def init_parameters(self, mean=None, covariance=None):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    print(model)

    for step in range(2000):

        model.zero_grad()

        y_pred = model(x)

        loss = torch.mean(torch.pow(y_pred-y,2))

        loss.backward(retain_graph=True)
        optimizer.step()

        if step % 100 == 0:
            print("loss at step {} = {:.3f}".format(step,loss))

    import pdb; pdb.set_trace()
