################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#


################################################################################

from __future__ import absolute_import, division, print_function

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

################################################################################


class LSTM(nn.Module):

    def __init__(
        self, seq_length, input_dim, num_hidden, num_classes, batch_size, device="cpu"
    ):
        super(LSTM, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device

        # Initialize LSTM parameters
        self.W_xi = nn.Parameter(torch.randn(input_dim, num_hidden))
        self.W_hi = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.b_i = nn.Parameter(torch.zeros(num_hidden))

        self.W_xf = nn.Parameter(torch.randn(input_dim, num_hidden))
        self.W_hf = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.b_f = nn.Parameter(torch.zeros(num_hidden))

        self.W_xg = nn.Parameter(torch.randn(input_dim, num_hidden))
        self.W_hg = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.b_g = nn.Parameter(torch.zeros(num_hidden))

        self.W_xo = nn.Parameter(torch.randn(input_dim, num_hidden))
        self.W_ho = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.b_o = nn.Parameter(torch.zeros(num_hidden))

        self.W_hy = nn.Parameter(torch.randn(num_hidden, num_classes))
        self.b_y = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        assert x.size() == (
            self.batch_size,
            self.seq_length,
            self.input_dim,
        ), f"Expected input size ({self.batch_size}, {self.seq_length}, {self.input_dim}), got {x.size()}"

        # Initialize hidden state and cell state
        h_t = torch.zeros(self.batch_size, self.num_hidden).to(self.device)
        c_t = torch.zeros(self.batch_size, self.num_hidden).to(self.device)

        # Loop through time steps
        for t in range(self.seq_length):
            x_t = x[:, t, :]  # Input at time step t

            # Source: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
            i_t = torch.sigmoid(
                torch.matmul(x_t, self.W_xi) + torch.matmul(h_t, self.W_hi) + self.b_i
            )
            f_t = torch.sigmoid(
                torch.matmul(x_t, self.W_xf) + torch.matmul(h_t, self.W_hf) + self.b_f
            )
            g_t = torch.tanh(
                torch.matmul(x_t, self.W_xg) + torch.matmul(h_t, self.W_hg) + self.b_g
            )
            o_t = torch.sigmoid(
                torch.matmul(x_t, self.W_xo) + torch.matmul(h_t, self.W_ho) + self.b_o
            )

            c_t = f_t * c_t + i_t * g_t  # i keep forgetting to forget you :)
            h_t = o_t * torch.tanh(c_t)

        # Output layer
        output = torch.matmul(h_t, self.W_hy) + self.b_y
        return output
