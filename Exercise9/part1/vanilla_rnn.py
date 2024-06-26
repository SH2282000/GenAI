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
import torch.nn.functional as F

################################################################################


class VanillaRNN(nn.Module):

    def __init__(
        self, seq_length, input_dim, num_hidden, num_classes, batch_size, device="cpu"
    ):
        super(VanillaRNN, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device

        # Initialize parameters
        self.W_hh = nn.Parameter(
            torch.randn(num_hidden, num_hidden)
        )  # Hidden state transition matrix
        self.W_xh = nn.Parameter(
            torch.randn(input_dim, num_hidden)
        )  # Input to hidden state matrix
        self.W_hy = nn.Parameter(
            torch.randn(num_hidden, num_classes)
        )  # Hidden to output matrix
        self.b_h = nn.Parameter(torch.zeros(num_hidden))  # Hidden state bias
        self.b_y = nn.Parameter(torch.zeros(num_classes))  # Output bias

    def forward(self, x):
        assert x.size() == (
            self.batch_size,
            self.seq_length,
            self.input_dim,
        ), f"Expected input size ({self.batch_size}, {self.seq_length}, {self.input_dim}), got {x.size()}"

        # Initialize hidden state
        h_t = torch.zeros(self.batch_size, self.num_hidden).to(self.device)

        # Loop through time steps
        for t in range(self.seq_length):
            x_t = x[:, t, :]  # Input at time step t
            h_t = torch.tanh(
                torch.matmul(x_t, self.W_xh) + torch.matmul(h_t, self.W_hh) + self.b_h
            )

        # Output layer
        output = torch.matmul(h_t, self.W_hy) + self.b_y
        return output
