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

        self.Wg = nn.Parameter(torch.Tensor(input_dim + num_hidden, num_hidden))
        self.Wi = nn.Parameter(torch.Tensor(input_dim + num_hidden, num_hidden))
        self.Wf = nn.Parameter(torch.Tensor(input_dim + num_hidden, num_hidden))
        self.Wo = nn.Parameter(torch.Tensor(input_dim + num_hidden, num_hidden))

        self.bg = nn.Parameter(torch.Tensor(num_hidden))
        self.bi = nn.Parameter(torch.Tensor(num_hidden))
        self.bf = nn.Parameter(torch.Tensor(num_hidden))
        self.bo = nn.Parameter(torch.Tensor(num_hidden))

        self.Woh = nn.Parameter(torch.Tensor(num_hidden, num_classes))
        self.bo_logits = nn.Parameter(torch.Tensor(num_classes))

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.Wg, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wi, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wf, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wo, a=math.sqrt(5))
        nn.init.constant_(self.bg, 0)
        nn.init.constant_(self.bi, 0)
        nn.init.constant_(self.bf, 0)
        nn.init.constant_(self.bo, 0)
        nn.init.kaiming_uniform_(self.Woh, a=math.sqrt(5))
        nn.init.constant_(self.bo_logits, 0)

    def lstm_step(self, lstm_state_tuple, x):
        c_prev, h_prev = lstm_state_tuple

        x_and_h = torch.cat((x, h_prev), dim=1)

        preact_g = torch.matmul(x_and_h, self.Wg) + self.bg
        preact_i = torch.matmul(x_and_h, self.Wi) + self.bi
        preact_f = torch.matmul(x_and_h, self.Wf) + self.bf
        preact_o = torch.matmul(x_and_h, self.Wo) + self.bo

        g = torch.tanh(preact_g)
        i = torch.sigmoid(preact_i)
        f = torch.sigmoid(preact_f)
        o = torch.sigmoid(preact_o)

        c = g * i + c_prev * f
        h = torch.tanh(c) * o

        return c, h

    def forward(self, x):
        batch_size = x.size(1)
        h = torch.zeros(batch_size, self.num_hidden, device=x.device)
        c = torch.zeros(batch_size, self.num_hidden, device=x.device)

        for t in range(self.seq_length):
            x = x[t]
            c, h = self.lstm_step((c, h), x)

        logits = torch.matmul(h, self.Woh) + self.bo_logits

        return logits

    def compute_loss(self, logits, targets):
        loss = F.cross_entropy(logits, targets)
        return loss

    def accuracy(self, logits, targets):
        pred_class = torch.argmax(logits, dim=1)
        true_class = torch.argmax(targets, dim=1)
        accuracy = (pred_class == true_class).float().mean()
        return accuracy
