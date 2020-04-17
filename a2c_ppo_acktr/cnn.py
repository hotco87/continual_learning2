import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import NNBase
#from a2c_ppo_acktr.model import NNBase
#from a2c_ppo_acktr.utils import init
from a2c_ppo_acktr.nn import NNBase
from a2c_ppo_acktr.utils import init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        from collections import OrderedDict
        self.main = nn.Sequential(OrderedDict([
            ('0',init_(nn.Conv2d(num_inputs, 32, 8, stride=4))), ('1',nn.ReLU()),
             ('2',init_(nn.Conv2d(32, 64, 4, stride=2))), ('3',nn.ReLU()),
              ('4',init_(nn.Conv2d(64, 32, 3, stride=1))), ('6',nn.ReLU()), ('7',Flatten()),
                ('8',init_(nn.Linear(32 * 7 * 7, hidden_size))), ('9',nn.ReLU())
               ]))

        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
        #                        constant_(x, 0))
        #
        # self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    # def forward(self, inputs, rnn_hxs, masks):
    #     x = self.main(inputs / 255.0) # logit
    #
    #     if self.is_recurrent:
    #         x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
    #
    #     return self.critic_linear(x), x, rnn_hxs
    def forward(self, inputs, rnn_hxs, masks):

        x = self.main(inputs / 255.0) # logit
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return  x, rnn_hxs

class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
