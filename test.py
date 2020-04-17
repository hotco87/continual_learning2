import torch
from torch.distributions.categorical import Categorical
m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
a = m.sample()  # equal probability of 0, 1, 2, 3
print(a)
print(m.mode())