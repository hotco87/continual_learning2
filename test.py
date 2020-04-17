# import torch
# from torch.distributions.categorical import Categorical
# m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
# a = m.sample()  # equal probability of 0, 1, 2, 3
# print(a)
# print(m.mode())


import torch
import torch.nn as nn
import torch.nn.functional as F

N = 10
C = 5
# softmax output by teacher
logits = torch.rand(N, C)
p = torch.softmax(logits, dim=1)
print("p: ",p)
# softmax output by student
q = torch.softmax(logits, dim=1)
print("q: ",q)
#q = torch.ones(N, C)
q.requires_grad = True
# KL Diverse
kl_loss = nn.KLDivLoss()(torch.log(q), p)
print("######")
print("torch.log(q) ",torch.log(q))
print(torch.log_softmax(logits,dim=1))
print("######")

kl_loss.backward()

# grad = q.grad
#
# q.grad.zero_()
# ce_loss = torch.mean(torch.log(q) * p)
# ce_loss.backward()
#
# grad_check = q.grad
# print (grad)
# print (grad_check)