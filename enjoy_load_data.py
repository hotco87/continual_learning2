import argparse
import os
import sys
import numpy as np
import torch
from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
import numpy as np
sys.path.append('a2c_ppo_acktr')

rollouts_list = torch.load("./collect_data/rollouts_list.pt")
print(np.shape(rollouts_list))
print(np.shape(rollouts_list)[0])
print(type(np.shape(rollouts_list)[0]))


# for i in rollouts_list:
#     print(np.shape(i.obs))
#     print(np.shape(i.actions))
    #print(i.obs[:-1].view(-1))
#print(rollouts_list.size())
#print(rollouts_list.obs[:-1].view(-1))


# filename = "rollout.obj"
# filehandler = open(filename, 'wb')
# pickle.dump(rollouts, filehandler)
#
# ################################################################
#
# import pickle
# filehandler = open(filename, 'rb')
# saved_rollouts = pickle.load(filehandler)
#
# print(saved_rollouts.obs[:-1].view(-1))
# print(11)





#
# save_state = [t.size() for t in save_state]
# save_action = torch.LongTensor(save_action)
# save_value = torch.FloatTensor(save_value)
# save_probs = [t.size() for t in save_probs]
#
# torch.save(save_state,"./collect_data/save_state.pt")
# torch.save(save_action,"./collect_data/save_action.pt")
# torch.save(save_action,"./collect_data/save_value.pt")
# torch.save(save_probs,"./collect_data/save_probs.pt")

#################################################
# Data Load ####+

# save_state = torch.load("save_state.pt")
# save_action = torch.load("save_state.pt")
#
# for i, y in zip(save_state,save_action):
#     print("#########")
#     print(i)
#     print(y)