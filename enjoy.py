import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='PongNoFrameskip-v4',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
args = parser.parse_args()

args.det = not args.non_det
import json

file_path = "config.json"
setup_json = json.load(open(file_path, 'r'))
env_conf = setup_json["Default"]
for i in setup_json.keys():
    if i in args.env_name:
        env_conf = setup_json[i]

device = torch.device("cpu")
env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1,
    None,
    None,
    device,
    env_conf,
    allow_early_resets=False)

# Get a render function
render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training

# from a2c_ppo_acktr.cnn import CNNBase
# from a2c_ppo_acktr.model import Policy
#
# a = CNNBase(env.observation_space.shape[0], recurrent=False)
# actor_critic = Policy(
#     env.observation_space.shape,
#     env.action_space,
#     # (obs_shape[0], ** base_kwargs)
#     base=a,
#     # base_kwargs={'recurrent': args.recurrent_policy}
# )

#actor_critic, ob_rms = torch.load(os.path.join(args.load_dir + "/a2c/" + args.env_name + ".pt"))
# actor_critic, ob_rms = torch.load(os.path.join(args.load_dir , args.env_name + ".pt"))
# actor_critic.to(device)

from a2c_ppo_acktr.cnn import CNNBase
from a2c_ppo_acktr.model import Policy

a = CNNBase(env.observation_space.shape[0], recurrent=False)

actor_critic = Policy(
	env.observation_space.shape,
	env.action_space,
	# (obs_shape[0], ** base_kwargs)
	base=a,
	# base_kwargs={'recurrent': args.recurrent_policy}
)
actor_critic.to(device)
save_model, ob_rms = torch.load('./trained_models/a2c/PongNoFrameskip-v4.pt')
#save_model2, ob_rms2 = torch.load('./trained_models/a2c/DemonAttackNoFrameskip-v4.pt')


actor_critic.load_state_dict(save_model.state_dict())
actor_critic.to(device)
actor_critic.eval()

from a2c_ppo_acktr.storage import RolloutStorage
rollouts = RolloutStorage(5, 16, #step, process
                              env.observation_space.shape, env.action_space,
                              actor_critic.recurrent_hidden_state_size)

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()
rollouts.obs[0].copy_(obs)
rollouts.to(device)

if render_func is not None:
    render_func('human')

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

total_reward = 0
save_state = []
save_action = []
save_value = []
save_probs = []

rollouts_list = []
from collections import deque
episode_rewards = deque(maxlen=10)

import pickle

for i in range(3):
    step = 0
    obs = env.reset()
    for j in range(500):
#    while True:
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states, dist = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)

        # Obser reward and next obs
        obs, reward, done, infos = env.step(action)

        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])

        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

        rollouts.insert(obs, recurrent_hidden_states, action,
                        action_log_prob, value, reward, masks, bad_masks, dist)
        rollouts_list.append(rollouts)
        #print(np.shape(obs)
        #new_action.append(action)
        #print(action)
        # if (step >= 0):
        #     #print("start")
        #     save_state.append(obs)
        #     save_action.append(action)
        #     save_value.append(value)
        #     save_probs.append(dist.probs)

        masks.fill_(0.0 if done else 1.0)
        step += 1

        if args.env_name.find('Bullet') > -1:
            if torsoId > -1:
                distance = 5
                yaw = 0
                humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
                p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

        if render_func is not None:
            render_func('human')

        if done:
            break

import numpy as np

#rollouts_list
#rollouts_list = [t.size() for t in rollouts_list]

torch.save(rollouts_list,"./collect_data/rollouts_list.pt")
rollouts_list = torch.load("./collect_data/rollouts_list.pt")

for i in rollouts_list:
    print(np.shape(i.obs))
    print(np.shape(i.actions))
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