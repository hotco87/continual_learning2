import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
#from a2c_ppo_acktr.distributions import Categorical
from a2c_ppo_acktr.utils import init
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

args = get_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

log_dir = os.path.expanduser(args.log_dir + args.env_name)
eval_log_dir = log_dir + "_eval"
utils.cleanup_log_dir(log_dir)
utils.cleanup_log_dir(eval_log_dir)

torch.set_num_threads(1)
device = torch.device("cuda:0" if args.cuda else "cpu")

# envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
#                          args.gamma, log_dir, device, False)

# envs = make_vec_envs(args.env_name, args.seed, 1,
#                          args.gamma, log_dir, device, False)

import json
file_path = "config.json"
setup_json = json.load(open(file_path, 'r'))
env_conf = setup_json["Default"]
env_conf2 = setup_json["Default"]
for i in setup_json.keys():
	if i in args.env_name:
		env_conf = setup_json[i]
	if i in args.env_name2:
		env_conf2 = setup_json[i]

envs = make_vec_envs(args.env_name, args.seed, 1,
                         args.gamma, args.log_dir, device, env_conf, False)
# 2 game
envs2 = make_vec_envs(args.env_name2, args.seed, 1,
                         args.gamma, args.log_dir2, device, env_conf2, False)

from a2c_ppo_acktr.cnn import CNNBase

a = CNNBase(envs.observation_space.shape[0], recurrent=False)

actor_critic = Policy(
	envs.observation_space.shape,
	envs.action_space,
	# (obs_shape[0], ** base_kwargs)
	base=a,
	# base_kwargs={'recurrent': args.recurrent_policy}
)
actor_critic.to(device)

actor_critic2 = Policy(
	envs2.observation_space.shape,
	envs2.action_space,
	base=a)
# base_kwargs={'recurrent': args.recurrent_policy})
actor_critic2.to(device)

import torch
save_model, ob_rms = torch.load('./trained_models/onegame/a2c/PongNoFrameskip-v4.pt')
save_model2, ob_rms2 = torch.load('./trained_models/onegame/a2c/DemonAttackNoFrameskip-v4.pt')


actor_critic.load_state_dict(save_model.state_dict())
actor_critic.to(device)
actor_critic.eval()

actor_critic2.load_state_dict(save_model2.state_dict())
actor_critic2.to(device)
actor_critic2.eval()

if args.algo == 'a2c':
	agent = algo.A2C_ACKTR(
		actor_critic,
		actor_critic2,
		args.value_loss_coef,
		args.entropy_coef,
		lr=args.lr,
		eps=args.eps,
		alpha=args.alpha,
		max_grad_norm=args.max_grad_norm)

rollouts = RolloutStorage(args.num_steps, args.num_processes,
                          envs.observation_space.shape, envs.action_space,
                          actor_critic.recurrent_hidden_state_size)


for i  in actor_critic.parameters():
	print(type(i))
	#print(i)
print(type(actor_critic.parameters()))

print("##############################")
for name,param in actor_critic.named_parameters():
	print(name)
print("111111111##############################")
a = 0
for name,param in actor_critic2.named_parameters():
	a +=1
	#print(name)
	#"base.main.4.weight"
	actor_critic.register_parameter("base.critic_linear.weight", param)

print("2222111111111##############################")
for name, param in actor_critic.named_parameters():
	print(name)
	#print(type(param.data), param.size())

	#actor_critic.register_parameter(param)

# for i  in actor_critic.parameters():
# 	print(type(i))

	# if name == "base.main.4.weight":
	# #if name =="dist.linear.weight":
	# 	print(param)
	#if name in ["base.main.4.weight"]:
	#	print(param[10])

# print("############################")
# for name,param in actor_critic2.named_parameters():
# 	#if name == "dist.linear.weight":
# 	if name =="base.main.4.weight":
# 		print(param)

# for name,param in actor_critic2.named_parameters():
# 	#if name == "dist.linear.weight":
# 	if name =="base.main.4.weight":
# 		print(param)
	#print(name)
	# if name in ["base.main.4.weight"]:
	# 	print(param[10])

