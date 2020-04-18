import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs

from a2c_ppo_acktr.model import Policy
#from a2c_ppo_acktr.model import NNBase

from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



    log_dir = os.path.expanduser(args.log_dir + args.env_name + "/onegame/"+ "/distill/")
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    # log_dir2 = os.path.expanduser(args.log_dir2 + args.env_name2 +"onegame")
    # eval_log_dir2 = log_dir + "_eval"
    # utils.cleanup_log_dir(log_dir2)
    # utils.cleanup_log_dir(eval_log_dir2)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    import json
    file_path = "config.json"
    setup_json = json.load(open(file_path, 'r'))
    env_conf = setup_json["Default"]
    for i in setup_json.keys():
        if i in args.env_name:
            env_conf = setup_json[i]

# 1 game
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, log_dir, device, env_conf, False)
# 2 game
#     envs2 = make_vec_envs(args.env_name2, args.seed, args.num_processes,
#                          args.gamma, log_dir2, device, env_conf, False)

    save_model, ob_rms = torch.load('./trained_models/onegame/a2c/PongNoFrameskip-v4.pt')

    from a2c_ppo_acktr.cnn import CNNBase

    a = CNNBase(envs.observation_space.shape[0], recurrent=False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        #(obs_shape[0], ** base_kwargs)
        base=a,
        #base_kwargs={'recurrent': args.recurrent_policy}
         )
    actor_critic.load_state_dict(save_model.state_dict())
    actor_critic.to(device)

    # actor_critic2 = Policy(
    #     envs2.observation_space.shape,
    #     envs2.action_space,
    #     base=a)
        #base_kwargs={'recurrent': args.recurrent_policy})
    #actor_critic2.load_state_dict(save_model.state_dict())
    #actor_critic2.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)

    rollouts_load = torch.load("./collect_data/rollouts_list.pt")
    len_rollouts_load = np.shape(rollouts_load)[0]
    for i in rollouts_load:
        i.to(device)

    # rollouts2 = RolloutStorage(args.num_steps, args.num_processes,
    #                           envs2.observation_space.shape, envs2.action_space,
    #                           actor_critic2.recurrent_hidden_state_size)

    #obs = envs.reset()
    #rollouts_load.obs[0].copy_(obs)
    #rollouts_load.to(device)

    # obs2 = envs2.reset()
    # rollouts2.obs[0].copy_(obs2)
    # rollouts2.to(device)

    #episode_rewards = deque(maxlen=10)
    #episode_rewards2 = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(num_updates):
        #for step in range(args.num_steps):
            # with torch.no_grad():
            #     value, action, action_log_prob, recurrent_hidden_states, dist = actor_critic.act(
            #         rollouts_load[j%len_rollouts_load].obs[step], rollouts_load[j%len_rollouts_load].recurrent_hidden_states[step],
            #         rollouts_load[j%len_rollouts_load].masks[step])
        action_loss = agent.update(rollouts_load[j % len_rollouts_load])
        #print(j)
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, "onegame","distill",args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and j!=0 :#and len(episode_rewards) > 1:
            total_reward = 0
            recurrent_hidden_states1 = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
            env = make_vec_envs(args.env_name,args.seed + 1000,1,None,None,device,env_conf,allow_early_resets=False)
            obs = env.reset()
            #print("j: ",j)
            step = 0
            masks1 = torch.zeros(1, 1)
            while True:
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states1, dist = actor_critic.act(
                        obs, recurrent_hidden_states1, masks1, deterministic=True)
                obs, reward, done, infos = env.step(action)
                total_reward += reward
                step += 1
                if done.any():
                     break

            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes reward \n action_loss {}"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        total_reward.mean(), action_loss))
            # print(
            #     "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
            #     .format(j, total_num_steps,
            #             int(total_num_steps / (end - start)),
            #             len(episode_rewards2), np.mean(episode_rewards2),
            #             np.median(episode_rewards2), np.min(episode_rewards2),
            #             np.max(episode_rewards2), dist_entropy2, value_loss2,
            #             action_loss2))

        if (args.eval_interval is not None  #and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

            # ob_rms2 = utils.get_vec_normalize(envs2).ob_rms
            # evaluate(actor_critic2, ob_rms2, args.env_name2, args.seed,
            #          args.num_processes, eval_log_dir2, device)

        # if (args.eval_interval is not None and len(episode_rewards) > 1
        #         and j % args.eval_interval == 0):
        #     ob_rms = utils.get_vec_normalize(envs).ob_rms
        #     evaluate(actor_critic, ob_rms, args.env_name, args.seed,
        #              args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
