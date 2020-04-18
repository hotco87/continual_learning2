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

    log_dir = os.path.expanduser(args.log_dir + args.env_name + "/onegame/" +"/distill/")
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    log_dir2 = os.path.expanduser(args.log_dir2 + args.env_name2 +"onegame")
    eval_log_dir2 = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir2)
    utils.cleanup_log_dir(eval_log_dir2)

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
    envs2 = make_vec_envs(args.env_name2, args.seed, args.num_processes,
                         args.gamma, log_dir2, device, env_conf, False)

    save_model, ob_rms = torch.load('./trained_models/PongNoFrameskip-v4.pt')

    from a2c_ppo_acktr.cnn import CNNBase

    a = CNNBase(envs.observation_space.shape[0], recurrent=False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        #(obs_shape[0], ** base_kwargs)
        base=a,
        #base_kwargs={'recurrent': args.recurrent_policy}
         )
    #actor_critic.load_state_dict(save_model.state_dict())
    actor_critic.to(device)

    actor_critic2 = Policy(
        envs2.observation_space.shape,
        envs2.action_space,
        base=a)
        #base_kwargs={'recurrent': args.recurrent_policy})
    #actor_critic2.load_state_dict(save_model.state_dict())
    actor_critic2.to(device)

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

    rollouts2 = RolloutStorage(args.num_steps, args.num_processes,
                              envs2.observation_space.shape, envs2.action_space,
                              actor_critic2.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    obs2 = envs2.reset()
    rollouts2.obs[0].copy_(obs2)
    rollouts2.to(device)

    episode_rewards = deque(maxlen=10)
    episode_rewards2 = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(num_updates):
        # if args.use_linear_lr_decay:
        #     # decrease learning rate linearly
        #     utils.update_linear_schedule(
        #         agent.optimizer, j, num_updates,
        #         agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, _ = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
                value2, action2, action_log_prob2, recurrent_hidden_states2, _ = actor_critic2.act(
                    rollouts2.obs[step], rollouts2.recurrent_hidden_states[step],
                    rollouts2.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            obs2, reward2, done2, infos2 = envs2.step(action2)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
            for info2 in infos2:
                if 'episode' in info2.keys():
                    episode_rewards2.append(info2['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            masks2 = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done2])
            bad_masks2 = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info2.keys() else [1.0]
                 for info2 in infos2])

            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
            rollouts2.insert(obs2, recurrent_hidden_states2, action2,
                            action_log_prob2, value2, reward2, masks2, bad_masks2)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
            next_value2 = actor_critic2.get_value(
                rollouts2.obs[-1], rollouts2.recurrent_hidden_states[-1],
                rollouts2.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        rollouts2.compute_returns(next_value2, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy, value_loss2, action_loss2, dist_entropy2 = agent.update(rollouts, rollouts2)

        rollouts.after_update()
        rollouts2.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))
            torch.save([
                actor_critic2,
                getattr(utils.get_vec_normalize(envs2), 'ob_rms2', None)
            ], os.path.join(save_path, args.env_name2 + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards2), np.mean(episode_rewards2),
                        np.median(episode_rewards2), np.min(episode_rewards2),
                        np.max(episode_rewards2), dist_entropy2, value_loss2,
                        action_loss2))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

            ob_rms2 = utils.get_vec_normalize(envs2).ob_rms
            evaluate(actor_critic2, ob_rms2, args.env_name2, args.seed,
                     args.num_processes, eval_log_dir2, device)

        # if (args.eval_interval is not None and len(episode_rewards) > 1
        #         and j % args.eval_interval == 0):
        #     ob_rms = utils.get_vec_normalize(envs).ob_rms
        #     evaluate(actor_critic, ob_rms, args.env_name, args.seed,
        #              args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
