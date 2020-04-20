import torch
import torch.nn as nn
import torch.optim as optim

from a2c_ppo_acktr.algo.kfac import KFACOptimizer


class A2C_ACKTR():
    def __init__(self,
                 saved_actor_critic,
                 actor_critic,
                 actor_critic2,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False):

        self.saved_actor_critic = saved_actor_critic
        self.actor_critic = actor_critic
        self.actor_critic2 = actor_critic2
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            self.optimizer = optim.RMSprop(
                actor_critic.parameters(), lr, eps=eps, alpha=alpha)
            self.optimizer2 = optim.RMSprop(
                actor_critic2.parameters(), lr, eps=eps, alpha=alpha)

    def update(self, saved_rollouts ,rollouts, rollouts2):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        obs_shape2 = rollouts2.obs.size()[2:]
        action_shape2 = rollouts2.actions.size()[-1]
        num_steps2, num_processes2, _ = rollouts2.rewards.size()

        saved_values, saved_action_log_probs, saved_dist_entropy, _, saved_dist = self.saved_actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values, action_log_probs, dist_entropy, _,  dist = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values2, action_log_probs2, dist_entropy2, _, dist2 = self.actor_critic2.evaluate_actions(
            rollouts2.obs[:-1].view(-1, *obs_shape2),
            rollouts2.recurrent_hidden_states[0].view(
                -1, self.actor_critic2.recurrent_hidden_state_size),
            rollouts2.masks[:-1].view(-1, 1),
            rollouts2.actions.view(-1, action_shape2))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        ###########################################################
        values2 = values2.view(num_steps2, num_processes2, 1)
        action_log_probs2 = action_log_probs2.view(num_steps2, num_processes2, 1)

        advantages2 = rollouts2.returns[:-1] - values2
        value_loss2 = advantages2.pow(2).mean()

        action_loss2 = -(advantages2.detach() * action_log_probs2).mean()

        ###################################################################################################
        ### add ###
        ################ teacher ####################
        action_t = saved_dist.logits
        action_t = action_t.view(num_steps, num_processes, 6)  # student
        ################ student ####################
        action_s = dist.logits#sample()
        action_s = action_s.view(num_steps, num_processes, 6) # student
        ###################################################################################################
        import torch.nn.functional as F
        temperature = 2
        distillation_loss = nn.KLDivLoss(reduction='batchmean')(input=F.log_softmax(action_s / temperature, dim=-1),
                                                                target=F.softmax(action_t / temperature, dim=-1))
        saved_action_loss = distillation_loss.mean()

        saved_values = saved_values.view(num_steps, num_processes, 1)
        loss_fn = torch.nn.MSELoss()
        saved_value_loss = loss_fn(saved_values, values).mean() # preds, target

        #advantages = rollouts.returns[:-1] - values
        #value_loss = advantages.pow(2).mean()
        #action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        #action_loss = -(advantages.detach() * action_log_probs).mean()

        #print("value_loss2 :", value_loss2)
        #print("advantages2 ", advantages2)
        ###########################################################

        # print("action_loss: ",action_loss)
        # (value_loss * self.value_loss_coef + action_loss -
        #  dist_entropy * self.entropy_coef).backward()
        # print(value_loss2 * self.value_loss_coef + action_loss2 -
        #  dist_entropy2 * self.entropy_coef)
        self.optimizer.zero_grad()
        self.optimizer2.zero_grad()


        # (value_loss2 * self.value_loss_coef + action_loss2 -
        #   dist_entropy2 * self.entropy_coef +
        #  value_loss + action_loss - dist_entropy * self.entropy_coef).backward()

        # (value_loss2 * self.value_loss_coef + action_loss2 -
        #  dist_entropy2 * self.entropy_coef +

        # (value_loss2 * self.value_loss_coef + action_loss2 - dist_entropy * self.entropy_coef).backward()
        # (value_loss* self.value_loss_coef+ action_loss - dist_entropy * self.entropy_coef + saved_value_loss*0.005 + saved_action_loss*0.01).backward()
        (value_loss2 * self.value_loss_coef + action_loss2 - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef + saved_value_loss * 0.005 + saved_action_loss * 0.01).backward()


        #action_loss.backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.actor_critic2.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()
        self.optimizer2.step()



        return value_loss.item(), action_loss.item(), dist_entropy.item(),\
               value_loss2.item(), action_loss2.item(), dist_entropy2.item()

        # return value_loss.item(), action_loss.item(), dist_entropy.item(), \
        #        value_loss2.item(), action_loss2.item(), dist_entropy2.item()
