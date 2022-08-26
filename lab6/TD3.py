'''DLP DDPG Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class GaussianNoise:
    def __init__(self, dim, mu=None, std=None):
        self.mu = mu if mu else np.zeros(dim)
        self.std = std if std else np.ones(dim) * .1

    def sample(self):
        return np.random.normal(self.mu, self.std)


class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        ## TODO ##
        transitions = random.sample(self.buffer,batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device) 
                for x in zip(*transitions))
        raise NotImplementedError


class ActorNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super().__init__()
        ## TODO ##
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], action_dim),
            nn.Tanh()
        )
        # raise NotImplementedError

    def forward(self, x):
        ## TODO ##
        return self.net(x)
        raise NotImplementedError


class CriticNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super().__init__()
        h1, h2 = hidden_dim
        self.critic_head = nn.Sequential(
            nn.Linear(state_dim + action_dim, h1),
            nn.ReLU(),
        )
        self.critic = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x, action):
        x = self.critic_head(torch.cat([x, action], dim=1))
        return self.critic(x)


class TD3:
    def __init__(self, args,max_action):
        ## behavior network
        self._actor_net = ActorNet().to(args.device)
        self._critic_net1 = CriticNet().to(args.device)
        self._critic_net2 = CriticNet().to(args.device)
        ## target network
        self._target_actor_net = ActorNet().to(args.device)
        self._target_critic_net1 = CriticNet().to(args.device)
        self._target_critic_net2 = CriticNet().to(args.device)
        ## initialize target network
        self._target_actor_net.load_state_dict(self._actor_net.state_dict())
        self._target_critic_net1.load_state_dict(self._critic_net1.state_dict())
        self._target_critic_net2.load_state_dict(self._critic_net2.state_dict())
        ## TODO ##
        self._actor_opt = torch.optim.Adam(self._actor_net.parameters(), lr=args.lra)
        self._critic_opt1 = torch.optim.Adam(self._critic_net1.parameters(), lr=args.lrc)
        self._critic_opt2 = torch.optim.Adam(self._critic_net2.parameters(), lr=args.lrc)
        # raise NotImplementedError
        # action noise
        self._action_noise = GaussianNoise(dim=2)
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.gamma = args.gamma

        self.max_action = max_action

    def select_action(self, state, noise=True):
        '''based on the behavior (actor) network and exploration noise'''
        ## TODO ##
        if noise:
            action_noise = self._action_noise.sample()
        else:
            action_noise = np.zeros(self._action_noise.sample().shape)
        action_noise = torch.from_numpy(action_noise).to(self.device)
        with torch.no_grad():
            now_state = torch.from_numpy(state).to(self.device)
            action = self._actor_net(now_state) + action_noise
        return action.cpu().numpy()
        raise NotImplementedError

    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, action, [reward / 100], next_state,
                            [int(done)])

    def update(self,args,epoch):
        # update the behavior networks
        self._update_behavior_network(self.gamma,args,epoch)
        # update the target networks
        self._update_target_network(self._target_actor_net, self._actor_net,
                                    self.tau)
        self._update_target_network(self._target_critic_net1, self._critic_net1,
                                    self.tau)
        self._update_target_network(self._target_critic_net2, self._critic_net2,
                                    self.tau)

    def _update_behavior_network(self, gamma,args,epoch):
        actor_net, critic_net1, critic_net2= self._actor_net, self._critic_net1,self._critic_net2
        actor_opt, critic_opt1, critic_opt2 = self._actor_opt, self._critic_opt1, self._critic_opt2

        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(self.batch_size, self.device)

        ## update critic ##
        # critic loss
        ## TODO ##
        q_value1 = self._critic_net1(state, action)
        q_value2 = self._critic_net2(state, action)
        with torch.no_grad():
            noise = torch.ones_like(action).data.normal_(0,args.policy_noise).to(self.device).clamp(-args.noise_clip, args.noise_clip)
            a_next = (self._target_actor_net(next_state) + noise).clamp(-self.max_action,self.max_action)
            q_next = min(self._target_critic_net1(next_state, a_next), self._target_critic_net2(next_state,a_next))
            q_target = reward + gamma * q_next * (1 - done)
        criterion = nn.MSELoss()
        critic_loss1 = criterion(q_value1, q_target)
        critic_loss2 = criterion(q_value2, q_target)
        # raise NotImplementedError
        # optimize critic
        actor_net.zero_grad()
        critic_net1.zero_grad()
        critic_net2.zero_grad()
        critic_loss1.backward()
        critic_loss2.backward()
        critic_opt1.step()
        critic_opt2.step()

        ## update actor ##
        # actor loss
        ## TODO ##
        if epoch % args.policy_delay:
            action = self._actor_net(state)
            actor_loss = -self._critic_net1(state, action).mean()
            # raise NotImplementedError
            # optimize actor
            actor_net.zero_grad()
            critic_net1.zero_grad()
            critic_net2.zero_grad()
            actor_loss.backward()
            actor_opt.step()

    @staticmethod
    def _update_target_network(target_net, net, tau):
        '''update target network by _soft_ copying from behavior network'''
        for target, behavior in zip(target_net.parameters(), net.parameters()):
            ## TODO ##
            with torch.no_grad():
                target.copy_(target * (1.0 - tau) + behavior * tau)
            # raise NotImplementedError

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic1': self._critic_net1.state_dict(),
                    'critic2': self._critic_net2.state_dict(),
                    'target_actor': self._target_actor_net.state_dict(),
                    'target_critic1': self._target_critic_net1.state_dict(),
                    'target_critic2': self._target_critic_net2.state_dict(),
                    'actor_opt': self._actor_opt.state_dict(),
                    'critic_opt1': self._critic_opt1.state_dict(),
                    'critic_opt2': self._critic_opt2.state_dict(),
                }, model_path)
        else:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic1': self._critic_net1.state_dict(),
                    'critic2': self._critic_net2.state_dict(),
                }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._actor_net.load_state_dict(model['actor'])
        self._critic_net1.load_state_dict(model['critic1'])
        self._critic_net2.load_state_dict(model['critic2'])
        if checkpoint:
            self._target_actor_net.load_state_dict(model['target_actor'])
            self._target_critic_net1.load_state_dict(model['target_critic1'])
            self._target_critic_net2.load_state_dict(model['target_critic2'])
            self._actor_opt.load_state_dict(model['actor_opt'])
            self._critic_opt1.load_state_dict(model['critic_opt1'])
            self._critic_opt2.load_state_dict(model['critic_opt2'])


def train(args, env, agent, writer):
    print('Start Training')
    total_steps = 0
    ewma_reward = 0
    best_rewards = 8.5
    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        for t in itertools.count(start=1):
            # select action
            if total_steps < args.warmup:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update(args,episode)

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                # ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  episode)
                # writer.add_scalar('Train/Ewma Reward', ewma_reward,
                #                   total_steps)
                # print(
                #     'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}'
                #     .format(total_steps, episode, t, total_reward,
                #             ewma_reward))
                break
        if episode % 10 == 0:
            testing_rewards = test(args,env,agent,writer)
            print("episode:{}, avg_rewards:{}".format(episode, testing_rewards))
            if testing_rewards > best_rewards:
                best_rewards = testing_rewards
                print("get a better rewards:{}".format(testing_rewards))
                path = "./lab6/TD3/TD3_R_{}_LR_{}_{}_Batch_{}_G_{}.pth".format(testing_rewards,args.lra,args.lrc,args.batch_size,args.gamma)
                agent.save(path)
    env.close()


def test(args, env, agent, writer):
    print('Start Testing')
    seeds = (args.seed + i for i in range(10))
    # seeds = [0,18,21,23,24,30,35,44,46,55]
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        # env.seed(seed)
        state = env.reset(seed = seed)
        ## TODO ##
        for t in itertools.count(start=1):
            if args.render:
                env.render()

            action = agent.select_action(state,noise = False)
            next_state, reward, done, _ = env.step(action)

            state = next_state
            total_reward += reward

        # ...
            if done:
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                if args.test_only:
                    print('Episode: {}\tTotal reward: {:.2f}'.format(n_episode, total_reward))
                # if total_reward > 290:
                #     print("seed = {}".format(seed))
                rewards.append(total_reward)
                break;
        #         ...
        # raise NotImplementedError
    if args.test_only:
        print('Average Reward', np.mean(rewards))
    env.close()
    return np.mean(rewards) / 30.0


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cpu')
    parser.add_argument('-m', '--model', default='./lab6/TD3/TD3.pth')
    parser.add_argument('--logdir', default='./lab6/TD3')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=1200, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--capacity', default=500000, type=int)
    parser.add_argument('--lra', default=1e-3, type=float)
    parser.add_argument('--lrc', default=1e-3, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--tau', default=.005, type=float)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)
    parser.add_argument('--policy_delay', default=2, type=int)
    parser.add_argument('--policy_noise', default=0.2, type=float)
    parser.add_argument('--noise_clip', default=0.5, type=float)
    args = parser.parse_args()

    ## main ##
    env = gym.make('LunarLanderContinuous-v2')
    max_action = float(env.action_space.high[0])
    agent = TD3(args,max_action)
    writer = SummaryWriter(args.logdir)
    # writer = 1
    if not args.test_only:
        train(args, env, agent, writer)
        # agent.save(args.model)
    # agent.load(args.model)
    avg_rewards = test(args, env, agent, writer)
    if args.test_only:
        print("avg rewards:{}".format(avg_rewards))



if __name__ == '__main__':
    main()
