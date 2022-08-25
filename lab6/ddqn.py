'''DLP DQN Lab'''
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
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device) for x in zip(*transitions))


class Net(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=(400, 300)):
        super().__init__()
        ## TODO ##
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], action_dim),
        )

        # raise NotImplementedError

    def forward(self, x):
        ## TODO ##
        return self.net(x)
        raise NotImplementedError


class DQN:
    def __init__(self, args):
        
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)
        '''
        behavior_net use to select action while training
        target_net use to hold the target, cuz behavior_net changes frequently
        but it's not what we expected since frequently change target will also affect our performance
        target_net得到的Q會是我們更新用的指標，behavior net是我們主要更新的對象，也是決定action的對象
        如果沒有target_net 單純只用behavior net的話，頻繁更新behavior net會使得我們作為指標的Q不斷更新，造成性能受影響
        '''


        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        ## TODO ##
        # self._behavior_net = nn.DataParallel(self._behavior_net)
        # self._target_net = nn.DataParallel(self._target_net)
        self._optimizer = torch.optim.Adam(self._behavior_net.parameters(),args.lr)
        # self._optimizer = nn.DataParallel(self._optimizer).module
        # raise NotImplementedError
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
         ## TODO ##

        # randomly act in the environment with probability epsilon
        if random.random() <= epsilon:
            return action_space.sample()
        else:
            with torch.no_grad():
                status_quo = torch.from_numpy(state).to(self.device)
                action = self._behavior_net(status_quo)
                return action.argmax().item()
        # raise NotImplementedError

    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, [action], [reward / 10], next_state,[int(done)])

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(self.batch_size, self.device)

        ## TODO ##
        # get q values of some randomly picked actions
        q_value = torch.gather(self._behavior_net(state), dim=1, index=action.long())
        with torch.no_grad():
            next_action = torch.max(self._behavior_net(next_state),1)[1].view(-1,1)
            # predict q value of next state
            q_next = self._target_net(next_state).gather(1,next_action.long())
            # compute q value of next state with done flag
            q_target = reward + gamma * q_next * (1 - done)
        '''
        之前在dqn當中我們是前往target_net當中找出能得到最大Q value的actoin
        但這有可能會有過於樂觀的可能,因為target_net跟behavior_net不相同
        所以有可能我們挑出來的這個最大值，根本不會是behavior_net可能採取的動作
        因此在ddqn當中，我們是從behavior_net中挑選action，再去target_net計算理論上的Q value
        '''
        
        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target)
        # raise NotImplementedError
        # optimize
        self._optimizer.zero_grad()
        loss.backward()
        # use gradient clipping to avoid gradient explosion
        # loss where gradient is too large will be clipped to a maximum value(5)
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()

    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        ## TODO ##
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        # raise NotImplementedError

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'behavior_net': self._behavior_net.state_dict(),
                    'target_net': self._target_net.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])


def train(args, env, agent, writer):
    print('Start Training')
    action_space = env.action_space
    total_steps, epsilon = 0, 1.
    ewma_reward = 0
    best_rewards = 8.5
    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        for t in itertools.count(start=1):
            # select action
            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                action = agent.select_action(state, epsilon, action_space)
                epsilon = max(epsilon * args.eps_decay, args.eps_min)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update(total_steps)

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                #ewma = Exponentially Weighted Moving-Average
                # ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                # writer.add_scalar('Train/Episode Reward', total_reward,
                #                   episode)
                # writer.add_scalar('Train/Ewma Reward', ewma_reward,
                #                   total_steps)
                # print(
                #     'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                #     .format(total_steps, episode, t, total_reward, ewma_reward,
                #             epsilon))
                break
        if episode % 10 == 0:
            testing_rewards = test(args,env,agent,writer)
            print("episode:{}, avg_rewards:{}".format(episode, testing_rewards))
            if testing_rewards > best_rewards:
                best_rewards = testing_rewards
                print("get a better rewards:{}".format(testing_rewards))
                path = "./lab6/ddqn/dqn_R_{}_LR_{}_Batch_{}_G_{}.pth".format(testing_rewards,args.lr,args.batch_size,args.gamma)
                agent.save(path)
    env.close()


def test(args, env, agent, writer):
    print('Start Testing')
    action_space = env.action_space
    epsilon = args.test_epsilon
    seeds = [17, 18, 21, 24, 26, 44, 65, 68, 69, 75]
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()
        ## TODO ##
        for t in itertools.count(start=1):
            if args.render:
                env.render()
            # select action
            action = agent.select_action(state, epsilon, action_space)

            # execute action
            next_state, reward, done, _ = env.step(action)

            state = next_state
            total_reward += reward
        # ...
            if done:
                # writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                print('Episode: {}\tTotal reward: {:.2f}'.format(n_episode, total_reward))
                rewards.append(total_reward)
                break
        #         ...
        # raise NotImplementedError
    avg_rewards = np.mean(rewards) / 30.0
    print('Average Reward', np.mean(rewards))
    env.close()
    return avg_rewards


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='./lab6/ddqn/ddqn.pth')
    parser.add_argument('--logdir', default='./lab6/ddqn')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=1200, type=int)
    parser.add_argument('--capacity', default=10000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=.0005, type=float)
    parser.add_argument('--eps_decay', default=.995, type=float)
    parser.add_argument('--eps_min', default=.01, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=1000, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)
    parser.add_argument('--test_epsilon', default=.001, type=float)
    args = parser.parse_args()

    ## main ##
    env = gym.make('LunarLander-v2')
    agent = DQN(args)
    # writer = SummaryWriter(args.logdir)
    writer = 1
    if not args.test_only:
        train(args, env, agent, writer)
        # agent.save(args.model)
    agent.load(args.model)
    avg_rewards = test(args, env, agent, writer)
    print("avg rewards:{}".format(avg_rewards))


if __name__ == '__main__':
    main()
