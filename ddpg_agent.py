from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import random

from model import Actor, Critic

import random


BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-4
WEIGHT_DECAY = 0 # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment"""

    def __init__(self,state_size,action_size,random_seed, n_agents):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """

        self.seed = random.seed(random_seed)
        # Actor Network w/ Target Network
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),lr = LR_ACTOR)

        # Critic Network w/ Target Network
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Init action noise object
        self.noise = OUNoise((n_agents, action_size), random_seed)

        # Replay buffer
        self.memory = ReplayBuffer( BUFFER_SIZE, BATCH_SIZE ,random_seed)

    def act(self,state, add_noise = True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise == True:
            action += self.noise.sample()
        return action

    def reset(self):
        self.noise.reset()

    def step(self,state,action,reward,next_state,done):
        for s,a,r,ns,d in zip(state, action, reward, next_state, done):
            self.memory.add(s,a,r,ns,d)
            
        if len(self.memory) > BATCH_SIZE:
#             for i in range(20):
            states,actions,rewards,next_states,dones  = self.memory.sample()
            self.learn(states,actions,rewards,next_states,dones)

    def learn(self,states,actions,rewards,next_states,dones):
        # Update Critic
        next_actions = self.actor_target(next_states)
        Q_targets_next = GAMMA * self.critic_target(next_states, next_actions)
        Q_targets = rewards + Q_targets_next * (1 - dones)
        Q_expects = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expects, Q_targets)
            # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actions_pred = self.actor_local(states)
        performance = -self.critic_local(states, actions_pred).mean()
            # Maximize the performance
        self.actor_optimizer.zero_grad()
        performance.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self.soft_update(local_model = self.actor_local, target_model = self.actor_target,  tau= TAU)
        self.soft_update(local_model = self.critic_local, target_model = self.critic_target, tau= TAU)
    
    def soft_update(self, local_model, target_model, tau):
        # https://pytorch.org/docs/stable/nn.html#torch.nn.Module.parameters
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_((1.0-tau) * target_param.data + tau * local_param.data)


class ReplayBuffer():
    def __init__(self, buffer_size, batch_size, seed ):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state,action,reward,next_state,done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k = self.batch_size)
        # stack the batch samples, convert into tensor float type, and use cpu or gpu.
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.size = size
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        # In here the state means the action noise.
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
#         dx = self.theta * (self.mu - x) + self.sigma * np.random.random(self.size)
        self.state = x + dx
        return self.state