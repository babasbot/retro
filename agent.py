from collections import deque, namedtuple
from neural import DDQN
from torch import nn
import random, numpy as np
import torch 
import torch.optim as optim

"""
[B, unassigned, SELECT, START, ↑, ↓, ←, →, A]
"""

action_space = [
    [0, 0, 0, 0, 0, 0, 0, 1, 0], # →
    [0, 0, 0, 0, 0, 0, 0, 1, 1], # → + a
    [1, 0, 0, 0, 0, 0, 0, 1, 0], # → + b
    [1, 0, 0, 0, 0, 0, 0, 1, 1], # → + a + b
]

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayMemory(object):
    def __init__(self, maxlen=100_000):
        self.memory = deque(maxlen=maxlen)
    
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(
            self, 
            state_space, 
            save_dir, 
            replay_memory_len=1_000_000,
            batch_size=32,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.99999975,
            lr=0.00025,
            burnin=100_000,
            learn_every=4,
            sync_every=10_000,
        ):
        self.state_space = state_space  
        self.action_space = action_space 

        self.memory = ReplayMemory(maxlen=replay_memory_len)
        self.batch_size = batch_size 

        # discount rate
        self.gamma = gamma 

        # exploration rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min 
        self.epsilon_decay = epsilon_decay 

        self.lr = lr 

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = DDQN(self.state_space, len(self.action_space), device=self.device)
        self.optimizer = optim.Adam(self.net.online.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = burnin           # experiences before training
        self.learn_every = learn_every # experiences between updates to Q_online
        self.sync_every = sync_every   # experiences between Q_target and Q_online sync

        self.save_every = 5e5
        self.save_dir = save_dir

        self.current_step = 0
        
    def act(self, state):
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(len(self.action_space))
        else:
            action_values = self.net.online(state.unsqueeze(0).to(self.device))
            # action_idx = torch.argmax(action_values, axis=1).item()
            action_idx = torch.argmax(action_values).item()
            
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        self.current_step += 1

        return action_idx

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(
            state.to(self.device).unsqueeze(0),
            torch.tensor([action], dtype=torch.long).to(self.device),
            torch.tensor([reward], dtype=torch.float).to(self.device),
            next_state.to(self.device).unsqueeze(0),
            torch.tensor([done], dtype=torch.bool).to(self.device),
        )

    def learn(self):
        if self.current_step < self.burnin:
            return None, None

        if self.current_step % self.learn_every != 0:
            return None, None

        minibatch = self.memory.sample(self.batch_size) 
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.stack(states).to(self.device).squeeze(1)

        actions = torch.tensor(actions, dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        next_states = torch.stack(next_states).to(self.device).squeeze(1)
        
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        current_q_values = self.net.online(states).gather(1, actions).to(self.device)
            
        next_state_actions = self.net.online(next_states).max(1)[1].view(-1, 1).to(self.device)
        next_q_values = self.net.target(next_states).gather(1, next_state_actions).squeeze().to(self.device)

        td_target = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(current_q_values.squeeze(), td_target)

        self.optimizer.zero_grad()
        
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.online.parameters(), 1.0)
        self.optimizer.step()

        if self.current_step % self.sync_every == 0:
            self.net.target.load_state_dict(self.net.online.state_dict())

        if self.current_step % self.save_every == 0:
            self.save()

        return loss.item(), current_q_values.mean().item()

    def save(self):
        save_path = self.save_dir / f"net_{int(self.current_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.online.state_dict(),
                exploration_rate=self.epsilon
            ),
            save_path
        )

