import random
import collections

import numpy as np
import torch
import torch.nn.functional as F


class ReplayBuffer:
    """
    Experience replay buffer for off-policy reinforcement learning.

    The buffer stores transitions of the form:
        (state, action, reward, next_state, done)

    During training, we sample random mini-batches from the buffer to break
    temporal correlations between consecutive transitions.
    """

    def __init__(self, capacity):
        """
        Initialize the replay buffer.

        Args:
            capacity (int): maximum number of transitions to store
        """
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Store one transition in the replay buffer.

        Args:
            state (np.ndarray): current state
            action (int): action taken at the current state
            reward (float): reward received after taking the action
            next_state (np.ndarray): next state
            done (bool): whether the episode terminates after this transition
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a random mini-batch of transitions.

        Args:
            batch_size (int): number of transitions to sample

        Returns:
            states (np.ndarray): shape (batch_size, state_dim)
            actions (tuple): length batch_size
            rewards (tuple): length batch_size
            next_states (np.ndarray): shape (batch_size, state_dim)
            dones (tuple): length batch_size
        """
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        return (
            np.array(states, dtype=np.float32),
            actions,
            rewards,
            np.array(next_states, dtype=np.float32),
            dones,
        )

    def size(self):
        """
        Return the current number of stored transitions.
        """
        return len(self.buffer)


class QNet(torch.nn.Module):
    """
    Q-network with one hidden layer.

    Input:
        state

    Output:
        Q-values for all discrete actions
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        """
        Initialize the Q-network.

        Args:
            state_dim (int): dimension of the state space
            hidden_dim (int): number of hidden units
            action_dim (int): number of discrete actions
        """
        super(QNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        Forward pass of the Q-network.

        Args:
            x (torch.Tensor): shape (batch_size, state_dim)

        Returns:
            torch.Tensor: Q-values for all actions, shape (batch_size, action_dim)
        """
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN:
    """
    Deep Q-Network (DQN) for discrete action spaces.
    """

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        """
        Initialize the DQN agent.

        Args:
            state_dim (int): dimension of the state space
            hidden_dim (int): hidden dimension of the Q-network
            action_dim (int): number of discrete actions
            learning_rate (float): learning rate for Adam
            gamma (float): discount factor
            epsilon (float): exploration probability in epsilon-greedy policy
            target_update (int): update frequency of the target network
            device (torch.device): cpu or cuda
        """
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device

        # Online Q-network
        self.q_net = QNet(state_dim, hidden_dim, action_dim).to(device)

        # Target Q-network
        self.target_q_net = QNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)

        # Counter used for periodic target network updates
        self.count = 0

    def take_action(self, state):
        """
        Select an action using an epsilon-greedy policy.

        With probability epsilon, choose a random action.
        Otherwise, choose the action with the highest predicted Q-value.

        Args:
            state (np.ndarray): current state, shape (state_dim,)

        Returns:
            int: selected action
        """
        # Explore: uniformly random action.
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_dim))

        # Exploit: greedy action w.r.t. the online Q-network.
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.q_net(state).argmax().item()
        return int(action)

    def predict_action(self, state):
        """
        Select the greedy action.

        This is useful during evaluation.

        Args:
            state (np.ndarray): current state

        Returns:
            int: greedy action
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = self.q_net(state).argmax().item()
        return int(action)

    def update(self, transition_dict):
        """
        Update the online Q-network using one mini-batch of transitions.

        Args:
            transition_dict (dict): contains
                - 'states'
                - 'actions'
                - 'rewards'
                - 'next_states'
                - 'dones'
        """
        states = torch.tensor(
            transition_dict["states"], dtype=torch.float32
        ).to(self.device)
        actions = torch.tensor(
            transition_dict["actions"], dtype=torch.long
        ).view(-1, 1).to(self.device)
        rewards = torch.tensor(
            transition_dict["rewards"], dtype=torch.float32
        ).view(-1, 1).to(self.device)
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float32
        ).to(self.device)
        dones = torch.tensor(
            transition_dict["dones"], dtype=torch.float32
        ).view(-1, 1).to(self.device)

        # Compute current Q values
        q_values = self.q_net(states).gather(1, actions)

        # Compute TD target
        with torch.no_grad():
            # Greedy value of the next state under the target network.
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            # DQN target: r + gamma * max_a' Q_target(s', a') * (1 - done)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Compute DQN loss
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))

        # Optimize the Q-network
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        # Periodically update the target network
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1

    def save(self, path):
        """
        Save model parameters.
        """
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_q_net": self.target_q_net.state_dict(),
            },
            path,
        )

    def load(self, path):
        """
        Load model parameters.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_q_net.load_state_dict(checkpoint["target_q_net"])