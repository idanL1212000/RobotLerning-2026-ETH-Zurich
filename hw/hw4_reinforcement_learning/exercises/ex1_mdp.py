import copy
import numpy as np


class PolicyIteration:
    """
    Policy iteration for a finite tabular MDP.

    Attributes:
        env:
            Environment object. We assume:
                - env.n_states: int
                - env.n_actions: int
                - env.P[s][a] = [(prob, next_state, reward, done)]
        theta: float
            Convergence threshold for policy evaluation.
        gamma: float
            Discount factor.
        v: np.ndarray, shape (n_states,)
            State-value function.
        pi: np.ndarray, shape (n_states, n_actions)
            Stochastic policy. Each row pi[s] is a probability distribution
            over actions at state s.
    """

    def __init__(self, env, theta=1e-3, gamma=0.9):
        """Initialize policy iteration."""
        self.env = env
        self.theta = theta
        self.gamma = gamma

        # Initialize value function to zeros.
        self.v = np.zeros(self.env.n_states, dtype=float)

        # Initialize policy to uniform random.
        self.pi = np.ones((self.env.n_states, self.env.n_actions), dtype=float)
        self.pi /= self.env.n_actions

    def policy_evaluation(self):
        """
        Evaluate the current policy until convergence.

        Input:
            Uses:
                - self.pi: np.ndarray, shape (n_states, n_actions)
                - self.v:  np.ndarray, shape (n_states,)
                - self.env.P

        Output:
            Updates:
                - self.v: np.ndarray, shape (n_states,)
        """
        while True:
            max_diff = 0.0
            new_v = np.zeros_like(self.v)

            for s in range(self.env.n_states):
                qsa_list = []
                for a in range(self.env.n_actions):
                    qsa = 0.0

                    # q_pi(s, a) = sum_{s'} p(s', r | s, a) * [r + gamma * v(s')]
                    # Terminal transitions (done) are not bootstrapped.
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        qsa += prob * (
                            reward + self.gamma * self.v[next_state] * (1.0 - done)
                        )

                    # Weight by the probability of taking a under the current policy.
                    qsa_list.append(self.pi[s][a] * qsa)

                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))

            self.v = new_v

            # Stop when the value function has converged.
            if max_diff < self.theta:
                break

    def policy_improvement(self):
        """
        Improve the current policy greedily with respect to the current value function.

        Input:
            Uses:
                - self.v: np.ndarray, shape (n_states,)
                - self.env.P

        Output:
            Updates and returns:
                - self.pi: np.ndarray, shape (n_states, n_actions)

        We assign equal probability to all greedy actions (ties are allowed).
        """
        for s in range(self.env.n_states):
            qsa_list = []
            for a in range(self.env.n_actions):
                qsa = 0.0

                for prob, next_state, reward, done in self.env.P[s][a]:
                    qsa += prob * (
                        reward + self.gamma * self.v[next_state] * (1.0 - done)
                    )

                qsa_list.append(qsa)

            max_q = max(qsa_list)
            num_best_actions = sum(np.isclose(qsa_list, max_q))
            self.pi[s] = [
                1.0 / num_best_actions if np.isclose(q, max_q) else 0.0
                for q in qsa_list
            ]

        return self.pi

    def policy_iteration(self):
        """
        Run policy iteration until the policy no longer changes.

        Input:
            Uses:
                - env.P
                - self.theta
                - self.gamma

        Output:
            v:  np.ndarray, shape (n_states,)
                Final converged value function.
            pi: np.ndarray, shape (n_states, n_actions)
                Final improved policy.

        Main loop:
            1. Policy evaluation
            2. Policy improvement
            3. Stop when the policy is unchanged
        """
        while True:
            old_pi = copy.deepcopy(self.pi)

            # 1. Evaluate the current policy, 2. improve it greedily.
            self.policy_evaluation()
            new_pi = self.policy_improvement()

            if np.allclose(old_pi, new_pi):
                break

        return self.v, self.pi


class ValueIteration:
    """
    Value iteration for a finite tabular MDP.

    Attributes:
        env:
            Environment object with tabular transition model env.P.
        theta: float
            Convergence threshold.
        gamma: float
            Discount factor.
        v: np.ndarray, shape (n_states,)
            State-value function.
        pi: np.ndarray, shape (n_states, n_actions)
            Greedy policy extracted from the converged value function.
    """

    def __init__(self, env, theta=1e-3, gamma=0.9):
        """Initialize value iteration."""
        self.env = env
        self.theta = theta
        self.gamma = gamma

        self.v = np.zeros(self.env.n_states, dtype=float)
        self.pi = np.zeros((self.env.n_states, self.env.n_actions), dtype=float)

    def value_iteration(self):
        """
        Run value iteration until convergence.

        Input:
            Uses:
                - self.v: np.ndarray, shape (n_states,)
                - self.env.P

        Output:
            Returns:
                - v:  np.ndarray, shape (n_states,)
                - pi: np.ndarray, shape (n_states, n_actions)
        """
        while True:
            max_diff = 0.0
            new_v = np.zeros_like(self.v)

            for s in range(self.env.n_states):
                qsa_list = []
                for a in range(self.env.n_actions):
                    qsa = 0.0

                    for prob, next_state, reward, done in self.env.P[s][a]:
                        qsa += prob * (
                            reward + self.gamma * self.v[next_state] * (1.0 - done)
                        )

                    qsa_list.append(qsa)

                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))

            self.v = new_v

            # Stop when the value function has converged.
            if max_diff < self.theta:
                break

        self.get_policy()
        return self.v, self.pi

    def get_policy(self):
        """
        Extract a greedy policy from the converged value function.

        Input:
            Uses:
                - self.v: np.ndarray, shape (n_states,)

        Output:
            Updates:
                - self.pi: np.ndarray, shape (n_states, n_actions)
        """
        for s in range(self.env.n_states):
            qsa_list = []
            for a in range(self.env.n_actions):
                qsa = 0.0

                for prob, next_state, reward, done in self.env.P[s][a]:
                    qsa += prob * (
                        reward + self.gamma * self.v[next_state] * (1.0 - done)
                    )

                qsa_list.append(qsa)

            max_q = max(qsa_list)
            num_best_actions = sum(np.isclose(qsa_list, max_q))
            self.pi[s] = [
                1.0 / num_best_actions if np.isclose(q, max_q) else 0.0
                for q in qsa_list
            ]