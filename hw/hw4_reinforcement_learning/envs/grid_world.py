class CliffWalkingEnv:
    """
    Stochastic Cliff Walking gridworld.

    State:
        state = row * ncol + col

    Actions:
        0: up, 1: down, 2: left, 3: right

    Transition model:
        P[s][a] = [(prob, next_state, reward, done), ...]

    With probability (1 - slip_chance), the intended action is taken.
    With probability slip_chance, the agent slips to one of the other
    actions uniformly at random.
    """

    def __init__(self, ncol: int = 12, nrow: int = 4, slip_chance: float = 0.01):
        self.ncol = ncol
        self.nrow = nrow
        self.slip_chance = slip_chance

        # Number of states and actions
        self.n_states = ncol * nrow
        self.n_actions = 4

        # Action symbols
        self.action_meaning = ["^", "v", "<", ">"]

        # Special states
        self.start_state = (nrow - 1) * ncol
        self.goal_state = nrow * ncol - 1
        self.cliff_states = list(range((nrow - 1) * ncol + 1, nrow * ncol - 1))

        # Transition model
        self.P = self._build_transition_model(slip_chance=self.slip_chance)

    def _build_transition_model(self, slip_chance: float = 0.01):
        """
        Build the transition model P.

        Returns:
            P[s][a]: list of possible outcomes after taking action a in state s.
        """
        P = [[[] for _ in range(self.n_actions)] for _ in range(self.n_states)]

        # (dx, dy) for [up, down, left, right]
        moves = [
            (0, -1),
            (0, 1),
            (-1, 0),
            (1, 0),
        ]

        for row in range(self.nrow):
            for col in range(self.ncol):
                state = row * self.ncol + col

                for action in range(self.n_actions):
                    # Terminal states stay unchanged
                    if state in self.cliff_states or state == self.goal_state:
                        P[state][action] = [(1.0, state, 0.0, True)]
                        continue

                    for executed_action in range(self.n_actions):
                        dx, dy = moves[executed_action]

                        # Next position with boundary clipping
                        next_col = min(self.ncol - 1, max(0, col + dx))
                        next_row = min(self.nrow - 1, max(0, row + dy))
                        next_state = next_row * self.ncol + next_col

                        reward = -1.0
                        done = False

                        if next_state in self.cliff_states:
                            reward = -100.0
                            done = True
                        elif next_state == self.goal_state:
                            done = True

                        if executed_action == action:
                            prob = 1 - slip_chance
                        else:
                            prob = slip_chance / (self.n_actions - 1)

                        P[state][action].append((prob, next_state, reward, done))

        return P

    def state_to_pos(self, state: int):
        """Convert state index to (row, col)."""
        row = state // self.ncol
        col = state % self.ncol
        return row, col

    def pos_to_state(self, row: int, col: int):
        """Convert (row, col) to state index."""
        return row * self.ncol + col