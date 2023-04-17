import random
import gym
from kaggle_environments import make, evaluate
from gym import spaces
import numpy as np

from connectx.lookahead import multistep_agent_factory


class ConnectFourGymV0(gym.Env):
    def __init__(self, agent2="random"):
        ks_env = make("connectx", debug=True)
        self.env = ks_env.train([None, agent2])
        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns
        # Learn about spaces here: http://gym.openai.com/docs/#spaces
        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(low=0, high=2, shape=(1, self.rows, self.columns), dtype=int)
        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 1)
        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None

    def reset(self):
        self.obs = self.env.reset()
        return np.array(self.obs["board"]).reshape(1, self.rows, self.columns)

    def change_reward(self, old_reward, done):
        if old_reward == 1:  # The agent won the game
            return 1
        elif done:  # The opponent won the game
            return -1
        else:  # Reward 1/42
            return 1 / (self.rows * self.columns)

    def step(self, action):
        # Check if agent's move is valid
        is_valid = self.obs["board"][int(action)] == 0
        if is_valid:  # Play the move
            self.obs, old_reward, done, _ = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else:  # End the game and penalize agent
            print("NOT VALID", action)
            reward, done, _ = -10, True, {}
        return np.array(self.obs["board"]).reshape(1, self.rows, self.columns), reward, done, _


class ConnectFourGymV1(ConnectFourGymV0):
    def __init__(self, agent2="random"):
        super().__init__(agent2)
        self.reward_range = (-10, 3)

    def change_reward(self, old_reward, done):
        if old_reward == 1:  # The agent won the game
            reward = 2 + (1 / self.obs["step"] / (1 / 7))
            return reward
        elif done:  # The opponent won the game
            return -1
        else:  # Reward 1/42
            return 1 / (self.rows * self.columns)


class ConnectFourGymV2(ConnectFourGymV1):
    def __init__(self, agent2="random"):
        ks_env = make("connectx", debug=True)

        # Not random seed to make sure the same game is played
        # ks_env.seed(42)

        self.agents = [None, agent2]
        random.shuffle(self.agents)
        self.env = ks_env.train(self.agents)
        self.rows = ks_env.configuration.rows
        self.columns = ks_env.configuration.columns
        # Learn about spaces here: http://gym.openai.com/docs/#spaces
        self.action_space = spaces.Discrete(self.columns)
        env_spaces = {
            "board": spaces.Box(low=0, high=2, shape=(1, self.rows, self.columns), dtype=int),
            "mark": spaces.Discrete(3),
        }
        self.observation_space = spaces.Dict(env_spaces)
        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 3)
        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None

    def _get_obs(self):
        obs = {
            "board": np.array(self.obs["board"]).reshape(1, self.rows, self.columns),
            "mark": self.obs["mark"],
        }
        # print(obs)
        return obs

    def reset(self):
        self.obs = self.env.reset()
        return self._get_obs()

    def step(self, action):
        # Check if agent's move is valid
        is_valid = self.obs["board"][int(action)] == 0
        if is_valid:  # Play the move
            self.obs, old_reward, done, _ = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else:  # End the game and penalize agent
            reward, done, _ = -10, True, {}

        return self._get_obs(), reward, done, _


class ConnectFourGymV3(ConnectFourGymV2):
    def __init__(self, agent2="random"):
        super().__init__(agent2)
        self.reward_range = (-10, 1)

    def change_reward(self, old_reward, done):
        if old_reward == 1:  # The agent won the game
            # reward = 1 + (1 / self.obs["step"] / (1 / 7))
            return 1
        elif done:  # The opponent won the game
            return -1
        else:  # Reward 1/42
            return 0

    def _get_obs(self):
        obs = {
            "board": np.array(self.obs["board"]).reshape(1, self.rows, self.columns),
            "mark": self.obs["mark"],
        }
        return obs

    def reset(self):
        self.obs = self.env.reset()
        return self._get_obs()

    def step(self, action):
        # Check if agent's move is valid
        is_valid = self.obs["board"][int(action)] == 0
        if is_valid:  # Play the move
            self.obs, old_reward, done, _ = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else:  # End the game and penalize agent
            reward, done, _ = -10, True, {}

        return self._get_obs(), reward, done, _


class ConnectFourGymV4(ConnectFourGymV2):
    def __init__(self, agent2="random"):
        super().__init__(agent2)
        self.reward_range = (-10, 1)

    def change_reward(self, old_reward, done):
        if old_reward == 1:  # The agent won the game
            # reward = 1 + (1 / self.obs["step"] / (1 / 7))
            return 1
        elif done:  # The opponent won the game
            return -1
        else:  # Reward 1/42
            return 1 / (self.rows * self.columns)


class ConnectFourGymV5(ConnectFourGymV2):
    def reset(self):
        ks_env = make("connectx", debug=True)

        # Not random seed to make sure the same game is played
        # ks_env.seed(42)
        random.shuffle(self.agents)
        self.env = ks_env.train(self.agents)

        ret = super().reset()
        return ret


def print_board(obs):
    board = obs["board"]
    mark = obs["mark"]
    print("Mark", mark)
    for i in range(6):
        print(board[i * 7 : (i + 1) * 7])


class ConnectFourGymV6(ConnectFourGymV5):
    def __init__(self, agent2="random"):
        super().__init__(agent2)
        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 2)

    def change_reward(self, old_reward, done):
        if old_reward == 1:  # The agent won the game
            min_win = 6 + (self.obs["mark"] - 1)
            bonus_moves = self.obs["step"] - min_win
            reward = 2 - (bonus_moves / 42)
            # print(f"WON after {self.obs['step']} reward {reward}", bonus_moves, min_win)
            return reward
        elif done:  # The opponent won the game
            # return -1
            min_lost = 6 + (self.obs["mark"] - 1)
            bonus_moves = self.obs["step"] - min_lost
            reward = -1 - (bonus_moves / 42)
            # print(f"LOST after {self.obs['step']} reward {reward}")
            return reward
        else:  # Reward 1/84
            return 1 / (self.rows * self.columns)


class ConnectFourGymV7(ConnectFourGymV5):
    def __init__(self, agent2="random"):
        super().__init__(agent2)
        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 1)

    def change_reward(self, old_reward, done):
        if old_reward == 1:  # The agent won the game
            return 1
        elif done:  # The opponent won the game
            return -5
        else:  # Reward 1/84
            return 1 / (self.rows * self.columns)


class ConnectFourGymV8(ConnectFourGymV5):
    def __init__(self, agent2="random"):
        super().__init__(agent2)
        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 2)

    def change_reward(self, old_reward, done):
        if old_reward == 1:  # The agent won the game
            return 2 - max(1, self.obs["step"] - 7) / 42
        elif done:  # The opponent won the game
            return -2 - max(1, self.obs["step"] - 7) / 42
        else:  # Reward 1/84
            return 1 / (self.rows * self.columns)


class ConnectFourGymV9(ConnectFourGymV5):
    def reset(self):
        ks_env = make("connectx", debug=True)

        available_adv = ["random", multistep_agent_factory(), "negamax"]
        self.agents = [None, random.choice(available_adv)]
        # Not random seed to make sure the same game is played
        # ks_env.seed(42)
        random.shuffle(self.agents)
        self.env = ks_env.train(self.agents)

        ret = super().reset()
        return ret


class ConnectFourGymV10(ConnectFourGymV8):
    def reset(self):
        ks_env = make("connectx", debug=True)

        # available_adv = [multistep_agent_factory(), "negamax"]
        available_adv = ["random", multistep_agent_factory(), "negamax"]
        self.agents = [None, random.choice(available_adv)]
        # Not random seed to make sure the same game is played
        # ks_env.seed(42)
        random.shuffle(self.agents)
        self.env = ks_env.train(self.agents)

        ret = super().reset()
        return ret


class ConnectFourGymV12(ConnectFourGymV8):
    def __init__(self, adv_agents=[]):
        super().__init__("random")
        self.adv_agents = adv_agents

    def reset(self):
        ks_env = make("connectx", debug=True)

        # available_adv = [multistep_agent_factory(), "negamax"]
        self.agents = [None, random.choice(self.adv_agents)]
        # Not random seed to make sure the same game is played
        # ks_env.seed(42)
        random.shuffle(self.agents)
        # print(self.agents)
        self.env = ks_env.train(self.agents)

        ret = super().reset()
        return ret
