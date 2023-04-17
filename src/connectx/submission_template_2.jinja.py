def agent(obs, config):
    from copy import copy
    import numpy as np
    import torch as th
    from torch import nn as nn
    import torch.nn.functional as F
    from torch import tensor
    import random

    import logging

    logging.basicConfig(filename="eval.log", encoding="utf-8", level=logging.DEBUG)

    # (0): Linear(in_features=45, out_features=48, bias=True)
    #   (1): ReLU()
    #   (2): Linear(in_features=48, out_features=48, bias=True)
    #   (3): ReLU()
    #   (4): Linear(in_features=48, out_features=48, bias=True)
    #   (5): ReLU()
    # )
    # (value_net): Sequential(
    #   (0): Linear(in_features=45, out_features=48, bias=True)
    #   (1): ReLU()
    #   (2): Linear(in_features=48, out_features=48, bias=True)
    #   (3): ReLU()
    #   (4): Linear(in_features=48, out_features=48, bias=True)
    #   (5): ReLU()

    try:

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

                # ------- Start model network ----------

                #           (0): Linear(in_features=45, out_features=90, bias=True)
                #   (1): ReLU()
                #   (2): Linear(in_features=90, out_features=180, bias=True)
                #   (3): ReLU()
                #   (4): Linear(in_features=180, out_features=180, bias=True)
                #   (5): ReLU()
                #   (6): Linear(in_features=180, out_features=90, bias=True)
                #   (7): ReLU()
                #   (8): Linear(in_features=90, out_features=45, bias=True)
                #   (9): ReLU()

                self.extractor1 = nn.Flatten(1, -1)
                self.extractor2 = nn.Flatten(1, -1)

                self.policy1 = nn.Linear(45, 90)
                self.policy2 = nn.Linear(90, 90)
                self.policy3 = nn.Linear(90, 90)
                self.action = nn.Linear(90, 7)

                # ------- End model network ----------

            def forward(self, x):
                # ------- Start forward ----------
                x_board = self.extractor1(x["board"])
                mark_array = [0, 1, 0] if x["mark"] == 1 else [0, 0, 1]
                x_mark = th.tensor(mark_array).reshape(1, 3)
                # x = x_board.concat(x["mark"])
                x = th.cat((x_board, x_mark), 1)
                # x = th.FloatTensor(x)
                x = F.relu(self.policy1(x))
                x = F.relu(self.policy2(x))
                x = F.relu(self.policy3(x))
                # ------- end forward ----------
                x = self.action(x)
                x = x.argmax()
                return x

        class ValueNet(nn.Module):
            def __init__(self):
                super(ValueNet, self).__init__()

                # ------- Start model network ----------

                #           (0): Linear(in_features=45, out_features=90, bias=True)
                #   (1): ReLU()
                #   (2): Linear(in_features=90, out_features=180, bias=True)
                #   (3): ReLU()
                #   (4): Linear(in_features=180, out_features=180, bias=True)
                #   (5): ReLU()
                #   (6): Linear(in_features=180, out_features=90, bias=True)
                #   (7): ReLU()
                #   (8): Linear(in_features=90, out_features=45, bias=True)
                #   (9): ReLU()

                self.extractor1 = nn.Flatten(1, -1)
                self.extractor2 = nn.Flatten(1, -1)

                self.value1 = nn.Linear(45, 90)
                self.value2 = nn.Linear(90, 90)
                self.value3 = nn.Linear(90, 90)
                self.value = nn.Linear(90, 1)

                # ------- End model network ----------

            def forward(self, x):
                # ------- Start forward ----------
                x_board = self.extractor1(x["board"])
                mark_array = [0, 1, 0] if x["mark"] == 1 else [0, 0, 1]
                x_mark = th.tensor(mark_array).reshape(1, 3)
                # x = x_board.concat(x["mark"])
                x = th.cat((x_board, x_mark), 1)
                # x = th.FloatTensor(x)
                x = F.relu(self.value1(x))
                x = F.relu(self.value2(x))
                x = F.relu(self.value3(x))
                # ------- end forward ----------
                x = self.value(x)
                return x

        # Policy network
        model = Net()
        model = model.float()
        state_dict = {{state_dict}}

        model.load_state_dict(state_dict)
        model = model.to("cpu")
        model = model.eval()

        # Value network
        model_value = ValueNet()
        model_value = model_value.float()
        value_net_dict = {{value_net_dict}}

        model_value.load_state_dict(value_net_dict)
        model_value = model_value.to("cpu")
        model_value = model_value.eval()

        obs_reshaped = tensor(obs["board"]).reshape(1, 1, config.rows, config.columns).float()
        board_2d = obs_reshaped.reshape(config.rows, config.columns)

        def check_future(start_board, mark=obs["mark"], actions=[], depth=0, max_depth=3):
            if depth == max_depth:
                return actions

            sim_start_board = copy(start_board)

            possible_actions = []
            for i in range(0, config.columns):
                if any(sim_start_board[:, i] == 0):
                    row_index = np.where(sim_start_board[:, i] == 0)[0][-1]
                    sim_board = copy(sim_start_board)
                    sim_board[row_index, i] = mark
                    # logging.info(f"i {i}")
                    # logging.info(sim_board)

                    # value = model_value({"board": sim_board, "mark": obs["mark"]})
                    sim_reshaped = tensor(sim_board).reshape(1, 1, config.rows, config.columns).float()
                    action = model({"board": sim_reshaped, "mark": mark})

                    # logging.info(sim_board.shape)

                    possible_actions.append((action.item(), i, sim_board))
            logging.info(possible_actions)
            possible_actions.sort(key=lambda x: x[0], reverse=True)
            best_action = possible_actions[0]

            return check_future(
                best_action[2], 1 if mark == 2 else 2, actions + [best_action[1]], depth=depth + 1, max_depth=max_depth
            )

            # logging.info(possible_actions)

        action = model({"board": obs_reshaped, "mark": obs["mark"]})
        actions = check_future(board_2d)
        logging.info(actions)
        logging.info(action)
        is_valid = any(board_2d[:, int(action)] == 0)

        if not is_valid:
            logging.info(f"Not valid action")
            moves = []
            for c in range(1, config.columns):
                if any(board_2d[:, c] == 0):
                    moves.append(c)
            return random.choice(moves)
        return int(action)
    except Exception as e:
        logging.error(e)
