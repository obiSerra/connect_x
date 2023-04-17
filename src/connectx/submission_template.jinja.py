def agent(obs, config):
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

        model = Net()
        model = model.float()
        state_dict = {{state_dict}}

        model.load_state_dict(state_dict)
        model = model.to("cpu")
        model = model.eval()

        obs_reshaped = tensor(obs["board"]).reshape(1, 1, config.rows, config.columns).float()
        board_2d = obs_reshaped.reshape(config.rows, config.columns)

        action = None

        # if obs["step"] < 5:
        #     action = 3

        if action is None:
            action = model({"board": obs_reshaped, "mark": obs["mark"]})
        is_valid = any(board_2d[:, int(action)] == 0)

        if not is_valid:
            logging.info("Not valid action")
            moves = []
            for c in range(1, config.columns):

                if any(board_2d[:, c] == 0):
                    moves.append(c)
            return random.choice(moves)
        return int(action)
    except Exception as e:
        logging.error(e)
