import time
import numpy as np
from random import choice
from scipy.ndimage import convolve


import logging

LOGGING_FILE = "debug.log"

logging.basicConfig(filename=LOGGING_FILE, encoding="utf-8", level=logging.DEBUG)


class LookAgent:
    def __init__(self) -> None:
        self.predict_max_depth = 2

    def _convert_board(self, board):
        return np.array(board).reshape(self.rows, self.columns)

    def count_in_a_row_old(self, board, to_count=3):
        found = [0, 0]
        return found

    def count_in_a_row(self, board, to_count=3):
        found = [0, 0]
        for r in range(self.rows):
            for c in range(self.columns):
                p_ver, p_hor, p_d1, p_d2 = 0, 0, 0, 0
                if c + (to_count - 1) < self.columns:
                    hor = [board[r][c + cnt] for cnt in range(to_count)]
                    p_hor = np.prod(hor)
                if r + (to_count - 1) < self.rows:
                    ver = [board[r + cnt][c] for cnt in range(to_count)]
                    p_ver = np.prod(ver)
                if r + (to_count - 1) < self.rows and c + (to_count - 1) < self.columns:
                    d_1 = [board[r + cnt][c + cnt] for cnt in range(to_count)]
                    p_d1 = np.prod(d_1)
                if r + (to_count - 1) < self.rows and c + (to_count - 1) >= 0:
                    d_2 = [board[r + cnt][c - cnt] for cnt in range(to_count)]
                    p_d2 = np.prod(d_2)

                complete = pow(2, to_count)
                if p_ver == complete:
                    found[1] += 1
                if p_hor == complete:
                    found[1] += 1
                if p_d1 == complete:
                    found[1] += 1
                if p_d2 == complete:
                    found[1] += 1

                if p_ver == 1:
                    found[0] += 1
                if p_hor == 1:
                    found[0] += 1
                if p_d1 == 1:
                    found[0] += 1
                if p_d2 == 1:
                    found[0] += 1

        return found

    def _simulate_next_moves(self, starting_board, mark, moves=[]):
        boards = []
        for c in range(self.columns):
            if starting_board[0][c] != 0:
                continue
            board = np.copy(starting_board)
            row_index = np.where(board[:, c] == 0)[0][-1]
            board[row_index, c] = mark
            boards.append({"moves": moves + [c], "board": board})
        return boards

    def prune(self, boards):
        hashed_boards = {}
        for board in boards:
            hashed_boards[board["board"].tostring()] = board

        return [v for v in hashed_boards.values()]

    def simulate_step(self, board, mark, moves=[], max_depth=3, depth=0):
        # check the index of the mark in the eval lines
        my_index = self.mark - 1
        oth_index = 1 if my_index == 0 else 0

        in_a_row_4 = self.count_in_a_row(board, to_count=4)
        if sum(in_a_row_4) > 0:
            return [{"moves": moves, "board": board, "final": 1 if sum(in_a_row_4) == 1 else 2}]

        # if max depth is reached, return the board
        if depth >= max_depth:
            return [{"moves": moves, "board": board, "final": 0}]

        # simulate all the next moves
        simulated_steps = self._simulate_next_moves(board, mark, moves)

        nxt = []
        for board in simulated_steps:
            nxt += self.simulate_step(
                board["board"], 2 if mark == 1 else 1, moves=board["moves"], max_depth=max_depth, depth=depth + 1
            )
        nxt = self.prune(nxt)

        final = []
        with_vals = []
        for b in nxt:
            if b["final"]:
                final.append(b)
                continue

            in_a_row_3 = self.count_in_a_row(b["board"], to_count=3)
            with_vals.append({"board": b["board"], "moves": b["moves"], "val": in_a_row_3, "final": 0})

        with_vals.sort(key=lambda x: x["val"][my_index] - x["val"][oth_index], reverse=True)

        return [nxt[0]] + final

    def step(self, obs, config):

        st = time.time()

        self.step_num = obs["step"]
        self.mark = obs["mark"]
        self.rows = config["rows"]
        self.columns = config["columns"]

        self.board = self._convert_board(obs["board"])
        simulated = self.simulate_step(self.board, self.mark, max_depth=self.predict_max_depth)
        # logging.info("simulated_steps")

        winning = [s for s in simulated if s["final"] == self.mark]
        move = None
        try:
            if len(winning) > 0:
                move = winning[0]["moves"][0]
            else:
                move = [s for s in simulated if s["final"] == 0][0]["moves"][0]
        except Exception:
            valid_moves = [b for b in [r for r in range(6)] if obs.board[b] == 0]
            move = choice(valid_moves)
            # logging.info(f"random move {move}")

        # logging.info(f"best move {move}")

        # get the end time
        et = time.time()

        # get the execution time
        elapsed_time = et - st
        logging.info(f"elapsed_time {elapsed_time}")
        return move


def agent_factory(move_predict=1):
    agent = LookAgent()

    def multistep_agent(obs, config):
        try:
            agent.step(obs, config)
            return None
        except Exception as e:
            logging.error(e)

    return multistep_agent
