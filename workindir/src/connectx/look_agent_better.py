import time
import numpy as np
from random import choice
from scipy.ndimage import convolve


import logging

LOGGING_FILE = "debug.log"

# Create a custom logger
logger = logging.getLogger(__name__)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(LOGGING_FILE)

# Create formatters and add it to handlers
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)


class ConnectFourAgent:
    def __init__(self, n=3):
        self.n = n  # depth of search
        # Predefined column order for exploration (center first)
        self.column_order = [3, 2, 4, 1, 5, 0, 6]

    def _convert_board(self, board):
        return np.array(board).reshape(self.rows, self.columns)

    def step(self, obs, config):
        self.step_num = obs["step"]
        self.rows = config["rows"]
        self.columns = config["columns"]

        board = self._convert_board(obs["board"])
        self.player_id = obs["mark"]
        self.opponent_id = 2 if self.player_id == 1 else 1
        valid_columns = self.get_valid_columns(board)
        best_score = -float("inf")
        best_col = valid_columns[0]  # default to first valid column
        alpha = -float("inf")
        beta = float("inf")
        for col in self.order_columns(valid_columns):
            new_board = self.simulate_move(board, col, self.player_id)
            if self.check_win(new_board, self.player_id):
                return col  # Immediate win
            score = self.minimax(new_board, self.n - 1, alpha, beta, False)
            if score > best_score:
                best_score = score
                best_col = col
            alpha = max(alpha, best_score)
        # logger.info(best_col)
        return int(best_col)

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or self.is_terminal(board):
            return self.evaluate(board)
        valid_columns = self.get_valid_columns(board)
        if maximizing_player:
            max_score = -float("inf")
            for col in self.order_columns(valid_columns):
                new_board = self.simulate_move(board, col, self.player_id)
                score = self.minimax(new_board, depth - 1, alpha, beta, False)
                max_score = max(max_score, score)
                alpha = max(alpha, max_score)
                if beta <= alpha:
                    break
            return max_score
        else:
            min_score = float("inf")
            for col in self.order_columns(valid_columns):
                new_board = self.simulate_move(board, col, self.opponent_id)
                score = self.minimax(new_board, depth - 1, alpha, beta, True)
                min_score = min(min_score, score)
                beta = min(beta, min_score)
                if beta <= alpha:
                    break
            return min_score

    def get_valid_columns(self, board):
        return [col for col in range(7) if board[0][col] == 0]

    def order_columns(self, columns):
        # Order columns starting from center
        return sorted(columns, key=lambda x: abs(x - 3))

    def simulate_move(self, board, col, player_id):
        new_board = np.copy(board)
        for row in reversed(range(6)):
            if new_board[row][col] == 0:
                new_board[row][col] = player_id
                return new_board
        raise ValueError("Column is full")

    def check_win(self, board, player_id):
        # Check horizontal
        for row in range(6):
            for col in range(4):
                if (
                    board[row][col] == player_id
                    and board[row][col + 1] == player_id
                    and board[row][col + 2] == player_id
                    and board[row][col + 3] == player_id
                ):
                    return True
        # Check vertical
        for col in range(7):
            for row in range(3):
                if (
                    board[row][col] == player_id
                    and board[row + 1][col] == player_id
                    and board[row + 2][col] == player_id
                    and board[row + 3][col] == player_id
                ):
                    return True
        # Check diagonal down (top-left to bottom-right)
        for row in range(3):
            for col in range(4):
                if (
                    board[row][col] == player_id
                    and board[row + 1][col + 1] == player_id
                    and board[row + 2][col + 2] == player_id
                    and board[row + 3][col + 3] == player_id
                ):
                    return True
        # Check diagonal up (bottom-left to top-right)
        for row in range(3, 6):
            for col in range(4):
                if (
                    board[row][col] == player_id
                    and board[row - 1][col + 1] == player_id
                    and board[row - 2][col + 2] == player_id
                    and board[row - 3][col + 3] == player_id
                ):
                    return True
        return False

    def is_terminal(self, board):
        if self.check_win(board, self.player_id) or self.check_win(
            board, self.opponent_id
        ):
            return True
        return np.all(board != 0)

    def evaluate(self, board):
        if self.check_win(board, self.player_id):
            return 1000
        if self.check_win(board, self.opponent_id):
            return -1000
        if np.all(board != 0):
            return 0

        agent_score = 0
        opponent_score = 0

        # Check all lines
        lines = self.get_all_lines(board)
        for line in lines:
            a, o = self.evaluate_line(line)
            agent_score += a
            opponent_score += o

        return agent_score - opponent_score

    def get_all_lines(self, board):
        lines = []
        # Horizontal
        for row in range(6):
            for col in range(4):
                lines.append(
                    [
                        board[row][col],
                        board[row][col + 1],
                        board[row][col + 2],
                        board[row][col + 3],
                    ]
                )
        # Vertical
        for col in range(7):
            for row in range(3):
                lines.append(
                    [
                        board[row][col],
                        board[row + 1][col],
                        board[row + 2][col],
                        board[row + 3][col],
                    ]
                )
        # Diagonal down
        for row in range(3):
            for col in range(4):
                lines.append(
                    [
                        board[row][col],
                        board[row + 1][col + 1],
                        board[row + 2][col + 2],
                        board[row + 3][col + 3],
                    ]
                )
        # Diagonal up
        for row in range(3, 6):
            for col in range(4):
                lines.append(
                    [
                        board[row][col],
                        board[row - 1][col + 1],
                        board[row - 2][col + 2],
                        board[row - 3][col + 3],
                    ]
                )
        return lines

    def evaluate_line(self, line):
        agent_count = 0
        opponent_count = 0
        empty_count = 0
        for cell in line:
            if cell == self.player_id:
                agent_count += 1
            elif cell == self.opponent_id:
                opponent_count += 1
            else:
                empty_count += 1
        if agent_count > 0 and opponent_count > 0:
            return (0, 0)  # Blocked line
        if agent_count > 0:
            if agent_count == 3 and empty_count == 1:
                return (100, 0)
            elif agent_count == 2 and empty_count == 2:
                return (10, 0)
            elif agent_count == 1 and empty_count == 3:
                return (1, 0)
            else:
                return (0, 0)
        elif opponent_count > 0:
            if opponent_count == 3 and empty_count == 1:
                return (0, 100)
            elif opponent_count == 2 and empty_count == 2:
                return (0, 10)
            elif opponent_count == 1 and empty_count == 3:
                return (0, 1)
            else:
                return (0, 0)
        else:
            return (0, 0)


def agent_factory():
    agent = ConnectFourAgent()

    def agent_fn(obs, config):
        try:
            return agent.step(obs, config)
        except Exception as e:
            logger.error(e)

    return agent_fn
