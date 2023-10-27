import torch; import numpy as np

class ConnectFourEnvironment():
    
    def __init__(self):
        
        self.HEIGHT = 6; self.WIDTH = 7
        self.action_space = 7; self.observation_space = 42
        self.turn = 0 # Keep track of the turn number.
        self.board = torch.zeros((self.HEIGHT, self.WIDTH), dtype = torch.float32) # Initialize a 7*6 board, using torch.float32 as that is standard for PyTorch.
        self.row_cache = np.zeros(self.WIDTH, dtype = int) # Initialize a 7 element array to keep track of the next empty row in each column.
        self.reset()


    def drop_piece(self, col, piece):
        """
        Drops a piece in the specified column.
        Does not perform any checks, so it is possible to drop a piece in a full column.
        """
        self.board[self.row_cache[col]][col] = piece # Drop the piece in the first empty row.
        self.row_cache[col] += 1 # Increment the row cache.
        return self.row_cache[col] - 1 # Return the row the piece was dropped in.

    def is_valid_location(self, col):
        """
        Checks that the column is not full.
        """
        return self.row_cache[col] < self.HEIGHT


    def winning_move(self, piece, col, row):
        """
        Checks if the player has won.

        NOTE:
        We have tried to make this as efficient as possible, but it is still not very fast.
        The reason why we want this to be efficient is because the method is called every time a player makes a move,
        and when you do hundreds of thousands of iterations, it adds up.
        """

        if self.turn < 7:
            return False  # It is not possible to win before turn 7.
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)] # Iterate over all directions, horizontal, vertical, diagonal up and diagonal down.
        for dr, dc in directions:
            count = 0
            
            for i in range(-3, 4):
                r, c = row + i * dr, col + i * dc
                if 0 <= r < self.HEIGHT and 0 <= c < self.WIDTH and self.board[r][c] == piece: # Checking if coordinate is within the board and if the piece is the same as the one we are looking for.
                    count += 1
                    if count == 4: # If we have had four consecutive pieces in a row, we have won.
                        return True
                else:
                    count = 0
                    if i > 0: # It is no longer possible to win in this direction.
                        break
        return False

    def reset(self):
        """
        Reset the game to start a new training run.
        The board is set to all zeros.
        """
        self.board.fill_(0); self.row_cache.fill(0); self.turn = 0 # Reset the board, row cache, and turn counter.
        return self.board.flatten()

    def step(self, action):
        """
        The agent does an action, and the environment returns the next state, the reward, and whether the game is over.
        The action number corresponds to the column which the piece should be dropped in.
        return: (next_state, reward, done)
        """
        
        self.turn += 1
        piece = -1 if self.turn % 2 == 0 else 1
        row = self.drop_piece(action, piece)
        outputBoard = self.board.flatten()
        
        if not self.is_valid_location(action): # Illegal move, opponent wins.
            return (outputBoard, -1, True)
        
        if self.winning_move(piece, action, row): # Bot won
            return (outputBoard, 1, True)
        
        if self.turn == 42: # Draw
            return (outputBoard, 0, True)
        
        else: # Game continues
            return (outputBoard, 0, False)