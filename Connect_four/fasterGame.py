import numpy as np

class ConnectFour:
    
    def __init__(self):
        
        self.ROW_COUNT = 6
        self.COLUMN_COUNT = 7
        self.turn = 0
        self.running = True
        self.action_space = 7

    def drop_piece(self, row, col, piece):
        self.board[row][col] = piece

    def is_valid_location(self, col):
        return self.board[self.ROW_COUNT - 1][col] == 0

    def get_next_open_row(self, col):
        for r in range(self.ROW_COUNT):
            if self.board[r][col] == 0:
                return r

    def winning_move(self, piece):
        
        # Check horizontal locations for win
        for c in range(self.COLUMN_COUNT - 3):
            for r in range(self.ROW_COUNT):
                if self.board[r][c] == piece and self.board[r][c+1] == piece and self.board[r][c+2] == piece and self.board[r][c+3] == piece:
                    return True

        # Check vertical locations for win
        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT - 3):
                if self.board[r][c] == piece and self.board[r+1][c] == piece and self.board[r+2][c] == piece and self.board[r+3][c] == piece:
                    return True

        # Check positively sloped diagonals
        for c in range(self.COLUMN_COUNT - 3):
            for r in range(self.ROW_COUNT - 3):
                if self.board[r][c] == piece and self.board[r+1][c+1] == piece and self.board[r+2][c+2] == piece and self.board[r+3][c+3] == piece:
                    return True

        # Check negatively sloped diagonals
        for c in range(self.COLUMN_COUNT - 3):
            for r in range(3, self.ROW_COUNT):
                if self.board[r][c] == piece and self.board[r-1][c+1] == piece and self.board[r-2][c+2] == piece and self.board[r-3][c+3] == piece:
                    return True

    def reset(self):
        self.board = np.zeros((self.ROW_COUNT, self.COLUMN_COUNT))
        return self.board.flatten()
    

    def generate_action(self):
        self.turn += 1
        legal_moves = [x for x in range(7) if self.is_valid_location(x)]
        action = np.random.choice(legal_moves)
        row = self.get_next_open_row(action)
        piece = -1 if self.turn % 2 == 0 else 1
        self.drop_piece(row, action, piece)

        if self.winning_move(-1 * piece):
            return (self.board.flatten(), -1, True)
        if self.turn == 42:
            return (self.board.flatten(), 0, True)
        else:
            if piece == 1:
                return (self.board.flatten(), 0, False)
            
            else:
                return (self.board.flatten() * -1, 0, False)

        #return action

    def step(self, action):
        """
        The agent does an action, and the environment returns the next state, the reward, and whether the game is over.
        The action number corresponds to the column which the piece should be dropped in.
        return: (next_state, reward, done)
        """
        
        self.turn += 1
        row = self.get_next_open_row(action)
        piece = -1 if self.turn % 2 == 0 else 1
        self.drop_piece(row, action, piece)
        

        if self.winning_move(piece):
            if piece == 1:
                return (self.board.flatten(), 1, True)
            else:
                return (self.board.flatten() * -1, 1, True)
        
        if self.turn == 42:
            return (self.board.flatten(), 0, True)
        
        returnself.generate_action()
        
       
if __name__ == "__main__":
    print("NEIN")
