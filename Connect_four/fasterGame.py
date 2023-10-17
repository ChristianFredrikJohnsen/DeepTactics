import numpy as np

class ConnectFour:
    
    def __init__(self):
        

        self.ROW_COUNT = 6
        self.COLUMN_COUNT = 7
        self.turn = 0
        self.running = True
        self.action_space = self.ActionSpace(7)



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
        return self.board
    

    def generate_action(self):
        self.turn += 1
        legal_moves = [x for x in range(6) if self.is_valid_location(x)]
        action = np.random.choice(legal_moves)
        self.step(action)
        return np.random.randint(0, 7)

    def step(self, action):

        row = self.get_next_open_row(action)
        piece = (-1) ** (self.turn % 2)
        self.drop_piece(row, action, piece)
        self.turn += 1

        if self.winning_move(piece):
            if piece == 1:
                return (self.board, 1, True)
            else:
                return (self.board * -1, 1, True)
        
        self.generate_action()
        

        else:
            if piece == 1:
                return (self.board, 0, False)
            else:
                return (self.board * -1, 0, False)

        


   
    ### reset
    ### step


if __name__ == "__main__":
    game = ConnectFour()
    game.run()
