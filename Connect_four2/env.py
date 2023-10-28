import torch
from debug_utils.print_board import print_state

class ConnectFourEnvironment():
    
    def __init__(self, opponent):
        
        self.HEIGHT = 6; self.WIDTH = 7
        self.action_space = 7; self.observation_space = 42
        self.turn = 0 # Keep track of the turn number.
        self.board = torch.zeros((self.HEIGHT, self.WIDTH), dtype = torch.float32) # Initialize a 7*6 board, using torch.float32 as that is standard for PyTorch.
        self.row_cache = torch.zeros(self.WIDTH, dtype = torch.int8) # Initialize a 7 element array to keep track of the next empty row in each column.
        self.opponent_Q_network = opponent # Load the opponent, which is an earlier version of the trained Q-network.

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

    def opponent_act(self, state):
        """
        The opponent is using a greedy policy.
        """
        return torch.argmax(self.opponent_Q_network(state)).item()

    def reset(self, agent_moves_first):
        """
        Reset the game to start a new training run.
        The board is set to all zeros if the agent makes the first move, otherwise the opponent makes the first move.
        """
        self.board.fill_(0); self.row_cache.fill_(0); self.turn = 0 # Reset the board, row cache, and turn counter.
        
        if not agent_moves_first:
            self.opponent_move() # The opponent makes the first move.
        
        return self.board.flatten().clone() # Return the board as a flattened array.

    def step(self, action):
        """
        The agent does an action, and the environment returns the next state, the reward, and whether the game is over.
        The action number corresponds to the column which the piece should be dropped in.
        return: (next_state, reward, done)
        """
        return self.player1_move(action) # The agent makes a move.
        

    def player1_move(self, action):
        """
        Player1 (the agent) makes a move.
        """
        self.turn += 1; piece = -1 if self.turn % 2 == 0 else 1
        board = self.get_correct_board_representation(piece) # Make sure that the agent sees the board from its own perspective.
        if not self.is_valid_location(action): # If the bot wants to make an illegal move, the opponent wins.
            return (board, -1, True)
        
        row = self.drop_piece(action, piece) # Drop the piece in the specified column.
        next_state = self.get_correct_board_representation(piece)
        
        if self.winning_move(piece, action, row): # Bot won
            return (next_state, 1, True)
        
        if self.turn == 42: # Draw
            return (next_state, 0, True)

        return self.opponent_move() # Opponent's turn
    
    def opponent_move(self):
        """
        The opponent (an earlier version of the trained Q-network) makes a move.
        """
        self.turn += 1; piece = -1 if self.turn % 2 == 0 else 1
        state = self.get_correct_board_representation(piece)
        action = self.opponent_act(state) # Make sure that the opponent sees the board from its own perspective.
        if not self.is_valid_location(action): # If the opponent wants to make an illegal move, the agent wins.
            return (state, 1, True)
        
        row = self.drop_piece(action, piece) # Drop the piece in the specified column.
        next_state = self.board.flatten().clone() # Clone the board to avoid referencing the same object.
        
        if self.winning_move(piece, action, row): # Opponent won
            return (next_state, -1, True) # Rewards are given from the perspective of the agent.
        
        if self.turn == 42: # Draw
            return (next_state, 0, True)
        
        else: # The game is not over.
            return (next_state, 0, False) # This is where you will usually end up.
        
    def get_correct_board_representation(self, piece):
        """
        Makes sure that the player sees the board from its own perspective.
        Also makes sure that the board is returned as a flattened array, and that the board is cloned to avoid
        a problem where the stored board is changed after the board is returned.
        """
        return self.board.flatten().clone() * piece

def print_procedure(action):
    print(f'State before action:')
    print_state(env.board.flatten())
    state, reward, done = env.step(action)
    print(f'Player: {2 if env.turn % 2 == 0 else 1}, Action: {action}')
    print(f'Next state:')
    print_state(state)
    print(f'reward: {reward}, done: {done}')

if __name__ == '__main__':
    env = ConnectFourEnvironment()

    # Simulating some moves for debugging purposes.
    print_procedure(3) # Move 1, Player 1
    print_procedure(3) # Move 2, Player 2
    print_procedure(3) # Move 3, Player 1
    print_procedure(3) # Move 4, Player 2
    print_procedure(3) # Move 5, Player 1
    print_procedure(3) # Move 6, Player 2
    print_procedure(3) # Move 7, Illegal move, Player 1 should lose.
