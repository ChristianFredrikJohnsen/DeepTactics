import torch
from QNetwork import QNetwork
from debug_utils.print_board import print_state

class ConnectFourEnvironment():
    
    def __init__(self, opponent):
        
        self.HEIGHT = 6; self.WIDTH = 7
        self.action_space = 7; self.observation_space = 42
        self.turn = 0 # Keep track of the turn number.
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available, otherwise use CPU.
        self.board = torch.zeros((self.HEIGHT, self.WIDTH), dtype = torch.float32, device = self.device) # Initialize a 7*6 board, using torch.float32 as that is standard for PyTorch.
        self.row_cache = torch.zeros(self.WIDTH, dtype = torch.int8) # Initialize a 7 element array to keep track of the next empty row in each column.
        self.opponent_Q_network = opponent.to(self.device) # Load the opponent, which is an earlier version of the trained Q-network.
        self.legal_moves = torch.ones(self.action_space, dtype = torch.float32, device = self.device) # Initialize a 7 element array to keep track of the legal moves.

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
    
    def potential_winning_move(self, piece, col, row):
        """
        Checks if the player has won.

        NOTE:
        We have tried to make this as efficient as possible, but it is still not very fast.
        The reason why we want this to be efficient is because the method is called every time a player makes a move,
        and when you do hundreds of thousands of iterations, it adds up.
        """

        if self.turn < 6:
            return False  # It is not possible to win before turn 6.
        
        row = self.drop_piece(col, piece) # Drop the piece in the first empty row.
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)] # Iterate over all directions, horizontal, vertical, diagonal up and diagonal down.
        for dr, dc in directions:
            count = 0
            
            for i in range(-3, 4):
                r, c = row + i * dr, col + i * dc
                if 0 <= r < self.HEIGHT and 0 <= c < self.WIDTH and self.board[r][c] == piece: # Checking if coordinate is within the board and if the piece is the same as the one we are looking for.
                    count += 1
                    if count == 4: # If we have had four consecutive pieces in a row, we have won.
                        self.board[row][col] = 0 # Remove the piece from the board.
                        self.row_cache[col] -= 1 # Undo the drop_piece method.
                        return True
                else:
                    count = 0
                    if i > 0: # It is no longer possible to win in this direction.
                        break
        
        # The possible move was not a winning move, so we must undo the drop_piece method.
        self.board[row][col] = 0 # Remove the piece from the board.
        self.row_cache[col] -= 1 # Undo the drop_piece method.

        return False

    def update_legal_moves(self):
        """
        Updates the list of legal moves.
        This method is used by the opponent to get a list of legal moves.
        """
        for i in range(self.action_space):
            self.legal_moves[i] = self.row_cache[i] < self.HEIGHT

    def opponent_act(self, state):
        """
        The opponent is using a greedy policy.
        We must give the opponent a list of legal moves, otherwise it may end
        up in an infinite loop where it tries to make an illegal move.
        """
        self.update_legal_moves() # Update the list of legal moves.
        qvals = self.opponent_Q_network(state) # Get the Q-values for the current state.
        for i in range(self.action_space):
            if not self.legal_moves[i]: qvals[i] = float('-inf') # If the move is illegal, we set the Q-value to -inf. 
        action = torch.argmax(qvals).item() # Get the action with the highest Q-value among the legal moves.

        return action

    def opponent_winning_move(self, piece):
        """
        Checks if the opponent has a winning move.
        If it does, then it plays it.
        """
        for i in range(self.action_space):
            if self.is_valid_location(i) and self.potential_winning_move(piece, i, self.row_cache[i]):
                self.drop_piece(i, piece)
                next_state = self.get_correct_board_representation(piece)
                return (next_state * (-1), -1, True) # Opponent won, so player gets reward -1.
            
        return None # No winning move was found.

    def player_winning_move(self, piece):
        """
        Checks if the player has a winning move (The agent who trains).
        If the player has a winning move, the opponent must block it.
        """
        opponent_piece = piece * (-1)
        for i in range(self.action_space):
            
            if self.is_valid_location(i) and self.potential_winning_move(opponent_piece, i, self.row_cache[i]):
                self.drop_piece(i, piece) # Block the player's winning move.
                next_state = self.get_correct_board_representation(piece)
                if self.turn == 42:
                    return (next_state * (-1), 0, True) # Draw
                return (next_state * (-1), 0, False) # The game goes on, a blocking move was made.
            
        return None # No blocking move was found.



    def reset(self, agent_moves_first):
        """
        Reset the game to start a new training run.
        The board is set to all zeros if the agent makes the first move, otherwise the opponent makes the first move.
        """
        self.board.fill_(0); self.row_cache.fill_(0); self.turn = 0 # Reset the board, row cache, and turn counter.
        
        if not agent_moves_first:
            self.opponent_move() # The opponent makes the first move.
        
        return self.board.flatten().clone() * (-1) # Return the board as a flattened array, and make sure that the agent sees the board from its own perspective.

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
        if not self.is_valid_location(action): # If the bot wants to make an illegal move, we punish it, and we force it to make a new (and hopefully legal) move.
            self.turn -= 1 # We do not want to increment the turn counter if the bot makes an illegal move, it must try again.
            return (board, -1, False)
        
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
        
        win = self.opponent_winning_move(piece)
        if win is not None: return win # If the opponent has a winning move, it plays it.

        block = self.player_winning_move(piece)
        if block is not None: return block # If the player has a winning move, the opponent must block it.
        
        # If the opponent does not have an immediate winning or blocking move, it plays whatever move it thinks is best.
        action = self.opponent_act(state) # Make sure that the opponent sees the board from its own perspective.
        self.drop_piece(action, piece) # Drop the piece in the specified column.
        next_state = self.get_correct_board_representation(piece) # Clone the board to avoid a problem where the stored board is changed after the board is returned.
        
        if self.turn == 42: # Draw
            return (next_state * (-1), 0, True)
        
        else: # The game is not over.
            return (next_state * (-1), 0, False) # This is where you will usually end up.
        
    def get_correct_board_representation(self, piece):
        """
        Makes sure that the player sees the board from its own perspective.
        Also makes sure that the board is returned as a flattened array, and that the board is cloned to avoid
        a problem where the stored board is changed after the board is returned.
        """
        return self.board.flatten().clone() * piece

def print_procedure(env, action):
    print(f'State before action:')
    print_state(env.board.flatten())
    print(f'Player: {1 if env.turn % 2 == 0 else 2}, Action: {action}')
    print(f'Turn: {env.turn}')
    state, reward, done = env.step(action)
    print(f'Next state:')
    print_state(state)
    print(f'reward: {reward}, done: {done}')

if __name__ == '__main__':
    print("You ran the environment file! Good job!")

    Qnetwork = QNetwork(42, 7, 1500)
    state_dict = torch.load("models/connect4_christian_bigboy.pk1", map_location=torch.device('cpu'))
    Qnetwork.load_state_dict(state_dict)
    env = ConnectFourEnvironment(Qnetwork)

    # Simulating some moves for debugging purposes.
    print_procedure(env, 3) # Move 1, Player 1
    print_procedure(env, 2) # Move 2, Player 2
    print_procedure(env, 1) # Move 3, Player 1
    print_procedure(env, 0) # Move 4, Player 2
    print_procedure(env, 2) # Move 5, Player 1
    print_procedure(env, 2) # Move 6, Player 2
    # print_procedure(3) # Move 2, Player 2
    # print_procedure(3) # Move 3, Player 1
    # print_procedure(3) # Move 4, Player 2
    # print_procedure(3) # Move 5, Player 1
    # print_procedure(3) # Move 6, Player 2
    # print_procedure(3) # Move 7, Illegal move, Player 1 should lose.
