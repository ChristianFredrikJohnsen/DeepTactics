import pygame; import torch; import numpy as np
from QNetwork import QNetwork; from icecream import ic; from debug_utils.print_board import print_state

class ConnectFourPygame:
    
    def __init__(self, path):
        
        pygame.init(); pygame.display.set_caption("Connect Four"); pygame.display.set_icon(pygame.image.load("cogito_blue.png")) # Initialize pygame and set the window caption and icon.
        self.WINDOW_WIDTH, self.WINDOW_HEIGHT = 700, 600; self.window = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT)) # Setting up the game window.
        self.BLACK = (0, 0, 0); self.RED = (255, 0, 0); self.YELLOW = (255, 255, 0) # Define some colors.

        self.HEIGHT = 6; self.WIDTH = 7; self.turn = 0 # Keep track of the turn number.
        self.column_cache = np.zeros(self.WIDTH, dtype = int) # Initialize a 7 element array to keep track of the next empty row in each column.
        self.SQUARESIZE = 100; self.RADIUS = int(self.SQUARESIZE / 2 - 5) # Square size for the columns and radius of the pieces.

        self.board = torch.tensor([[0] * self.WIDTH for r in range(self.HEIGHT)], dtype = torch.float32); self.turn = 0; self.running = True # Game logic variables.

        self.opponent = QNetwork(42, 7, 500) # Initialize the opponent. Q network with 42 inputs, 7 outputs and 500 hidden nodes.
        self.load_opponent(path)

        pygame.font.init(); self.font = pygame.font.Font(None, 74) # Setting up font for the text.
        self.draw_board() # Draw the board on the screen.

    def draw_board(self):
        """
        Method used to draw the board on the screen after a player has made a move.
        This method is also run when the game is started, just to get the white lines on the screen.
        """
        line_color = (255, 255, 255)  # White color for the lines
        
        for c in range(self.WIDTH):
            for r in range(self.HEIGHT):
                pygame.draw.rect(self.window, self.BLACK, (c * self.SQUARESIZE, r * self.SQUARESIZE + self.SQUARESIZE, self.SQUARESIZE, self.SQUARESIZE))
                pygame.draw.circle(self.window, self.BLACK, (int(c * self.SQUARESIZE + self.SQUARESIZE / 2), int(r * self.SQUARESIZE + self.SQUARESIZE + self.SQUARESIZE / 2)), self.RADIUS)

        # Draw vertical lines
        for x in range(0, self.WINDOW_WIDTH, self.SQUARESIZE):
            pygame.draw.line(self.window, line_color, (x, 0), (x, self.WINDOW_HEIGHT), 1)

        # Draw horizontal lines
        for y in range(self.SQUARESIZE, self.WINDOW_HEIGHT, self.SQUARESIZE):
            pygame.draw.line(self.window, line_color, (0, y), (self.WINDOW_WIDTH, y), 1)

        for c in range(self.WIDTH):
            for r in range(self.HEIGHT):
                if self.board[r][c] == 1:
                    pygame.draw.circle(self.window, self.RED, (int(c * self.SQUARESIZE + self.SQUARESIZE / 2), self.WINDOW_HEIGHT - int(r * self.SQUARESIZE + self.SQUARESIZE / 2)), self.RADIUS)
                elif self.board[r][c] == -1:
                    pygame.draw.circle(self.window, self.YELLOW, (int(c * self.SQUARESIZE + self.SQUARESIZE / 2), self.WINDOW_HEIGHT - int(r * self.SQUARESIZE + self.SQUARESIZE / 2)), self.RADIUS)
        pygame.display.update()

    def drop_piece(self, col, piece):
        """
        Drops a piece in the specified column.
        Does not perform any checks, so it is possible to drop a piece in a full column.
        """
        self.board[self.column_cache[col]][col] = piece # Drop the piece in the first empty row.
        self.column_cache[col] += 1 # Increment the row cache.
        return self.column_cache[col] - 1 # Return the row that the piece was dropped in.

    def is_valid_location(self, col):
        """
        Checks that the column is not full.
        """
        return self.column_cache[col] != self.HEIGHT

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
        
    def load_opponent(self, path):
        """
        Loads the parameters of the trained neural network.
        """
        self.opponent.load_state_dict(torch.load(path))

    def get_move(self, state):
        """
        Uses the loaded neural network to generate a move for the opponent.
        The opponent is always player 2, so the state is flipped before being passed to the neural network.
        We always choose the action that maximizes the q value.
        """
        self.print_qvalues(state)
        return torch.argmax(self.opponent(state)) # Get the best action

    def colorize(self, qvalue, q_values):
        """
        Returns a colorized string representing the q-value.
        The color intenisty is relative to the min and max q-value for the given state.
        Red means negative, green means positive.
        """
        # Map the value to a color intensity between 0 and 4.
        m = min(q_values); M = max(q_values)
        

        ### Code for finding the color intensity. ###
        # Intensity is relative to the min and max qvalue for the given state.
        intensity = -1

        if qvalue < 0:
            ranges = [abs(m) / 5 * i for i in range(1, 5)]
            for i, val in enumerate(ranges):
                if abs(qvalue) < val:
                    intensity = i
                    break
        else:
            ranges = [M / 5 * i for i in range(1, 5)]
            for i, val in enumerate(ranges):
                if qvalue < val:
                    intensity = i
                    break
        
        intensity = 4 if intensity == -1 else intensity # If the value is in the last range, set intensity to 4.
        

        # Define shades of green and red (these are ANSI color codes)
        shades_of_green = [226, 190, 154, 118, 82]; shades_of_red = [52, 88, 124, 160, 196]
        color_code = shades_of_green[intensity] if qvalue > 0 else shades_of_red[intensity]
        
        return f"\033[38;5;{color_code}m{qvalue:.5f}\033[0m" # Return the colorized value


    def print_qvalues(self, state):
        """
        Printing the q-values after the opponent has made a move.
        This is useful for debugging, and it is also interesting to see how the q-values change during training.
        """

        q_values = self.opponent(state).detach().numpy() # A list containing the q-values for each action
        column_names = [f"Column{i}" for i in range(1, len(q_values) + 1)] # Create a list of column names

        # Decide the width of each column. We take the max length between each header and its corresponding value
        col_widths = [max(len(str(col)), len(f'{val:.5f}')) for col, val in zip(column_names, q_values)] # We are specifying that the q-values should be printed with 8 decimals.

        header = " | ".join([str(col).ljust(width) for col, width in zip(column_names, col_widths)]) # Column names
        values_str = " | ".join([self.colorize(val, q_values).ljust(width) for val, width in zip(q_values, col_widths)]) # Colorized q-values

        print(header)
        print("-" * len(header))  # Separator
        print(values_str)

    def run(self):
        """
        Run the game with two human players.
        """
        while self.running:
            for event in pygame.event.get():
                self.running = False if event.type == pygame.QUIT else True # Quit the game if the user closes the window.

                if event.type == pygame.MOUSEBUTTONDOWN:
                    col = event.pos[0] // self.SQUARESIZE # The column where the piece should be dropped.
                    player = 1 if self.turn % 2 == 0 else -1 # The current player
                    if self.is_valid_location(col):
                        self.turn += 1
                        self.play_move(player, col)

    def run_with_bot(self, print_bot_board = False):
        """
        Play connect 4 against a bot trained with reinforcement learning.
        """
        while self.running:
            
            for event in pygame.event.get():
                self.running = False if event.type == pygame.QUIT else True # Quit the game if the user closes the window.

                if event.type == pygame.MOUSEBUTTONDOWN: # Make a move and then let the bot make a move.
                    col = event.pos[0] // self.SQUARESIZE # Get the column where the piece should be dropped.
                    player = 1 if self.turn % 2 == 0 else -1 # The current player
                    if self.is_valid_location(col):
                        self.turn += 1
                        self.play_move(player, col)
                        self.generate_ai_turn(print_bot_board)

    def generate_ai_turn(self, print_bot_board):
        """
        Using the trained neural network to generate a move for the opponent.
        """
        if not self.running: return
        bot_board = self.board * (-1) # Flip the board so that the bot can use the same neural network as the one used for training.
        if print_bot_board: print_state(bot_board.flatten())
        col = self.get_move((self.board * (-1)).flatten())
        player = 1 if self.turn % 2 == 0 else -1 # AI should always be player -1, but we check just in case.
        self.turn += 1 # Assuming that the bot makes a legal move.
        self.play_move(player, col)        

    def play_move(self, player, col):
        """
        Plays a move for the given player in the given column.
        Checks if the player has won after the move is made, or if the game is a draw.
        """

        row = self.drop_piece(col, player)

        if self.winning_move(player, col, row):
            self.player_won(player)

        if self.turn == 42: # 42 moves means that the entire board is full with pieces, and no one has won.
            self.draw()

        self.draw_board()

        pygame.display.update()


    def player_won(self, player):
        """
        Game procedure for when a player has won.
        Some text is displayed on the screen, and then the game is closed after 3 seconds.
        """

        self.draw_board()
        self.running = False
        player = 2 if player == -1 else player # Player -1 sounds stupid. We want to display player 2 instead.

        # Text        
        text_surface = self.font.render(f"Player {player} wins!", True, (0, 255, 0))
        text_rect = text_surface.get_rect(center=(self.WINDOW_WIDTH // 2, self.WINDOW_HEIGHT // 2))

        # Background rectangle
        rect_x, rect_y, rect_w, rect_h = text_rect.x - 10, text_rect.y - 10, text_rect.width + 20, text_rect.height + 20
        pygame.draw.rect(self.window, (0, 128, 0), (rect_x, rect_y, rect_w, rect_h))

        self.window.blit(text_surface, text_rect)
        pygame.display.update()
        
        pygame.time.wait(3000)
    
    def draw(self):
        """
        Game procedure for when the game ends in a draw.
        Some text is displayed on the screen, and then the game is closed after 3 seconds.
        """
        self.draw_board()
        self.running = False
        
        # Text
        text_surface = self.font.render(f"Draw!", True, (0, 0, 255))
        text_rect = text_surface.get_rect(center=(self.WINDOW_WIDTH // 2, self.WINDOW_HEIGHT // 2))
        
        # Background rectangle
        rect_x, rect_y, rect_w, rect_h = text_rect.x - 10, text_rect.y - 10, text_rect.width + 20, text_rect.height + 20
        pygame.draw.rect(self.window, (0, 0, 128), (rect_x, rect_y, rect_w, rect_h))
        
        self.window.blit(text_surface, text_rect)
        pygame.display.update()
        
        pygame.time.wait(3000)

if __name__ == "__main__":
    game = ConnectFourPygame("models/connect4_christian_abomination.pk1")
    game.run_with_bot(print_bot_board = True)
