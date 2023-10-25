import pygame
import sys
import torch
import numpy as np
from QNetwork import QNetwork

class ConnectFourPygame:
    
    def __init__(self):
        
        pygame.init()

        self.WIDTH, self.HEIGHT = 700, 600
        self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)

        self.ROW_COUNT = 6
        self.COLUMN_COUNT = 7
        self.SQUARESIZE = 100
        self.RADIUS = int(self.SQUARESIZE / 2 - 5)

        self.board = np.array([[0] * self.COLUMN_COUNT for r in range(self.ROW_COUNT)])
        self.turn = 0
        self.running = True

        self.opponent = QNetwork(42, 7, 100)

        pygame.font.init()
        self.font = pygame.font.Font(None, 74)

    def get_board(self):
        return self.board

    def draw_board(self):
        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT):
                pygame.draw.rect(self.window, self.BLACK, (c * self.SQUARESIZE, r * self.SQUARESIZE + self.SQUARESIZE, self.SQUARESIZE, self.SQUARESIZE))
                pygame.draw.circle(self.window, self.BLACK, (int(c * self.SQUARESIZE + self.SQUARESIZE / 2), int(r * self.SQUARESIZE + self.SQUARESIZE + self.SQUARESIZE / 2)), self.RADIUS)

        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT):
                if self.board[r][c] == 1:
                    pygame.draw.circle(self.window, self.RED, (int(c * self.SQUARESIZE + self.SQUARESIZE / 2), self.HEIGHT - int(r * self.SQUARESIZE + self.SQUARESIZE / 2)), self.RADIUS)
                elif self.board[r][c] == 2:
                    pygame.draw.circle(self.window, self.YELLOW, (int(c * self.SQUARESIZE + self.SQUARESIZE / 2), self.HEIGHT - int(r * self.SQUARESIZE + self.SQUARESIZE / 2)), self.RADIUS)
        pygame.display.update()

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
        self.board = [[0] * self.COLUMN_COUNT for r in range(self.ROW_COUNT)]

    def load_opponent(self, path):
        self.opponent.load_state_dict(torch.load(path))

    def get_move(self, state):
        print(self.opponent(torch.tensor(state, dtype = torch.float32)))
        return torch.argmax(self.opponent(torch.tensor(state, dtype = torch.float32))) # Get the best action

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    col = event.pos[0] // self.SQUARESIZE
                    if self.is_valid_location(col):
                        row = self.get_next_open_row(col)
                        self.drop_piece(row, col, self.turn % 2 + 1)

                        if self.winning_move(self.turn % 2 + 1):
                            self.draw_board()
                            print(f"Player {self.turn % 2 + 1} wins!")
                            self.running = False
                            
                            text_surface = self.font.render(f"Player {self.turn % 2 + 1} wins!", True, (255, 255, 255))
                            text_rect = text_surface.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
                            self.window.blit(text_surface, text_rect)
                            pygame.display.update()
                            
                            pygame.time.wait(3000)

                        self.turn += 1
                        self.draw_board()

            pygame.display.update()

    ### reset
    ### step
    def run_with_bot(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    col = event.pos[0] // self.SQUARESIZE
                    if self.is_valid_location(col):
                        row = self.get_next_open_row(col)
                        self.drop_piece(row, col, self.turn % 2 + 1)

                        if self.winning_move(self.turn % 2 + 1):
                            self.draw_board()
                            print(f"Player {self.turn % 2 + 1} wins!")
                            self.running = False
                            
                            text_surface = self.font.render(f"Player {self.turn % 2 + 1} wins!", True, (255, 255, 255))
                            text_rect = text_surface.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
                            self.window.blit(text_surface, text_rect)
                            pygame.display.update()
                            
                            pygame.time.wait(3000)
                        
                        self.turn += 1
                        self.draw_board()

                    pygame.display.update()

                    #AI turn
                    col = self.get_move((self.board * (-1)).flatten())
                    row = self.get_next_open_row(col)
                    self.drop_piece(row, col, self.turn % 2 + 1)

                    if self.winning_move(self.turn % 2 + 1):
                        self.draw_board()
                        print(f"Player {self.turn % 2 + 1} wins!")
                        self.running = False
                        
                        text_surface = self.font.render(f"Player {self.turn % 2 + 1} wins!", True, (255, 255, 255))
                        text_rect = text_surface.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
                        self.window.blit(text_surface, text_rect)
                        pygame.display.update()
                        
                        pygame.time.wait(3000)
                    
                    self.turn += 1
                    self.draw_board()
                
                
            pygame.display.update()

if __name__ == "__main__":
    game = ConnectFourPygame()
    game.run_with_bot()
