import torch; from colorama import Fore, Style; import numpy as np

def print_status(score, episode_num, win_check, state, results, epsilon):
    """
    Prints game information to the terminal.
    This includes the episode number, the score, the winrate, and the board state.
    """
    num_results = episode_num%win_check if episode_num < win_check else win_check
    player = 'red' if episode_num%2 == 1 else 'yellow'
    if(score == 1):
        print(f"{Fore.RED}player: {player}; episode: {episode_num}; score: {score}; epsilon: {epsilon}; winrate: {np.mean(results[:num_results])}{Style.RESET_ALL}")
    elif(score == -1):
        print(f"{Fore.YELLOW}player: {player}; episode: {episode_num}; score: {score}; epsilon: {epsilon}; winrate last {num_results} episodes: {np.mean(results[:num_results])}{Style.RESET_ALL}")

    print_board(torch.flip(state.reshape(6,7), [0]).cpu().numpy())

def print_state(state):
    """
    Prints the state to the terminal.
    Expects a flattened state as input.
    """
    print_board(torch.flip(state.reshape(6,7), [0]).cpu().numpy())
    print('~' * 20)

def print_board(board):
    """
    Prints the board to the terminal.
    """
    for i in range(6):
        for j in range(7):
            print_color(board[i][j])
        print()

def print_color(num):
    """
    Prints the piece number in the correct color.
    """
    if num == 1:
        print(f"{Fore.RED} 1 ", end='')     # Red
    elif num == -1:
        print(f"{Fore.YELLOW}-1 ", end='')  # Yellow
    elif num == 0:
        print(f"{Fore.WHITE} 0 ", end='')   # White