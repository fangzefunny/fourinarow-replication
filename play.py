import os 
import time
import argparse
import numpy as np
import pandas as pd

from utils.env_fn import *
from utils.model import *
from utils.fig_fn import *

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--game', '-g', help='the game to play', type=str, default='four_in_a_row')
parser.add_argument('--mode', '-m', help='human vs human or human vs ai', type=str, default='human_vs_ai')
args = parser.parse_args()
game_fn = eval(args.game)

# find the current path
pth = os.path.dirname(os.path.abspath(__file__))
dirs = [f'{pth}/data', f'{pth}/data/human_vs_human']
[os.mkdir(d) for d in dirs if not os.path.exists(d)]

def human_vs_human(collect_data=True, white_player='', black_player='', block_id=0):

    env = game_fn()
    if collect_data:
        # initialize a dict for game data collection 
        columns = ['board', 'play_to_move', 'action', 'done', 
                   'winner', 'trial', 'time_elapsed', 'rt']
        game_data = {c: [] for c in columns}
        block_id = int(block_id)
     
    # start the game 
    running = True
    clock = pygame.time.Clock()
    state = env.reset()
    block_start_time = time.time()
    state_present_time = time.time()
    t = 0
    while running:
        env.render(mode='game')
    
        # Get mouse clicks for human player input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Get the column from mouse position
                pos_x, pos_y = event.pos
                grid_x = pos_y // env.cell_size   # Row
                grid_y = pos_x // env.cell_size  # Column
                # Find the first available row from bottom
                action = (grid_x-1, grid_y-1)
                next_state, reward, done, info = env.step(action)
                response_time = time.time()

                # collect data 
                if collect_data:
                    board, player_to_move = state
                    game_data['board'].append(board2layout(board))
                    game_data['play_to_move'].append(player_to_move)
                    game_data['action'].append(str(action))
                    game_data['done'].append(done)
                    winner = 'None' if len(info)==0 else info['winner']
                    game_data['winner'].append(winner)
                    game_data['trial'].append(t)
                    elasped_seconds = response_time - block_start_time
                    game_data['time_elapsed'].append(elasped_seconds)
                    rt = response_time - state_present_time
                    game_data['rt'].append(rt)

                # update the trial and state 
                state_present_time = response_time
                state = next_state
                t += 1

                if done:
                    # save the collected data 
                    if collect_data:
                        fname = f"{pth}/data/human_vs_human/Human_vs_Human_data"
                        fname += f"-white={white_player}-black={black_player}-block={block_id}.csv"
                        pd.DataFrame(game_data).to_csv(fname, index=False)
                        block_id += 1
                    
                    # restart the game 
                    pygame.time.delay(1000)
                    print('New game')
                    state = env.reset()
                    env.render(mode='game')
                    # flip the players
                    white_player, black_player = black_player, white_player
                    block_start_time = time.time()
                    t = 0
        
        pygame.time.wait(150)  # Small delay to not hog CPU
    
    # Cap the frame rate
    clock.tick(30)

def human_vs_ai():
    # Initialize pygame
    pygame.init()
    
    # Initialize environment and model
    env = game_fn()
    env.reset()
    
    # Set up players and model
    human_player = 0  # Black pieces
    ai_player    = 1  # White piecesr
    dparams = default_params().to_list()
    model = BFS_agent(deepcopy(env), 
                           params=dparams)
    
    # Game state
    state = (env.board, human_player)
    done = False
    waiting_for_ai = False
    
    # Game loop
    running = True
    clock = pygame.time.Clock()
    
    print("\nGame starts! You play as black (0)")
    print("Click on a column to place your piece")
    
    while running:
        # Render current state
        env.render(mode='game')
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return
                
            # Handle human player input
            if event.type == pygame.MOUSEBUTTONDOWN and not waiting_for_ai and not done and state[1] == human_player:
                # Get column from mouse position
               # Get the column from mouse position
                pos_x, pos_y = event.pos
                grid_x = pos_y // env.cell_size   # Row
                grid_y = pos_x // env.cell_size   # Column
                # Find the first available row from bottom
                action = (grid_x-1, grid_y-1)
                state, reward, done, info = env.step(action)
                if done:
                    env.render(mode='game')
                    pygame.time.delay(1000)
                    print('New game')
                    if reward == 1:
                        print("\nGame Over! You win!")
                    else:
                        print("\nGame Over! It's a draw!")
                else:
                    waiting_for_ai = True

        # AI turn
        if not done and waiting_for_ai and state[1]==ai_player:
            print("\nAI is thinking...")
            action = model.get_action(state)
            state, reward, done, info = env.step(action)
            print(f"AI played: {action}")
            waiting_for_ai = False
            
            if done:
                env.render(mode='game')
                pygame.time.delay(1000)
                print('New game')
                if reward == 1:
                    print("\nGame Over! AI wins!")
                else:
                    print("\nGame Over! It's a draw!")
        
        # Update display
        pygame.display.flip()
        clock.tick(30)
        
        # Handle game restart
        if done:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                # Reset game
                env.reset()
                state = (env.board, human_player)
                done = False
                waiting_for_ai = False
                print("\nNew game! You play as black (0)")
                print("Click on a column to place your piece")


if __name__ == '__main__':

    if args.mode == 'human_vs_human':
        # load the game environment
        # env = game_fn()
        black_player = input("Black player: ")
        white_player = input("White player: ")
        block_id = input('Block id: ')
        human_vs_human(collect_data=True, 
                       white_player=white_player, 
                       black_player=black_player,
                       block_id=block_id)

    elif args.mode == 'human_vs_ai':
        
        human_vs_ai()
    else:
        raise ValueError('Invalid mode')