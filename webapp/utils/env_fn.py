import os 
import pygame
import pickle
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

pth = os.path.dirname(os.path.abspath(__file__))

class four_in_a_row:
    name = 'four_in_a_row'
    rows = 4
    cols = 9
    win_length = 4
    win_reward = 1
    lose_reward = -1
    draw_reward = 0
    center = np.array([1.5, 4])
    player1_color = 0 # black
    player2_color = 1 # white
    not_occupied = .75 # the available grid 
    cell_size = 100

    @staticmethod
    def get_valid_actions(board):
        if four_in_a_row.check_draw(board, four_in_a_row.not_occupied):
            return []
        elif four_in_a_row.check_win(board, four_in_a_row.not_occupied):
            return []
        else:
            l = np.vstack(np.where(board==four_in_a_row.not_occupied)).T
            return [(i[0], i[1]) for i in l]
    
    @staticmethod
    def check_win(board, not_occupied=.75):
        """Wining condition check
        
        Check if there are 
            1) 4 consecutive pieces in a row 
            2) 4 consecutive pieces in a column 
            3) 4 consecutive pieces in a diagonal 

        Horizontal:
            x x x x

        Vertical:
            x
            x
            x
            x

        Diagonal1:
            x
                x
                x
                    x

        Diagonal2:
            x
                x
                x
                    x

        The lru_cache is used to cache the result of the function. This 
        can be used to speed up the function.

        Inputs:
            board (np.ndarray): the board of the game
            not_occupied (float): the value of the not occupied grid
        Outputs:
            done (bool): whether the game is win or not finished
        """
        rows, cols = board.shape
        not_occupied = four_in_a_row.not_occupied
        # check horizontal
        # check horizontal
        for i in range(rows):
            for j in range(cols-3):
                # for each unoccupied grid
                q1 = board[i, j]!=not_occupied
                q2 = board[i, j]==board[i, j+1]==board[i, j+2]==board[i, j+3]
                if q1 and q2: return True
                    
        # check vertical
        for i in range(rows-3):
            for j in range(cols):
                q1 = board[i, j]!=not_occupied
                q2 = board[i, j]==board[i+1, j]==board[i+2, j]==board[i+3, j]
                if q1 and q2: return True

        # check diagonal1 (top-left to bottom-right)
        for i in range(rows-3):
            for j in range(cols-3):
                q1 = board[i, j]!=not_occupied
                q2 = board[i, j]==board[i+1, j+1]==board[i+2, j+2]==board[i+3, j+3]
                if q1 and q2: return True

        # check diagonal2 (top-right to bottom-left)
        for i in range(rows-3):
            for j in range(3, cols):
                q1 = board[i, j]!=not_occupied
                q2 = board[i, j]==board[i+1, j-1]==board[i+2, j-2]==board[i+3, j-3]
                if q1 and q2: return True

        return False

    @staticmethod
    def check_win_action(board, action, player_id=0):
        '''Check if the last action cause a win

        Check if there are 
            1) 4 consecutive pieces in a row 
            2) 4 consecutive pieces in a column 
            3) 4 consecutive pieces in a diagonal 

        Horizontal:
            x x x x

        Vertical:
            x
            x
            x
            x

        Diagonal1:
            x
                x
                    x
                        x

        Diagonal2:
                        x
                    x
                x
            x        
        
        Inputs:
            board (np.ndarray): the board of the game
            action (tuple): the action to take
            not_occupied (float): the value of the not occupied grid
        '''
        rows, cols = board.shape
        x, y = action
        
        # Check horizontal
        count = 0
        for c in range(max(0, y-3), min(cols, y+4)):
            if board[x, c]==player_id:
                count += 1
                if count >= 4:
                    return True
            else:
                count = 0

        # Check vertical
        count = 0
        for r in range(max(0, x-3), min(rows, x+4)):
            if board[r, y]==player_id:
                count += 1
                if count >= 4:
                    return True
            else:
                count = 0

        # Check diagonal1 (top-left to bottom-right)
        count = 0
        for i in range(-3, 4):
            r, c = x+i, y+i
            if 0<=r<rows and 0<=c<cols:
                if board[r, c]==player_id:
                    count += 1
                    if count >= 4:
                        return True
                else:
                    count = 0

        # Check diagonal2 (top-right to bottom-left)
        count = 0
        for i in range(-3, 4):
            r, c = x+i, y-i
            if 0<=r<rows and 0<=c<cols:
                if board[r, c]==player_id:
                    count += 1
                    if count >= 4:
                        return True
                else:
                    count = 0

        return False

    @staticmethod
    def check_draw(board, not_occupied=.75):
        return np.all(board!=not_occupied)
    
    @staticmethod
    def determine(board):
        if four_in_a_row.check_draw(board):
            return True
        elif four_in_a_row.check_win(board):
            return True
        return False

    def __init__(self, ):
        # Define the board 
        self._define_S()
        self._define_A()
        self._define_P_R()
        self.reset()
        # Initialize pygame
        self.pygame_initialized = False
        self.screen = None

    def _define_S(self):
        '''Define the state
            
        The state is defined by 
            1) pieces on the board 
            2) the current player 
        '''
        pass 

    def _define_A(self):
        '''Define the action
            
        The action is the coordinates of the piece to
        drop on the board.
        '''
        pass 

    def _define_P_R(self):
        '''Define the transition function
        
        The transition function is the function that
        takes the current state and the action and
        returns the next state.

        see the transit function for more details
        '''
        pass 
    
    def transit(self, state, action):
        '''Define the transition and reward functions (rules of the game)

        1) The piece can be dropped when the gird is not occupied.
        2) after one player drop a piece, it will be the other player's turn.
        3) the game is over when one player has 4 consecutive pieces 
            in a row, column, or diagonal.
        4) the reward is 1 for the winner, -1 for the loser, 
            and 0 for the draw and unfinised game.

        Inputs:
            state (tuple): (board, curr_player) the current state 
                * board: np.ndarray, the board of the game
                * curr_player: int, 1-black player, 2-white player
            action (tuple): (row, col) the action to take
                * row: int, the row to drop the piece
                * col: int, the column to drop the piece

        Outputs:
            tuple: (next_state, reward, done, info)
                * next_state: (board, curr_player), the next state of the game
                * reward: int, 1/-1/0 for the winner/loser/draw and unfinised game
                * done: bool, whether the game is over
                * info: dict, the info of the game

        '''
        # the board and the current player
        board, curr_player = state
        x, y = action
        reward, done, info = 0, False, {}

        # check if the action is valid
        valid = board[x, y]==four_in_a_row.not_occupied

        if valid:
            board_next = deepcopy(board)
            curr_player_next = curr_player
            # update the board
            board_next[x, y] = curr_player
            # switch the player
            curr_player_next = 1-curr_player

            # check if the game is draw
            draw = self.check_draw(board, four_in_a_row.not_occupied)

            # check if the game is finished
            if not draw:
                done = self.check_win_action(board, action, curr_player)
                if done:
                    #print('winner', last_player)
                    reward = four_in_a_row.win_reward 
                    winner = 'black' if curr_player==0 else 'white'
                    info = {'winner': winner}
                else:
                    reward = 0
                    info = {}
            else:
                done = True
                info = {'winner': 'draw'}

            next_state = (board_next, curr_player_next)
            return next_state, reward, done, info

        return state, reward, done, info
     
    def reset(self):
        """Reset the environment to initial state"""
        self.board = np.ones([self.rows, self.cols])*self.not_occupied
        self.curr_player = self.player1_color
        self.state = (self.board.copy(), self.curr_player)
        return self.state 
    
    def step(self, action):
        """Take a step in the environment"""
        next_state, reward, done, info = self.transit(self.state, action)
        self.state = next_state
        self.board = next_state[0]
        self.curr_player = next_state[1]
        return next_state, reward, done, info
                
    def render(self, mode='plt'):
        """Print the current state of the board
        
        Inputs:
            mode (str): the mode to render the board
                * 'plt': use plt to render the board
                * 'game': pygame mode 
        """
        if mode=='plt':
            empty_board = np.ones([self.rows, self.cols])*self.not_occupied
            # Draw filled scatter points on non-empty grids
            sns.heatmap(empty_board, cmap='gray', vmin=0, vmax=1, 
                        lw=.1, square=True, cbar=False)
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.board[i, j] != self.not_occupied:
                        color = self.board[i,j]*np.ones([3])
                        plt.scatter(j+.5, i+.5, s=500, color=color)
            plt.axis('off')
            plt.show()

        elif mode=='game':
            board = self.board.copy()
            curr_player = self.curr_player
            
            # Initialize pygame if needed
            if not self.pygame_initialized:
                pygame.init()
                self.pygame_initialized = True
                row_size = (self.rows + 2) * self.cell_size
                col_size = (self.cols + 2) * self.cell_size
                self.screen = pygame.display.set_mode((col_size, row_size))
                pygame.display.set_caption('Four in a Row')
                self.font = pygame.font.SysFont('Arial', int(self.cell_size * 0.4))
            
            # Fill the background
            self.screen.fill((255, 255, 255))

            # Draw "player to move" text at the top
            player = 'Black' if curr_player == self.player1_color else 'White'
            text = self.font.render(f'{player} to move', True, (0, 0, 0))
            text_rect = text.get_rect(center=(self.screen.get_width() // 2, self.cell_size // 2))
            self.screen.blit(text, text_rect)

            # Draw outer rectangle around the game board
            ad = .03
            pygame.draw.rect(
                self.screen,
                (80, 80, 80),  # Darker gray border
                ((1-ad)*self.cell_size, (1-ad)*self.cell_size, 
                (self.cols+ad*2)*self.cell_size, (self.rows+ad*2.2)*self.cell_size),
                width=4,  # Border width
            )
            
            # Draw the grid lines
            for i in range(self.rows):
                for j in range(self.cols):
                    # Draw background for each cell
                    bg_color = (200, 200, 200)
                    # Draw cell background
                    pygame.draw.rect(
                        self.screen, 
                        bg_color,
                        ((j+1) * self.cell_size, (i+1) * self.cell_size, self.cell_size, self.cell_size)
                    )
                    
                    # Draw grid lines
                    pygame.draw.rect(
                        self.screen,
                        (80, 80, 80),  # Grid line color
                        ((j+1) * self.cell_size, (i+1) * self.cell_size, self.cell_size, self.cell_size),
                        2  # Line width
                    )
            
            # Draw the pieces
            for i in range(self.rows):
                for j in range(self.cols):
                    if board[i, j] != self.not_occupied:
                        color = (0, 0, 0) if board[i, j] == self.player1_color else (255, 255, 255)
                        center = ((j+1) * self.cell_size + self.cell_size // 2, 
                                 (i+1) * self.cell_size + self.cell_size // 2)
                        radius = self.cell_size // 2 - 10
                        pygame.draw.circle(self.screen, color, center, radius)
            
            # Update display
            pygame.display.flip()

    # ---------------- for fitting ---------------- #

    @staticmethod
    def embed(state_idx):
        '''Embed the state indx into a state tuple

        Inputs:
            state_idx (int): the index of the state
        
        Outputs:
            state (tuple): the state of the game
                * board: np.ndarray, the board of the game
                * curr_player: int, 1-black player, 2-white player
        '''
        # load idx2state
        fname = f'{pth}/../data/human_vs_human-idx2state.pkl'
        with open(fname, 'rb') as f: idx2state = pickle.load(f)
        design = idx2state[state_idx]
        board = design[:-1].reshape([four_in_a_row.rows, four_in_a_row.cols])
        player_id = int(design[-1])
        return (board, player_id)

    @staticmethod
    def get_design_response_mat(sub_data):
        '''Get the design matrix of the state

        Combine the 
            * board: the board of the game
            * curr_player: the current player
            into a vector (1XK), and build a 
            design matrix (NxK) for the block data.
            N is the number of trials within the block.

        Inputs:
            sub_data (dict): the subject data
                * key: block_id, value: block_data

        Outputs:
            design_matrix (np.ndarray): the design matrix
            response_matrix (np.ndarray): the response matrix
        '''
        design_matrix = []
        response_matrix = []
        block_lst = sub_data.keys()
        for block in block_lst:
            block_data = sub_data[block]
            for _, row in block_data.iterrows():
                state_idx = row['state_idx']
                # save the state idx 
                design_matrix.append(state_idx)
                # save the action idx 
                action_idx = four_in_a_row.action2idx(eval(row['action']))
                response_matrix.append(action_idx)
        return np.array(design_matrix), np.array(response_matrix)
    
    @staticmethod
    def action2idx(action):
        '''Convert the coordinates to an integer'''
        x, y = action
        return x*four_in_a_row.cols+y
    
    @staticmethod
    def idx2action(idx):
        '''Convert the integer to coordinates'''
        return idx//four_in_a_row.cols, idx%four_in_a_row.cols



if __name__ == '__main__':

    env = four_in_a_row()
    env.reset()
    running = True
    # Game loop
    clock = pygame.time.Clock()

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

                if done:
                    env.render(mode='game')
                    pygame.time.delay(1000)
                    print('New game')
                    obs = env.reset()
        
        pygame.time.wait(150)  # Small delay to not hog CPU
    
    # Cap the frame rate
    clock.tick(30)

