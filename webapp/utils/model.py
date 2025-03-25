import numpy as np 
from copy import deepcopy
from utils.env_fn import *

# -------------- Basic units -------------- #

class queue:
    
    def __init__(self):
        self.elements = []
        
    def is_empty(self):
        return len(self.elements) == 0
        
    def put(self, x):
        self.elements.append(x)
        
    def get(self):
        if not self.is_empty():
            return self.elements.pop(0)
        return None

class Node:

    def __init__(self, state, action=None, parent=None, depth=0,
                 value=None, heuristic_fn=None):
        '''Node for a tree search

        The basic element in the tree search.
        Note that this node combien state and action 
        together, due to that the four in a row enviorment
        is deterministic.

        Inputs:
            state: tuple (board, player_id):
            action: tuple (row, col)
            parent: node
            depth: int
        '''
        # basic info 
        self.state     = state
        self.action    = action
        # tree info 
        self.parent    = parent
        self.children  = []
        self.value     = heuristic_fn(state) if heuristic_fn is not None else 0
        self.depth     = depth
        
class default_params:
    ''' Default parameters for the van plan agent
    
    Obtained from:
        https://github.com/basvanopheusden/ibs-development/blob/master/matlab/generate_resp_fourinarow.m
    '''
    lmbda = 0.02
    gamma = 0.02
    theta = 0.2
    delta = 0.005
    C     = 0.92498
    w_ce  = 0.60913
    w_c2  = 0.90444
    w_u2  = 0.45076
    w_c3  = 3.42720
    w_c4  = 15.17280

    def to_list(self):
        return [self.lmbda, self.gamma, self.theta, self.delta, self.C, 
                self.w_ce, self.w_c2, self.w_u2, self.w_c3, self.w_c4]

class basic_agent:

    def __init__(self, env):
        self.env = env

    def load_params(self, params: list):
        raise NotImplementedError
    
    def get_action(self, state: tuple):
        raise NotImplementedError
    
    def response_generator(self, params:list, design: np.array):
        self.load_params(params)
        state_lst = [self.env.embed(d) for d in design]
        action_lst = list(map(self.get_action, state_lst))
        return np.array([self.env.action2idx(action) for action in action_lst])
        
    def heuristic(self, state: tuple):
        raise NotImplementedError
    
# ----------- Heuristic Agents -------------- #

class heuristic_agent(basic_agent):
    '''Heuristic agent 

    The heuristic agent make decision based on pure 
    heuristic evaluation of the state
    '''
    name = 'heuristic_agent'
    p_names = ['lmbda', 'gamma', 'theta', 'delta', 'C', 
               'w_ce', 'w_c2', 'w_u2', 'w_c3', 'w_c4']
    p_bnds  = [(0,  1), (0,  1), (0, 10), (0,  1), (0,  2), 
               (0, 30), (0, 30), (0, 30), (0, 30), (0, 30)]
    p_pbnds = [(.001, .1), (.001, .1), (.001,  3), (.001, .1), (.7,.9), 
               (.1,  2), (.1,  2), (.1,  2), (.1,  5), (.1, 20)]
    n_params = len(p_names)

    def __init__(self, env):
        super().__init__(env)
        self.define_features()

    def load_params(self, params: list):
        self.lmbda = params[0]
        self.gamma = params[1]
        self.theta = params[2]
        self.delta = params[3]
        self.C     = params[4]
        self.w_ce  = params[5]
        self.w_c2  = params[6]
        self.w_u2  = params[7]
        self.w_c3  = params[8]
        self.w_c4  = params[9]
    
    def define_features(self):
        self.features = [
            'connected_2_feature', 
            'unconnected_2_feature', 
            'connected_3_feature', 
            'connected_4_feature'
        ]

    def get_action(self, state: tuple):
        '''greedy policy based on heuristic evaluation

        Inputs:
            state: a tuple (board, player_id)
        
        Outputs:
            action: a tuple (row, col)
        '''
        self.player_id  = state[1]
        self.opponent_id = 1-self.player_id
        # build the root node
        root = Node(state=state, 
                    parent=None, 
                    heuristic_fn=self.heuristic,
                    depth=0)
        # expand all root
        self.expand_node(root)
        # choose the best action based on the minimax algorithm
        return self.minmax(root).action
    
    def expand_node(self, node):
        state = node.state
        valid_actions = self.env.get_valid_actions(state[0])
        if len(valid_actions)>0:
            # expand the node throught breadth first search
            for action in valid_actions:
                # this deepcopy is important
                # as the transit function will modify the state
                # we need to keep the original state for the parent node
                # otherwise, the state will be modified
                state_next, _, done, info = self.env.transit(deepcopy(state), action)
                child_node = Node(
                    state=state_next, 
                    action=action, 
                    parent=node, 
                    heuristic_fn=self.heuristic,
                    depth=node.depth+1
                )
                node.children.append(child_node)
            # get the max value of the children
            max_value = self.minmax(node).value
            # prune the low value children
            node.children = [child for child in node.children if np.abs(child.value-max_value)<=self.theta]
    
    def minmax(self, node: Node):
        '''Minimax algorithm

        Choose the best action for the current player
        based on the minimax algorithm. If the player's 
        turn, the function returns the child with the
        highest value. If the opponent's turn, the function
        returns the child with the lowest value.

        Inputs:
            node: a Node object

        Outputs:
            best_node: a Node object
        '''
        # if the node is a leaf node, return the node
        if len(node.children) == 0: return node
        # if it is the player's turn, choose the child with the highest value
        if self.player_id==node.state[1]:
            best_node, best_value = None, -np.inf
            for child in node.children:
                if child.value > best_value:
                    best_node, best_value = child, child.value
            return best_node
        # if it is the opponent's turn, choose the child with the lowest value
        else:
            best_node, best_value = None, np.inf
            for child in node.children:
                if child.value < best_value:
                    best_node, best_value = child, child.value
            return best_node
    
    def heuristic(self, state: tuple, verbose=False):
        '''Heuristic evaluation

        The heuristic evaluation function for the van plan agent.

        The evaluation considers five features:
            1) value_center: whose pieces are closer to the center
            2) value_c2: connected 2-in-a-row features:  - x x -
            3) value_u2: unconnected 2-in-a-row features: x - x -
            4) value_c3: connected 3-in-a-row features: - x x x -
            5) value_c4: connected 4-in-a-row features: x x x x

        The heuristic value is the weighted sum of these features:

            value = w_center * value_center 
                        
                        + w_c2 * value_c2(self.player_id) 
                        + w_u2 * value_u2(self.player_id) 
                        + w_c3 * value_c3(self.player_id) 
                        + w_c4 * value_c4(self.player_id)

                        - w_c2 * value_c2(self.opponent_id) 
                        - w_u2 * value_u2(self.opponent_id) 
                        - w_c3 * value_c2(self.opponent_id) 
                        - w_c4 * value_c2(self.opponent_id)

                        + N(0, 1)

        Inputs:
            state: tuple (board, player_id)

        Outputs:
            value: float
        '''
        # get the board and player id 
        board, id_to_move = state
        player_pieces = np.vstack(np.where(board==self.player_id)).T
        C_player = 1 if id_to_move==self.player_id else self.C
        # the opponent id 
        opponent_id = int(1-self.player_id)
        opponent_pieces = np.vstack(np.where(board==opponent_id)).T
        C_opponent = 1 if id_to_move==self.opponent_id else self.C
        
        # get the center value
        value_center = self.get_center_value(self.env.center, 
                                             player_pieces, 
                                             opponent_pieces)
        
        # get the connected 2-in-a-row value
        feature_c2 = 0
        if 'connected_2_feature' in self.features:
            if verbose: print('\n connected 2-in-a-row:')
            if verbose: print('player:')
            f_c2_player = self.get_connected_2_feature(board, player_pieces, verbose=verbose)
            if verbose: print('opponent:')
            f_c2_opponent = self.get_connected_2_feature(board, opponent_pieces, verbose=verbose)
            feature_c2 = C_player*f_c2_player - C_opponent*f_c2_opponent
        
        # get the unconnected 2-in-a-row value
        feature_u2 = 0
        if 'unconnected_2_feature' in self.features:
            if verbose: print('\n unconnected 2-in-a-row:')
            if verbose: print('player:')
            f_u2_player = self.get_unconnected_2_feature(board, player_pieces, verbose=verbose)
            if verbose: print('opponent:')
            f_u2_opponent = self.get_unconnected_2_feature(board, opponent_pieces, verbose=verbose)
            feature_u2 = C_player*f_u2_player - C_opponent*f_u2_opponent
        
        # get the connected 3-in-a-row value
        feature_c3 = 0
        if 'connected_3_feature' in self.features:
            if verbose: print('\n connected 3-in-a-row:')
            if verbose: print('player:')
            f_c3_player = self.get_connected_3_feature(board, player_pieces, verbose=verbose)
            if verbose: print('opponent:')
            f_c3_opponent = self.get_connected_3_feature(board, opponent_pieces, verbose=verbose)
            feature_c3 = C_player*f_c3_player - C_opponent*f_c3_opponent
            
        # get the connected 4-in-a-row value
        feature_c4 = 0
        if 'connected_4_feature' in self.features:
            if verbose: print('\n connected 4-in-a-row:')
            if verbose: print('player:')
            f_c4_player = self.get_connected_4_feature(board, player_pieces, verbose=verbose)
            if verbose: print('opponent:')
            f_c4_opponent = self.get_connected_4_feature(board, opponent_pieces, verbose=verbose)
            feature_c4 = C_player*f_c4_player - C_opponent*f_c4_opponent

        # combine the value   
        noise = np.random.randn() 
        value = self.w_ce * value_center \
                    + self.w_c2 * feature_c2 \
                    + self.w_u2 * feature_u2 \
                    + self.w_c3 * feature_c3 \
                    + self.w_c4 * feature_c4 \
                    + noise
        
        return value
    
    @staticmethod
    def get_center_value(center, player_pieces, opponent_pieces):
        '''Estimate whose pieces are closer to the center

         V = sum_{i in player_pieces} 1/||i-center_coord|| 
                - sum_{j in opponent_pieces} 1/||j-center_coord||

        Inputs:
            state: tuple (board, player_id)

        Outputs:
            value: float
        '''
        player_dists = np.linalg.norm(player_pieces-center, axis=1)
        opponent_dists = np.linalg.norm(opponent_pieces-center, axis=1)
        return (1/player_dists).sum() - (1/opponent_dists).sum()
    
    @staticmethod
    def get_connected_2_feature(board, player_pieces, not_occupied=.75, verbose=False):
        '''Calculate the number of connected 2-in-a-row features
        
        Consider these three patterns:
        - xx-- : two connected pieces followed by two empty spaces
        - -xx- : empty space, two connected pieces, empty space
        - --xx : two empty spaces followed by two connected pieces
        
        Check in four directions: horizontal, vertical, diagonal1, diagonal2.
        
        Inputs:
            board: numpy array
            player_pieces: numpy array, the coordinates of the player's pieces
            not_occupied: value representing empty spaces (default: .75)
            verbose: bool, whether to print debug information
            
        Outputs:
            count: number of connected 2-in-a-row features
        '''
        if player_pieces is None or len(player_pieces) < 2:
            return 0
        
        rows, cols = board.shape
        count = 0
        
        # Directions to check: horizontal, vertical, diagonal1, diagonal2
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        direction_names = {(0, 1): 'horizontal', 
                           (1, 0): 'vertical', 
                           (1, 1): 'diagonal1', 
                           (1,-1): 'diagonal2'}
        
        # Create a set of player piece coordinates for faster lookup
        piece_set = set(map(tuple, player_pieces))
        player_color = int(board[player_pieces[0][0], player_pieces[0][1]])
        opponent_color = 1-player_color
        
        # To avoid double counting
        counted_features = set()
        
        # return 1 if a connected 2-in-a-row is found 
        # for each location (r,c) with orientation (dr, dc)
        # and 0 otherwise 
        for r, c in player_pieces:
            for dr, dc in directions:
                # Check if there's a connected piece next to current piece
                next_r, next_c = r + dr, c + dc
                if (next_r, next_c) not in piece_set:
                    continue
                    
                # Get coordinates for checking patterns
                prev_r, prev_c = r - dr, c - dc
                next2_r, next2_c = next_r + dr, next_c + dc
                next3_r, next3_c = next2_r + dr, next2_c + dc
                prev2_r, prev2_c = prev_r - dr, prev_c - dc
                
                # Check if coordinates are within bounds
                next2_valid = 0 <= next2_r < rows and 0 <= next2_c < cols
                next3_valid = 0 <= next3_r < rows and 0 <= next3_c < cols
                prev_valid = 0 <= prev_r < rows and 0 <= prev_c < cols
                prev2_valid = 0 <= prev2_r < rows and 0 <= prev2_c < cols
                
                # Create feature key for deduplication
                feature_key = None
                pattern = None
                
                # Check -xx- pattern
                if (prev_valid and next2_valid and
                      board[prev_r, prev_c] == not_occupied and
                      board[next2_r, next2_c] == not_occupied):
                    feature_key = (r, c, dr, dc)
                    pattern = "-xx-"
                    if verbose:
                        print(f'{r}, {c}: {direction_names[(dr, dc)]}, pattern: {pattern}')
                    count += 1
                
                # Check xx-- pattern
                if (next2_valid and next3_valid and
                    board[next2_r, next2_c] == not_occupied and
                    board[next3_r, next3_c] != opponent_color and
                    (not prev_valid or board[prev_r, prev_c] != player_color)):
                    feature_key = (r, c, dr, dc)
                    pattern = "xx--"
                    if verbose:
                        print(f'{r}, {c}: {direction_names[(dr, dc)]}, pattern: {pattern}')
                    count += 1
                
                # Check --xx pattern
                if (prev_valid and prev2_valid and
                      board[prev_r, prev_c] == not_occupied and
                      board[prev2_r, prev2_c] != opponent_color and
                      (not next2_valid or board[next2_r, next2_c] != player_color)):
                    feature_key = (prev2_r, prev2_c, dr, dc)
                    pattern = "--xx"
                    if verbose:
                        print(f'{r}, {c}: {direction_names[(dr, dc)]}, pattern: {pattern}')
                    count += 1
                
                # # If we found a pattern and haven't counted it yet
                # if feature_key and feature_key not in counted_features:
                #     counted_features.add(feature_key)
                #     if verbose:
                #         print(f'{r}, {c}: {direction_names[(dr, dc)]}, pattern: {pattern}')
                #     count += 1
        
        return count
    
    @staticmethod
    def get_unconnected_2_feature(board, player_pieces, not_occupied=.75, verbose=False):
        '''Calculate the number of unconnected 2-in-a-row features
        
        Consider these two patterns:
        - x-x- : piece, empty, piece, empty
        - -x-x : empty, piece, empty, piece
        
        If both patterns occur at the same position and direction, count only once.
        Check in four directions: horizontal, vertical, diagonal1, diagonal2.
        
        Inputs:
            board: numpy array
            player_pieces: numpy array, the coordinates of the player's pieces
            not_occupied: value representing empty spaces (default: .75)
            verbose: bool, whether to print debug information
            
        Outputs:
            count: number of unconnected 2-in-a-row features
        '''
        if player_pieces is None or len(player_pieces) < 2:
            return 0
        
        rows, cols = board.shape
        count = 0
        
        # Directions to check: horizontal, vertical, diagonal1, diagonal2
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        direction_names = {(0, 1): 'horizontal', 
                           (1, 0): 'vertical', 
                           (1, 1): 'diagonal1', 
                           (1,-1): 'diagonal2'}
        
        # Create a set of player piece coordinates for faster lookup
        piece_set = set(map(tuple, player_pieces))
        player_color = board[player_pieces[0][0], player_pieces[0][1]]
        
        # To avoid double counting
        counted_features = set()
        
        # return 1 if a unconnected 2-in-a-row is found 
        # for each location (r,c) with orientation (dr, dc)
        # and 0 otherwise 
        for r, c in player_pieces:
            for dr, dc in directions:
                # Get coordinates for checking patterns
                next_r, next_c = r + dr, c + dc
                next2_r, next2_c = next_r + dr, next_c + dc
                next3_r, next3_c = next2_r + dr, next2_c + dc
                prev_r, prev_c = r - dr, c - dc
                
                # Check if coordinates are within bounds
                next_valid = 0 <= next_r < rows and 0 <= next_c < cols
                next2_valid = 0 <= next2_r < rows and 0 <= next2_c < cols
                next3_valid = 0 <= next3_r < rows and 0 <= next3_c < cols
                prev_valid = 0 <= prev_r < rows and 0 <= prev_c < cols
                
                # Create feature key for deduplication
                feature_key = None
                pattern = None
                
                # Check x-x- pattern
                if (next_valid and next2_valid and
                    board[next_r, next_c] == not_occupied and
                    (next2_r, next2_c) in piece_set and
                    (next3_valid and board[next3_r, next3_c] == not_occupied) and
                    (not prev_valid or board[prev_r, prev_c] != player_color)):
                    feature_key = (r, c, dr, dc)
                    pattern = "x-x-"
                    if verbose:
                        print(f'{r}, {c}: {direction_names[(dr, dc)]}, pattern: {pattern}')
                    count += 1
                
                # Check -x-x pattern
                if (prev_valid and next_valid and
                      board[prev_r, prev_c] == not_occupied and
                      board[next_r, next_c] == not_occupied and
                      (next2_r, next2_c) in piece_set and
                      (not next3_valid or board[next3_r, next3_c] != player_color)):
                    feature_key = (r, c, dr, dc)
                    pattern = "-x-x"
                    if verbose:
                        print(f'{r}, {c}: {direction_names[(dr, dc)]}, pattern: {pattern}')
                    count += 1
                
                # # If we found a pattern and haven't counted it yet
                # if feature_key and feature_key not in counted_features:
                #     counted_features.add(feature_key)
                #     if verbose:
                #         print(f'{r}, {c}: {direction_names[(dr, dc)]}, pattern: {pattern}')
                #     count += 1
        
        return count
    
    @staticmethod
    def get_connected_3_feature(board, player_pieces, not_occupied=.75, verbose=False):
        '''Calculate the number of connected 3-in-a-row features
        
        Consider these two patterns:
        - xxx- : three connected pieces followed by empty space
        - -xxx : empty space followed by three connected pieces
        
        If both patterns occur at the same position and direction, count only once.
        Check in four directions: horizontal, vertical, diagonal1, diagonal2.
        
        Inputs:
            board: numpy array
            player_pieces: numpy array, the coordinates of the player's pieces
            not_occupied: value representing empty spaces (default: .75)
            verbose: bool, whether to print debug information
            
        Outputs:
            count: number of connected 3-in-a-row features
        '''
        if player_pieces is None or len(player_pieces) < 3:
            return 0
        
        rows, cols = board.shape
        count = 0
        
        # Directions to check: horizontal, vertical, diagonal1, diagonal2
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        direction_names = {(0, 1): 'horizontal', 
                           (1, 0): 'vertical', 
                           (1, 1): 'diagonal1', 
                           (1,-1): 'diagonal2'}
        
        # Create a set of player piece coordinates for faster lookup
        piece_set = set(map(tuple, player_pieces))
        player_color = int(board[player_pieces[0][0], player_pieces[0][1]])
        opponent_color = 1-player_color
        
        # To avoid double counting
        counted_features = set()
        
        # return 1 if a connected 3-in-a-row is found 
        # for each location (r,c) with orientation (dr, dc)
        # and 0 otherwise 
        for r, c in player_pieces:
            for dr, dc in directions:
                # Get coordinates for checking patterns
                next_r, next_c = r + dr, c + dc
                next2_r, next2_c = next_r + dr, next_c + dc
                next3_r, next3_c = next2_r + dr, next2_c + dc
                prev_r, prev_c = r - dr, c - dc
                
                # Check if we have three connected pieces
                if not ((next_r, next_c) in piece_set and 
                       (next2_r, next2_c) in piece_set):
                    continue
                    
                # Check if coordinates are within bounds
                next3_valid = 0 <= next3_r < rows and 0 <= next3_c < cols
                prev_valid = 0 <= prev_r < rows and 0 <= prev_c < cols
                
                # Create feature key for deduplication
                feature_key = None
                pattern = None
                
                # Check xxx- pattern
                if (next3_valid and 
                    board[next3_r, next3_c] != opponent_color and
                    (not prev_valid or board[prev_r, prev_c] != player_color)):
                    feature_key = (r, c, dr, dc)
                    pattern = "xxx-"
                    if verbose:
                        print(f'{r}, {c}: {direction_names[(dr, dc)]}, pattern: {pattern}')
                    count += 1
                
                # Check -xxx pattern
                if (prev_valid and 
                      board[prev_r, prev_c] != opponent_color and
                      (not next3_valid or board[next3_r, next3_c] != player_color)):
                    feature_key = (r, c, dr, dc)
                    pattern = "-xxx"
                    if verbose:
                        print(f'{r}, {c}: {direction_names[(dr, dc)]}, pattern: {pattern}')
                    count += 1
                
                # # If we found a pattern and haven't counted it yet
                # if feature_key and feature_key not in counted_features:
                #     counted_features.add(feature_key)
                #     if verbose:
                #         print(f'{r}, {c}: {direction_names[(dr, dc)]}, pattern: {pattern}')
                #     count += 1
        
        return count

    @staticmethod
    def get_connected_4_feature(board, player_pieces, not_occupied=.75, verbose=False):
        '''Calculate the number of connected 4-in-a-row features
        
        Look for pattern xxxx: four connected pieces in a row.
        Check in four directions: horizontal, vertical, diagonal1, diagonal2.
        
        Inputs:
            board: numpy array
            player_pieces: numpy array, the coordinates of the player's pieces
            not_occupied: value representing empty spaces (default: .75)
            verbose: bool, whether to print debug information
            
        Outputs:
            count: number of connected 4-in-a-row features
        '''
        if player_pieces is None or len(player_pieces) < 4:
            return 0
        
        rows, cols = board.shape
        count = 0
        
        # Directions to check: horizontal, vertical, diagonal1, diagonal2
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        direction_names = {(0, 1): 'horizontal', 
                           (1, 0): 'vertical', 
                           (1, 1): 'diagonal1', 
                           (1,-1): 'diagonal2'}
        
        # Create a set of player piece coordinates for faster lookup
        piece_set = set(map(tuple, player_pieces))
        player_color = int(board[player_pieces[0][0], player_pieces[0][1]])
        
        # To avoid double counting
        counted_features = set()
        
        # return 1 if a connected 4-in-a-row is found 
        # for each location (r,c) with orientation (dr, dc)
        # and 0 otherwise 
        for r, c in player_pieces:
            for dr, dc in directions:
                # Get coordinates for checking patterns
                next_r, next_c = r + dr, c + dc
                next2_r, next2_c = next_r + dr, next_c + dc
                next3_r, next3_c = next2_r + dr, next2_c + dc
                prev_r, prev_c = r - dr, c - dc
                
                # Check if we have four connected pieces
                if not ((next_r, next_c) in piece_set and 
                       (next2_r, next2_c) in piece_set and 
                       (next3_r, next3_c) in piece_set):
                    continue
                    
                # Check if coordinates are within bounds
                prev_valid = 0 <= prev_r < rows and 0 <= prev_c < cols
                
                # Create feature key for deduplication
                feature_key = None
                
                # Check xxxx pattern
                if not prev_valid or board[prev_r, prev_c] != player_color:
                    feature_key = (r, c, dr, dc)
                
                # If we found a pattern and haven't counted it yet
                if feature_key and feature_key not in counted_features:
                    counted_features.add(feature_key)
                    if verbose:
                        print(f'{r}, {c}: {direction_names[(dr, dc)]}, pattern: xxxx')
                    count += 1
        
        return count

# ------------ Basic plan agents ------------ #

class BFS_agent(heuristic_agent):
    '''Planning Agent from Van Opheusden et al. (2023)
    '''

    @staticmethod
    def show_node(node, not_occupied=0.75):
        board, player_id = node.state
        player_color = 'black' if player_id==0 else 'white'
        value = node.value
        rows, cols = board.shape
        empty_board = np.ones([rows, cols])*not_occupied
        # Draw filled scatter points on non-empty grids
        fig = plt.figure(figsize=(4, 2))
        sns.heatmap(empty_board, cmap='gray', vmin=0, vmax=1, 
                    lw=.5, linecolor='black', square=True, cbar=False)
        plt.title(f'V: {value:.2f}, Turn: {player_color}')
        for i in range(rows):
            for j in range(cols):
                if board[i, j] != not_occupied:
                    color = board[i,j]*np.ones([3])
                    last_action = node.action
                    edge_color = color
                    if (last_action is not None) and last_action==(i, j): 
                        edge_color = 'red'
                    plt.scatter(j+.5, i+.5, s=300, color=color, 
                                edgecolor=edge_color, linewidth=1)
        plt.axis('off')
    
        # This will block until the window is closed
        plt.show(block=True)

    def get_action(self, state):
        # assign player id
        self.player_id  = state[1]
        self.opponent_id = 1-self.player_id
        # randomly pick an action if lapse
        if self.lapse(self.lmbda):
            # get the valid actions
            valid_actions = self.env.get_valid_actions(state[0])
            idx = np.random.choice(len(valid_actions))
            return int(valid_actions[idx][0]), int(valid_actions[idx][1])
        # drop a feature
        self.drop_feature(self.delta)
        # construct the root node 
        root = Node(
            state=deepcopy(state),
            action=None, 
            parent=None, 
            heuristic_fn=self.heuristic,
            depth=0
        )
        # grow the tree
        node = root
        stop_search = False
        self.root_actions = []
        while not stop_search:  
            # select the node
            node = self.select_node(root)
            # check if the state is terminated
            self.expand_node(node)
            # backpropagate the value
            self.backpropagate(node)
            # check if the search should stop
            determined = self.determine(root)
            if self.stop(self.gamma) or determined: stop_search = True
        action = self.minmax(root).action
        return (int(action[0]), int(action[1]))
    
    # ------------ aux functions ------------- #
    
    def lapse(self, lmbda):
        return np.random.rand() < lmbda
    
    def drop_feature(self, delta):
        '''Drop a feature from the heuristic evaluation

        Each feature as a delta probability being dropped.
        - connected 2-in-a-row
        - unconnected 2-in-a-row
        - connected 3-in-a-row
        - connected 4-in-a-row
        
        Inputs:
            delta: float, the probability of dropping a feature
            
        '''
        features = ['connected_2_feature', 
                    'unconnected_2_feature', 
                    'connected_3_feature', 
                    'connected_4_feature']
        self.features = [f for f in features if np.random.rand() > delta]

    def stop(self, gamma):
        return np.random.rand() < gamma
    
    def determine(self, root):
        root_action = self.minmax(root).action
        self.root_actions.append(root_action)
        if len(self.root_actions) >= 50:
            last_50_actions=self.root_actions[-50:]
            if all(action==last_50_actions[0] for action in last_50_actions):
                return True
        return False
        
    def select_node(self, root):
        node = root
        while len(node.children)>0:
            node = self.minmax(node)
        return node
    
    def minmax(self, node: Node):
        '''Minimax algorithm

        Choose the best action for the current player
        based on the minimax algorithm. If the player's 
        turn, the function returns the child with the
        highest value. If the opponent's turn, the function
        returns the child with the lowest value.

        Inputs:
            node: a Node object

        Outputs:
            best_node: a Node object
        '''
        # if the node is a leaf node, return the node
        if len(node.children) == 0: return node
        # if it is the player's turn, choose the child with the highest value
        if self.player_id==node.state[1]:
            best_node, best_value = None, -np.inf
            for child in node.children:
                if child.value > best_value:
                    best_node, best_value = child, child.value
            return best_node
        # if it is the opponent's turn, choose the child with the lowest value
        else:
            best_node, best_value = None, np.inf
            for child in node.children:
                if child.value < best_value:
                    best_node, best_value = child, child.value
            return best_node
        
    def expand_node(self, node: Node):
        '''Expand the node through breadth first search'''
        state = node.state
        valid_actions = self.env.get_valid_actions(state[0])
        if len(valid_actions)>0:
            # expand the node throught breadth first search
            for action in valid_actions:
                # this deepcopy is important
                # as the transit function will modify the state
                # we need to keep the original state for the parent node
                # otherwise, the state will be modified
                state_next, _, done, info = self.env.transit(deepcopy(state), action)
                child_node = Node(
                    state=state_next, 
                    action=action, 
                    parent=node, 
                    heuristic_fn=self.heuristic,
                    depth=node.depth+1
                )
                node.children.append(child_node)
            # get the max value of the children
            max_value = self.minmax(node).value
            # prune the low value children
            node.children = [child for child in node.children if np.abs(child.value-max_value)<=self.theta]

    def backpropagate(self, node):
        '''Backpropagate the value of the node
        if the selected child is a termination node,
        the value is set to 10000 if the player to move is the current player,
        -10000 otherwise. If the child is not a termination node, the value
        is set to the value of the child.
        '''
        # update the value of the node 
        node.value = self.minmax(node).value

        # backpropagate the value to the parent node
        if node.parent is not None:
            self.backpropagate(node.parent)

