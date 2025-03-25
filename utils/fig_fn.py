import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def layout2board(layout):
    return np.array([[0.75 if c == '.' else 0 if c == '0' else 1. for c in row] for row in layout])

def board2layout(board):
    return [''.join(['.' if x==0.75 else '0' if x==0 else '1' for x in row]) for row in board]

def show_board(board, p_actions={}, not_occupied=0.75):
    rows, cols = board.shape
    empty_board = np.ones([rows, cols])*not_occupied
    # Draw filled scatter points on non-empty grids
    plt.figure(figsize=(4, 2))
    sns.heatmap(empty_board, cmap='gray', vmin=0, vmax=1, 
                lw=.5, linecolor='black', square=True, cbar=False)
    n_sim = sum(p_actions.values())
    for action, p in p_actions.items():
        plt.scatter(action[1]+.5, action[0]+.5, s=550, 
                    linewidth=0,
                    marker='s', color='red', alpha=p/n_sim)
    for i in range(rows):
        for j in range(cols):
            if board[i, j] != not_occupied:
                color = board[i,j]*np.ones([3])
                plt.scatter(j+.5, i+.5, s=300, color=color)
    plt.axis('off')
    plt.show()