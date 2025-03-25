import numpy as np
from utils.env_fn import four_in_a_row
from utils.model import heuristic_agent, accelerated_heuristic, Node
    
from copy import deepcopy
import time

def test_model_equivalence():
    """Test if accelerated_heuristic produces identical results to heuristic_agent"""
    
    # Initialize environment and both agents
    env = four_in_a_row()
    old_agent = heuristic_agent(env)
    new_agent = accelerated_heuristic(env)
    
    # Set same parameters for both agents
    params = [.05, .02, 2, .2, 1.2, .8, 1, .4, 3.5, 10]
    old_agent.load_params(params)
    new_agent.load_params(params)
    
    # Timing variables
    old_heuristic_time = 0
    new_heuristic_time = 0
    old_action_time = 0
    new_action_time = 0
    
    def compare_heuristics(board, player_id):
        nonlocal old_heuristic_time, new_heuristic_time
        
        state = (deepcopy(board), player_id)
        old_agent.player_id = player_id
        new_agent.player_id = player_id
        old_agent.opponent_id = 1 - player_id
        new_agent.opponent_id = 1 - player_id
        
        # Time old agent
        np.random.seed(0)
        start = time.perf_counter()
        old_value = old_agent.heuristic(deepcopy(state), verbose=False)
        old_heuristic_time += time.perf_counter() - start
        
        # Time new agent
        np.random.seed(0)
        start = time.perf_counter()
        new_value = new_agent.heuristic(deepcopy(state), verbose=False)
        new_heuristic_time += time.perf_counter() - start
        
        return np.isclose(old_value, new_value, rtol=1e-6)
    
    def compare_actions(board, player_id):
        nonlocal old_action_time, new_action_time
        
        state = (deepcopy(board), player_id)
        
        # Time old agent
        np.random.seed(0)
        start = time.perf_counter()
        old_action = old_agent.get_action(deepcopy(state))
        old_action_time += time.perf_counter() - start
        
        # Time new agent
        np.random.seed(0)
        start = time.perf_counter()
        new_action = new_agent.get_action(deepcopy(state))
        new_action_time += time.perf_counter() - start
        
        equal = old_action == new_action
        if not equal:
            print(f"\nActions differ: Old={old_action}, New={new_action}")
        return equal
    
    # ... existing test_boards definition ...
    
    # Run tests
    total_tests = len(test_boards) * 2  # For both players
    for i, board in enumerate(test_boards):
        for player_id in [0, 1]:
            heuristic_equal = compare_heuristics(board, player_id)
            action_equal = compare_actions(board, player_id)
            
            if not (heuristic_equal and action_equal):
                print(f"Test failed for board {i}, player {player_id}")
                print(f"Heuristic equal: {heuristic_equal}")
                print(f"Action equal: {action_equal}")
                return False
    
    # Print timing results
    print("\nTiming Results:")
    print(f"Heuristic Evaluation:")
    print(f"  Original: {old_heuristic_time:.6f} seconds total, {old_heuristic_time/total_tests:.6f} seconds per test")
    print(f"  Accelerated: {new_heuristic_time:.6f} seconds total, {new_heuristic_time/total_tests:.6f} seconds per test")
    print(f"  Speedup: {old_heuristic_time/new_heuristic_time:.2f}x")
    
    print(f"\nAction Selection:")
    print(f"  Original: {old_action_time:.6f} seconds total, {old_action_time/total_tests:.6f} seconds per test")
    print(f"  Accelerated: {new_action_time:.6f} seconds total, {new_action_time/total_tests:.6f} seconds per test")
    print(f"  Speedup: {old_action_time/new_action_time:.2f}x")
    
    print("\nAll tests passed! Models are functionally equivalent.")
    return True

# Run the test
if __name__ == "__main__":
    test_model_equivalence() 