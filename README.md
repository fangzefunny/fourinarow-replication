# fourinarow-replication

This project is for self-tutorial purposes and aims to replicate some results from the paper:
Van Opheusden, B., Kuperwajs, I., Galbiati, G., Bnaya, Z., Li, Y., and Ma, W. J. (2023). "Expertise increases planning depth in human gameplay." Nature, 618(7967), 1000-1005.

## Project Structure
Here are the important files:
* `utils/env_fn.py`: Implements the Four-in-a-Row game environment
* `utils/model.py`: Contains the two main models (heuristic agent and BFS agent)
* `play.py`: Allows interaction with the AI agent
* `demo/`: Four Jupyter notebook files that replicate key results from the original paper
* `webapp/`: Website implementation of the Four-in-a-Row game. Run `index.html` with web a browser (recommand Chrome)
  
## Implementation Notes

The Python BFS agent is designed for clarity but suffers from performance limitations. While easy to understand, its running speed make it impractical for fitting to any behavioral datasets in its current implementation. 


