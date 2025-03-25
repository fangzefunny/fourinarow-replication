# Minimal AI server that should work with basic Flask
from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/get_move', methods=['POST'])
def get_move():
    print("Received move request")
    try:
        # Get the data from the request
        data = request.get_json()
        print(f"Received data: {data}")
        
        # Extract board and player
        board = data.get('board', [])
        player = data.get('player', 0)
        
        print(f"Board: {board}, Player: {player}")
        
        # Simple random AI - find all empty cells
        empty_cells = []
        
        # Assuming board is a list of strings with '.', '0', '1'
        for r, row_str in enumerate(board):
            for c, cell in enumerate(row_str):
                if cell == '.':
                    empty_cells.append((r, c))
        
        print(f"Found {len(empty_cells)} empty cells")
        
        # Choose a random empty cell
        if empty_cells:
            row, col = random.choice(empty_cells)
            print(f"Selected move: {row}, {col}")
            return jsonify({'move': [row, col]})
        else:
            print("No valid moves")
            return jsonify({'move': [None, None]})
            
    except Exception as e:
        print(f"Error: {str(e)}")
        # Return a safe fallback
        return jsonify({'move': [0, 0], 'error': str(e)})

@app.route('/ping', methods=['GET'])
def ping():
    """Simple endpoint to check if server is running"""
    return jsonify({"status": "ok", "message": "AI server is running"})

# Very simple route for testing
@app.route('/', methods=['GET'])
def home():
    return "AI Server is running. Use /get_move with POST or /ping with GET."

if __name__ == '__main__':
    print("Starting Flask server on port 5000...")
    # Use host='0.0.0.0' to make it accessible from other devices on network
    app.run(debug=True, port=5000, host='0.0.0.0')