// Tic-tac-toe variant implementation (4x9 board with 4-in-a-row win condition)
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM fully loaded');
    
    // Game constants
    const ROWS = 4;
    const COLS = 9;
    const CELL_SIZE = 80;
    const EMPTY = null;
    const BLACK = 0;
    const WHITE = 1;
    
    // Game state
    let board = [];
    let currentPlayer = BLACK;
    let gameActive = false;
    let gameData = {
        board: [],
        play_to_move: [],
        action: [],
        done: [],
        winner: [],
        trial: [],
        time_elapsed: [],
        rt: []
    };
    let blockId = 0;
    let blockStartTime = 0;
    let statePresTime = 0;
    let trial = 0;
    let winner = null;
    
    // DOM elements
    const canvas = document.getElementById('game-board');
    const ctx = canvas.getContext('2d');
    const startBtn = document.getElementById('start-game');
    const downloadBtn = document.getElementById('download-data');
    const statusMessage = document.getElementById('status-message');
    const gameModeSelect = document.getElementById('game-mode');
    
    // Initialize listeners
    startBtn.addEventListener('click', startGame);
    downloadBtn.addEventListener('click', downloadGameData);
    canvas.addEventListener('click', handleCanvasClick);
    gameModeSelect.addEventListener('change', updateGameOptions);
    
    console.log('Event listeners attached');
    
    // Set canvas size
    canvas.width = COLS * CELL_SIZE;
    canvas.height = ROWS * CELL_SIZE;
    
    // Draw empty board initially
    drawEmptyBoard();
    
    // Game functions
    function startGame() {
        console.log('Starting game');
        
        // Reset game state
        board = Array(ROWS).fill().map(() => Array(COLS).fill(EMPTY));
        currentPlayer = BLACK;
        gameActive = true;
        winner = null;
        trial = 0;
        
        // Reset data collection
        gameData = {
            board: [],
            play_to_move: [],
            action: [],
            done: [],
            winner: [],
            trial: [],
            time_elapsed: [],
            rt: []
        };
        
        // Get block ID from input
        const blockIdInput = document.getElementById('block-id');
        blockId = parseInt(blockIdInput.value) || 0;
        
        // Set timestamps
        blockStartTime = Date.now();
        statePresTime = Date.now();
        
        // Disable download button until we have data
        downloadBtn.disabled = true;
        
        // Draw board and update status
        drawBoard();
        updateStatus();
        
        console.log('Game started with board size:', ROWS, 'x', COLS);
    }
    
    function handleCanvasClick(event) {
        if (!gameActive) {
            console.log('Game not active');
            return;
        }
        
        // Get click position
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        // Calculate row and column
        const row = Math.floor(y / CELL_SIZE);
        const col = Math.floor(x / CELL_SIZE);
        
        // Make move at this position
        makeMove(row, col);
    }
    
    function makeMove(row, col) {
        if (row < 0 || row >= ROWS || col < 0 || col >= COLS || !gameActive) {
            console.log('Invalid move or game not active');
            return;
        }
        
        // Check if the cell is already occupied
        if (board[row][col] !== EMPTY) {
            console.log('Cell already occupied');
            return;
        }
        
        console.log(`Making move: Player ${currentPlayer} at row ${row}, col ${col}`);
        
        // Save current board state for data collection
        const currentBoard = JSON.parse(JSON.stringify(board));
        const currentPlayerBeforeMove = currentPlayer;
        
        // Make the move
        board[row][col] = currentPlayer;
        
        // Record response time
        const moveTime = Date.now();
        const rt = (moveTime - statePresTime) / 1000;
        const timeElapsed = (moveTime - blockStartTime) / 1000;
        
        // Check if the game is over
        const gameIsOver = checkGameEnd();
        
        // Collect data with the new board format
        gameData.board.push(convertBoardToString(currentBoard));
        gameData.play_to_move.push(currentPlayerBeforeMove);
        gameData.action.push(JSON.stringify([row, col]));
        gameData.done.push(gameIsOver);
        gameData.winner.push(gameIsOver && winner !== null ? winner : 'None');
        gameData.trial.push(trial++);
        gameData.time_elapsed.push(timeElapsed);
        gameData.rt.push(rt);
        
        // Update state for next move
        statePresTime = moveTime;
        
        // Enable download button
        downloadBtn.disabled = false;
        
        // Change player if game is not over
        if (!gameIsOver) {
            currentPlayer = 1 - currentPlayer;
        }
        
        // Check if AI should move
        if (gameActive && !gameIsOver && 
            currentPlayer === WHITE && 
            gameModeSelect.value === 'human_vs_ai') {
            setTimeout(makeAIMove, 500);
        }
        
        // Update the board and status
        drawBoard();
        updateStatus();
    }
    
    // Enhanced AI communication with debugging
    async function getAIPythonMove(boardState, currentPlayer) {
        console.log("Requesting AI move for:", convertBoardToString(boardState), "Player:", currentPlayer);
        statusMessage.textContent = "AI is thinking...";
        
        try {
            const apiUrl = 'http://localhost:5000/get_move';
            console.log(`Sending request to ${apiUrl}`);
            
            const payload = {
                board: JSON.parse(convertBoardToString(boardState)),
                player: currentPlayer
            };
            console.log("Sending data:", JSON.stringify(payload));
            
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(payload),
            });
            
            if (!response.ok) {
                console.error(`HTTP error! status: ${response.status}`);
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log("Received AI response:", data);
            
            if (!data.move || data.move.length !== 2) {
                console.error("Invalid move format received:", data);
                throw new Error("Invalid move format");
            }
            
            return data.move; // [row, col]
        } catch (error) {
            console.error('Error getting AI move:', error);
            statusMessage.textContent = "AI service error. Using random move.";
            // Fallback to random move if API fails
            return getRandomMove();
        }
    }
    
    // Improved makeAIMove function
    async function makeAIMove() {
        if (!gameActive || currentPlayer !== WHITE) return;
        
        console.log("AI turn started");
        statusMessage.textContent = "AI is thinking...";
        
        try {
            // Get move from Python AI
            const move = await getAIPythonMove(board, currentPlayer);
            console.log("AI selected move:", move);
            
            if (move && move.length === 2 && 
                move[0] >= 0 && move[0] < ROWS && 
                move[1] >= 0 && move[1] < COLS &&
                board[move[0]][move[1]] === EMPTY) {
                
                setTimeout(() => {
                    makeMove(move[0], move[1]);
                }, 500);
            } else {
                console.error("Invalid or illegal move returned from AI:", move);
                throw new Error("Invalid move returned from AI");
            }
        } catch (error) {
            console.error("Error in AI move:", error);
            statusMessage.textContent = "AI error. Using random move.";
            
            // Fallback to random move after a short delay
            setTimeout(() => {
                const randomMove = getRandomMove();
                if (randomMove) {
                    makeMove(randomMove[0], randomMove[1]);
                }
            }, 500);
        }
    }
    
    // Better random move generator
    function getRandomMove() {
        let emptyCells = [];
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                if (board[r][c] === EMPTY) {
                    emptyCells.push([r, c]);
                }
            }
        }
        
        if (emptyCells.length === 0) {
            console.log("No empty cells available for random move");
            return null;
        }
        
        const randomIndex = Math.floor(Math.random() * emptyCells.length);
        console.log(`Selected random move ${emptyCells[randomIndex]} from ${emptyCells.length} options`);
        return emptyCells[randomIndex];
    }
    
    function checkGameEnd() {
        // Check for a win
        if (checkWin()) {
            gameActive = false;
            winner = currentPlayer;
            console.log(`Player ${currentPlayer} wins!`);
            return true;
        }
        
        // Check for a draw
        if (checkDraw()) {
            gameActive = false;
            winner = null;
            console.log('Game ended in a draw');
            return true;
        }
        
        return false;
    }
    
    function checkWin() {
        // Directions: horizontal, vertical, diagonal //, diagonal \\
        const directions = [[0, 1], [1, 0], [1, 1], [1, -1]];
        
        for (let row = 0; row < ROWS; row++) {
            for (let col = 0; col < COLS; col++) {
                if (board[row][col] === EMPTY) continue;
                
                const player = board[row][col];
                
                for (const [dx, dy] of directions) {
                    let count = 1;
                    
                    // Check in the direction
                    for (let i = 1; i <= 3; i++) {
                        const newRow = row + i * dx;
                        const newCol = col + i * dy;
                        
                        if (newRow < 0 || newRow >= ROWS || newCol < 0 || newCol >= COLS) break;
                        if (board[newRow][newCol] !== player) break;
                        
                        count++;
                    }
                    
                    if (count >= 4) {
                        return true;
                    }
                }
            }
        }
        
        return false;
    }
    
    function checkDraw() {
        // Check if there are any empty cells left
        for (let row = 0; row < ROWS; row++) {
            for (let col = 0; col < COLS; col++) {
                if (board[row][col] === EMPTY) {
                    return false;
                }
            }
        }
        return true;
    }
    
    function drawEmptyBoard() {
        // Draw empty board
        ctx.fillStyle = '#f0f0f0';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw grid lines
        ctx.strokeStyle = '#333333';
        ctx.lineWidth = 2;
        
        // Draw vertical lines
        for (let col = 1; col < COLS; col++) {
            ctx.beginPath();
            ctx.moveTo(col * CELL_SIZE, 0);
            ctx.lineTo(col * CELL_SIZE, ROWS * CELL_SIZE);
            ctx.stroke();
        }
        
        // Draw horizontal lines
        for (let row = 1; row < ROWS; row++) {
            ctx.beginPath();
            ctx.moveTo(0, row * CELL_SIZE);
            ctx.lineTo(COLS * CELL_SIZE, row * CELL_SIZE);
            ctx.stroke();
        }
    }
    
    function drawBoard() {
        // Draw empty board with grid
        drawEmptyBoard();
        
        // Draw pieces
        for (let row = 0; row < ROWS; row++) {
            for (let col = 0; col < COLS; col++) {
                if (board[row][col] !== EMPTY) {
                    const x = col * CELL_SIZE + CELL_SIZE / 2;
                    const y = row * CELL_SIZE + CELL_SIZE / 2;
                    const radius = CELL_SIZE * 0.4;
                    
                    if (board[row][col] === BLACK) {
                        // Draw black filled circle
                        ctx.fillStyle = '#000000';
                        ctx.beginPath();
                        ctx.arc(x, y, radius, 0, Math.PI * 2);
                        ctx.fill();
                    } else {
                        // Draw white filled circle without border
                        ctx.fillStyle = '#FFFFFF';
                        ctx.beginPath();
                        ctx.arc(x, y, radius, 0, Math.PI * 2);
                        ctx.fill();
                        
                        // Border removed as requested
                    }
                }
            }
        }
    }
    
    function updateStatus() {
        if (!gameActive) {
            if (winner !== null) {
                statusMessage.textContent = `Game over! ${winner === BLACK ? 'Black' : 'White'} wins!`;
            } else {
                statusMessage.textContent = 'Game over! It\'s a draw!';
            }
        } else {
            statusMessage.textContent = `${currentPlayer === BLACK ? 'Black' : 'White'}'s turn`;
        }
    }
    
    function updateGameOptions() {
        const mode = gameModeSelect.value;
        const playerInfo = document.getElementById('player-info');
        
        if (mode === 'human_vs_human') {
            playerInfo.style.display = 'flex';
        } else {
            playerInfo.style.display = 'none';
        }
    }
    
    function downloadGameData() {
        if (gameData.board.length === 0) {
            alert('No game data to download');
            return;
        }
        
        // Get player names
        const blackPlayer = document.getElementById('black-player').value || 'Player1';
        const whitePlayer = document.getElementById('white-player').value || 'Player2';
        
        // Create CSV content
        let csv = 'board,play_to_move,action,done,winner,trial,time_elapsed,rt\n';
        
        for (let i = 0; i < gameData.board.length; i++) {
            csv += `${gameData.board[i]},`;
            csv += `${gameData.play_to_move[i]},`;
            csv += `${gameData.action[i]},`;
            csv += `${gameData.done[i]},`;
            csv += `${gameData.winner[i]},`;
            csv += `${gameData.trial[i]},`;
            csv += `${gameData.time_elapsed[i]},`;
            csv += `${gameData.rt[i]}\n`;
        }
        
        // Create download link
        const gameMode = gameModeSelect.value;
        const fileName = `${gameMode}-white=${whitePlayer}-black=${blackPlayer}-block=${blockId}.csv`;
        
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
    
    // Convert board to requested string format
    function convertBoardToString(boardData) {
        const result = [];
        
        for (let row = 0; row < ROWS; row++) {
            let rowStr = '';
            for (let col = 0; col < COLS; col++) {
                if (boardData[row][col] === EMPTY) {
                    rowStr += '.';
                } else if (boardData[row][col] === BLACK) {
                    rowStr += '0';
                } else {
                    rowStr += '1';
                }
            }
            result.push(rowStr);
        }
        
        return JSON.stringify(result);
    }
    
    // Initialize the UI
    updateGameOptions();
    console.log('Game initialized and ready to start');
}); 