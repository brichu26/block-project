// Basic game state
let gridSize = 30;
let board = [];
let mines = [];
let flags = [];
let gameWon = false;
let gameOver = false;
let numberOfMines = 100; // You can adjust this value
let currentSlide = 0;

// Function to initialize the game board
function initializeBoard(numberOfMines) {
    // Create the board array
    board = Array(gridSize).fill(null).map(() => Array(gridSize).fill(0));
    mines = [];
    flags = [];
    gameWon = false;
    gameOver = false;

    // Place mines randomly
    while (mines.length < numberOfMines) {
        const x = Math.floor(Math.random() * gridSize);
        const y = Math.floor(Math.random() * gridSize);
        if (!mines.some(mine => mine.x === x && mine.y === y)) {
            mines.push({ x, y });
            board[x][y] = 'mine'; // Use 'mine' string instead of boolean
        }
    }

    // Calculate numbers for adjacent mines
    for (let x = 0; x < gridSize; x++) {
        for (let y = 0; y < gridSize; y++) {
            if (board[x][y] !== 'mine') {
                let adjacentMines = 0;
                for (let i = -1; i <= 1; i++) {
                    for (let j = -1; j <= 1; j++) {
                        if (i === 0 && j === 0) continue;
                        const nx = x + i;
                        const ny = y + j;
                        if (nx >= 0 && nx < gridSize && ny >= 0 && ny < gridSize && board[nx][ny] === 'mine') {
                            adjacentMines++;
                        }
                    }
                }
                board[x][y] = adjacentMines;
            }
        }
    }
    return board;
}

// Function to create the HTML board dynamically
function createBoard() {
    const boardElement = document.getElementById('minesweeper-board');
    boardElement.innerHTML = ''; // Clear any existing board
    boardElement.style.gridTemplateColumns = `repeat(${gridSize}, 24px)`; // Update grid columns

    for (let x = 0; x < gridSize; x++) {
        for (let y = 0; y < gridSize; y++) {
            const cellElement = document.createElement('div');
            cellElement.classList.add('cell');
            cellElement.id = `cell-${x}-${y}`;
            cellElement.dataset.x = x;
            cellElement.dataset.y = y;

            // Add click event listener
            cellElement.addEventListener('click', () => handleCellClick(x, y));

            // Add right-click event listener
            cellElement.addEventListener('contextmenu', (event) => {
                handleRightClick(x, y, event);
            });

            boardElement.appendChild(cellElement);
        }
    }
}

function handleCellClick(x, y) {
    if (gameOver || gameWon) return;

    const cellElement = document.getElementById(`cell-${x}-${y}`);

    if (cellElement.classList.contains('flagged')) return; // Prevent clicking flagged cells

    if (cellElement.classList.contains('revealed')) {
        // If the cell is already revealed, try revealing neighbors if flag count matches
        revealNeighborsIfFlagCountMatches(x, y);
        return;
    }

    if (board[x][y] === 'mine') {
        revealAllMines();
        gameOver = true;
        alert("Game Over!");
        return;
    }

    revealCell(x, y);
    checkWinCondition();
}

function revealCell(x, y) {
    if (x < 0 || x >= gridSize || y < 0 || y >= gridSize) return;
    const cellElement = document.getElementById(`cell-${x}-${y}`);
    if (cellElement.classList.contains('revealed')) return;
    if (cellElement.classList.contains('flagged')) return;

    cellElement.classList.add('revealed');

    const value = board[x][y];
    cellElement.textContent = value > 0 ? value : '';

    if (value === 0) {
        for (let i = -1; i <= 1; i++) {
            for (let j = -1; j <= 1; j++) {
                if (i === 0 && j === 0) continue;
                revealCell(x + i, y + j); // Recursive call for flood fill
            }
        }
    } else {
      cellElement.classList.add(getNumberClass(value));
    }
}

function getNumberClass(number) {
    switch (number) {
        case 1: return 'one';
        case 2: return 'two';
        case 3: return 'three';
        case 4: return 'four';
        case 5: return 'five';
        case 6: return 'six';
        case 7: return 'seven';
        case 8: return 'eight';
        default: return '';
    }
}

function handleRightClick(x, y, event) {
    event.preventDefault(); // Prevent default context menu
    if (gameOver || gameWon) return;

    const cellElement = document.getElementById(`cell-${x}-${y}`);

    if (!cellElement.classList.contains('revealed')) {
        cellElement.classList.toggle('flagged');
    }
}

function revealAllMines() {
    for (const mine of mines) {
        const cellElement = document.getElementById(`cell-${mine.x}-${mine.y}`);
        cellElement.classList.add('mine');
    }
}

function checkWinCondition() {
    let revealedCount = 0;
    for (let x = 0; x < gridSize; x++) {
        for (let y = 0; y < gridSize; y++) {
            if (document.getElementById(`cell-${x}-${y}`).classList.contains('revealed')) {
                revealedCount++;
            }
        }
    }

    if (revealedCount === (gridSize * gridSize) - numberOfMines) {
        gameWon = true;
        alert("You Win!");
    }
}

// Reset Game function
function resetGame() {
  board = [];
  mines = [];
  flags = [];
  gameWon = false;
  gameOver = false;
  initializeBoard(numberOfMines);
  createBoard();
}

function setDifficulty(newGridSize, newNumberOfMines) {
    gridSize = newGridSize;
    numberOfMines = newNumberOfMines;
    createBoard();
    initializeBoard(numberOfMines);
}

function revealNeighborsIfFlagCountMatches(x, y) {
    let flagCount = 0;
    for (let i = -1; i <= 1; i++) {
        for (let j = -1; j <= 1; j++) {
            if (i === 0 && j === 0) continue;
            const nx = x + i;
            const ny = y + j;
            if (nx >= 0 && nx < gridSize && ny >= 0 && ny < gridSize) {
                const neighborCell = document.getElementById(`cell-${nx}-${ny}`);
                if (neighborCell.classList.contains('flagged')) {
                    flagCount++;
                }
            }
        }
    }

    const cellValue = board[x][y];

    if (flagCount === cellValue) {
        for (let i = -1; i <= 1; i++) {
            for (let j = -1; j <= 1; j++) {
                if (i === 0 && j === 0) continue;
                const nx = x + i;
                const ny = y + j;
                if (nx >= 0 && nx < gridSize && ny >= 0 && ny < gridSize) {
                    if (!document.getElementById(`cell-${nx}-${ny}`).classList.contains('flagged'))
                    {
                        revealCell(nx, ny); // Call revealCell directly
                    }
                }
            }
        }
        checkWinCondition(); // Check win condition after revealing neighbors
    }
}

function createTutorialGrid(gridState) {
  let gridHTML = '<div class="tutorial-grid">';
  for (let row = 0; row < 3; row++) {
    gridHTML += '<div class="tutorial-row">';
    for (let col = 0; col < 3; col++) {
      const cellState = gridState[row][col];
      let cellClass = 'tutorial-cell';
      let cellContent = '';

      switch (cellState) {
        case 'mine':
          cellClass += ' mine';
          cellContent = 'M'; // Or an image of a mine
          break;
        case 'flag':
          cellClass += ' flagged';
          cellContent = 'F'; // Or an image of a flag
          break;
        case 'revealed':
          cellClass += ' revealed';
          cellContent = '1'; // Or the actual number
          break;
        default:
          break;
      }

      gridHTML += `<div class="${cellClass}">${cellContent}</div>`;
    }
    gridHTML += '</div>';
  }
  gridHTML += '</div>';
  return gridHTML;
}

const tutorialSlides = [
    {
        title: "Welcome to Minesweeper!",
        text: "The goal is to reveal all the tiles that don't contain mines. Numbers indicate how many mines are adjacent to a tile.",
        visual: [
          ['', '', ''],
          ['', '', ''],
          ['', '', '']
        ]
    },
    {
        title: "Revealing Tiles",
        text: "Click on a tile to reveal it. If you click on a mine, the game is over!",
        visual: [
          ['', '', ''],
          ['', '1', ''],
          ['', '', '']
        ]
    },
    {
        title: "Flagging Mines",
        text: "Right-click on a tile to flag it as a potential mine. This helps you keep track of where the mines are.",
        visual: [
          ['', 'F', ''],
          ['', '1', ''],
          ['', '', '']
        ]
    },
    {
        title: "Chord/Double Click",
        text: "If a revealed tile has the same number of flags as its number, click it to reveal the surrounding tiles!",
        visual: [
          ['', 'F', ''],
          ['', '1', ''],
          ['', '', '']
        ]
    }
];

 function showSlide() {
    const slide = tutorialSlides[currentSlide];
    document.getElementById('tutorial-title').textContent = slide.title;
    document.getElementById('tutorial-text').textContent = slide.text;
 let gridHTML = createTutorialGrid(slide.visual);
 let tempElement = document.createElement('div');
 tempElement.innerHTML = gridHTML;
 document.body.appendChild(tempElement);
    document.getElementById('tutorial-visual').innerHTML = createTutorialGrid(slide.visual);
}

// Function to handle the forfeit action
function forfeitGame() {
    if (gameOver || gameWon) return; // Do nothing if the game is already over or won
    revealAllMines(); // Function to reveal all mines
    gameOver = true; // Set the game state to over
    alert("You have forfeited the game!"); // Notify the player
    // Optionally, reset the game if needed
    // resetGame();
}

// Initialize the game when the page loads
window.onload = () => {
    setDifficulty(30, 100);

    // Difficulty button functionality
    const easyButton = document.getElementById('easy-button');
    easyButton.addEventListener('click', () => setDifficulty(10, 10));

    const mediumButton = document.getElementById('medium-button');
    mediumButton.addEventListener('click', () => setDifficulty(15, 30));

    const hardButton = document.getElementById('hard-button');
    hardButton.addEventListener('click', () => setDifficulty(30, 100));

    // Tutorial button functionality
    const tutorialButton = document.getElementById('tutorial-button');
    const tutorialPopup = document.getElementById('tutorial-popup');
    tutorialButton.addEventListener('click', () => {
        tutorialPopup.style.display = 'block';
        currentSlide = 0; // Reset to first slide
        showSlide();
    });

    // Previous button functionality
    const prevButton = document.getElementById('prev-button');
    prevButton.addEventListener('click', () => {
        currentSlide = Math.max(0, currentSlide - 1);
        showSlide();
    });

    // Next button functionality
    const nextButton = document.getElementById('next-button');
    nextButton.addEventListener('click', () => {
        currentSlide = Math.min(tutorialSlides.length - 1, currentSlide + 1);
        showSlide();
    });

    // Close button functionality
    const closeButton = document.getElementById('close-button');
    closeButton.addEventListener('click', () => {
        tutorialPopup.style.display = 'none';
    });

    // Reset button functionality
    const resetButton = document.getElementById('reset-button');
    resetButton.addEventListener('click', resetGame);

    // Forfeit button functionality
    const forfeitButton = document.getElementById('forfeit-button');
    if (forfeitButton) { // Ensure the button exists before adding listener
        forfeitButton.addEventListener('click', forfeitGame);
    }

    initializeBoard(numberOfMines);
    createBoard();
};
