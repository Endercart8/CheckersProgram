# checkers_env.py
import numpy as np
import copy

# Constants for board representation
EMPTY = 0
WHITE_PAWN = 1
WHITE_KING = 2
BLACK_PAWN = -1
BLACK_KING = -2

class CheckersEnv:
    def __init__(self, max_moves=200):
        self.board = self._init_board()
        self.current_player = 1  # 1 for White, -1 for Black
        self.done = False
        self.winner = None
        self.max_moves = max_moves
        self.move_count = 0

    def _init_board(self):
        """
        Creates the initial 8x8 checkers board.
        Black (-1) starts at top, White (1) at bottom.
        """
        board = np.zeros((8, 8), dtype=np.int8)
        for r in range(3):
            for c in range(8):
                if (r + c) % 2 == 1:
                    board[r, c] = BLACK_PAWN
        for r in range(5, 8):
            for c in range(8):
                if (r + c) % 2 == 1:
                    board[r, c] = WHITE_PAWN
        return board

    def clone(self):
        return copy.deepcopy(self)
        #return self.board.copy(), self.current_player
        #return copy.deepcopy(self), self.current_player

    def reset(self):
        """
        Reset the game to the initial starting position and return the initial state.
        """
        self.board = self._init_board()
        self.current_player = 1
        self.done = False
        self.winner = None
        self.move_count = 0
        # You may also return initial observation, e.g.:
        return self.board.copy(), self.current_player

    def state_to_tensor(self, state):
        """
        Convert the checkers board state into a tensor suitable for the neural net input.
        For example, encode positions of regular pieces and kings for both players in separate planes.
        Assume `state` is your internal board representation.
        """
        import numpy as np

        # Unpack the tuple (board, current_player)
        board, current_player = state

        # If already a tensor with channels, just return float32 version
        if len(board.shape) == 3:
            return board.astype(np.float32)

        tensor = np.zeros((5, 8, 8), dtype=np.float32)  # 5 planes now

        for r in range(8):
            for c in range(8):
                piece = board[r, c]
                if piece == 1:   # white pawn
                    tensor[0, r, c] = 1.0
                elif piece == 2: # white king
                    tensor[1, r, c] = 1.0
                elif piece == -1: # black pawn
                    tensor[2, r, c] = 1.0
                elif piece == -2: # black king
                    tensor[3, r, c] = 1.0

        # Add a plane filled with current_player (1 or -1) so net knows whose turn it is
        tensor[4, :, :] = current_player

        return tensor

    def get_legal_moves(self, board, current_player):
        """
        Returns a list of all legal moves for the current player.
        A move is ((start_row, start_col), [(r1, c1), (r2, c2), ...])
        where the path may include multiple jumps.
        """
        moves = []
        captures = []

        for r in range(8):
            for c in range(8):
                piece = board[r, c]
                if piece * current_player > 0: #if the piece is yours:
                    single_moves, single_captures = self._get_piece_moves(board, current_player, r, c, piece)
                    captures.extend(single_captures)
                    moves.extend(single_moves)

        # Forced capture rule
        if captures:
            return captures
        return moves

    def _get_piece_moves(self, board, current_player, r, c, piece):
        """
        Get non-capture and capture moves for a single piece.
        """

        moves = []
        captures = []

        directions = []
        if abs(piece) == WHITE_PAWN: #if a pawn
            directions = [(-1, -1), (-1, 1)] if piece > 0 else [(1, -1), (1, 1)]
        elif abs(piece) == WHITE_KING: #if a king
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        # Normal moves (if no captures)
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8 and board[nr, nc] == EMPTY:
                moves.append(((r, c), [(nr, nc)]))

        # Captures
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            jr, jc = r + 2 * dr, c + 2 * dc
            if (
                0 <= nr < 8 and 0 <= nc < 8 and #inside the rows of the board
                0 <= jr < 8 and 0 <= jc < 8 and #inside the collumns of the board
                board[nr, nc] * current_player < 0 and #opposide colored pieces
                board[jr, jc] == EMPTY #jump space enmpty
            ):
                # Capture found, explore multi-jumps recursively
                
                temp_env = self.clone()
                temp_env.board = board.copy()
                temp_env.current_player = current_player

                temp_env._make_move_no_switch(temp_env.board, ((r, c), [(jr, jc)]))
                further = temp_env._get_piece_moves(temp_env.board, current_player, jr, jc, temp_env.board[jr, jc])[1]
                if further:
                    #board = temp_env.board #update the board so be accurate after the jumps
                    for _, path in further:
                        captures.append(((r, c), [(jr, jc)] + path))
                else:
                    captures.append(((r, c), [(jr, jc)]))
                

        return moves, captures

    def step(self, move):
        """
        Applies a move and switches players.
        """
        self._make_move_no_switch(self.board, move)
        self.move_count += 1

        # Switch turn
        self.current_player *= -1

        # Check for game end
        if not self.get_legal_moves(self.board.copy(), self.current_player):
            self.done = True
            self.winner = -self.current_player  # Opponent wins
            if self.winner == 1:
                reward = 1.0
            elif self.winner == -1:
                reward = -1.0
            else:
                reward = 0.0
        elif self.move_count >= self.max_moves:
            self.done = True
            reward = 0.0
        else:
            reward = 0.0

        # Return the new state (board, current_player), reward, done, and empty info
        return (self.board.copy(), self.current_player), reward, self.done, {}

    def _make_move_no_switch(self, board, move):
        """
        Executes a move without changing turns.
        Handles captures and kinging.
        """
        hasKinged = False

        (start_r, start_c), path = move
        piece = board[start_r, start_c]
        board[start_r, start_c] = EMPTY

        r, c = start_r, start_c
        for nr, nc in path:
            # If it's a jump, remove captured piece
            if abs(nr - r) == 2:
                board[(r + nr) // 2, (c + nc) // 2] = EMPTY
            r, c = nr, nc
            # Check if it becomes a king
            if piece == WHITE_PAWN and r == 0:
                hasKinged = True
            elif piece == BLACK_PAWN and r == 7:
                hasKinged = True
                
        # Kinging
        if piece == WHITE_PAWN and hasKinged:
            board[r, c] = WHITE_KING
        elif piece == BLACK_PAWN and hasKinged:
            board[r, c] = BLACK_KING
        else:
            board[r, c] = piece

    def render(self):
        """
        Prints the board to console.
        """
        symbols = {
            EMPTY: ".",
            WHITE_PAWN: "w",
            WHITE_KING: "W",
            BLACK_PAWN: "b",
            BLACK_KING: "B"
        }
        print("  0 1 2 3 4 5 6 7")
        for r in range(8):
            print(r, " ".join(symbols[self.board[r, c]] for c in range(8)))
        print(f"Current player: {'White' if self.current_player == 1 else 'Black'}")
        if self.done:
            print(f"Game Over! Winner: {'White' if self.winner == 1 else 'Black'}")