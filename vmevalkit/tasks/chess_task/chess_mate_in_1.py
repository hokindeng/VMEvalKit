#!/usr/bin/env python3
"""
Chess Mate-in-1 Task Generator and Validator for VMEvalKit

This module creates chess positions where one side can deliver checkmate in exactly one move.
These are perfect for testing if video models can identify and demonstrate the final winning move.

Author: VMEvalKit Team
"""

import sys
import os
import json
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

# Add python-chess to path
chess_path = os.path.join(os.path.dirname(__file__), '..', '..', 'submodules', 'python-chess')
if chess_path not in sys.path:
    sys.path.append(chess_path)

import chess
import chess.svg


@dataclass
class MateIn1Puzzle:
    """Represents a mate-in-1 chess puzzle."""
    puzzle_id: str
    fen: str
    side_to_move: str  # "white" or "black"  
    mate_moves: List[str]  # List of moves that deliver checkmate (in SAN notation)
    difficulty: str  # "easy", "medium", "hard"
    description: str
    tags: List[str]  # ["back_rank", "smothered", "queen_sac", etc.]


class MateIn1Generator:
    """Generates and manages mate-in-1 chess puzzles."""
    
    def __init__(self):
        self.puzzles: List[MateIn1Puzzle] = []
        self._load_predefined_puzzles()
    
    def _load_predefined_puzzles(self):
        """Load ONLY verified working mate-in-1 positions."""
        
        # 1. Classic back-rank mate - VERIFIED WORKING ✓
        self.puzzles.append(MateIn1Puzzle(
            puzzle_id="back_rank_001",
            fen="6k1/5ppp/8/8/8/8/8/R6K w - - 0 1",
            side_to_move="white",
            mate_moves=["Ra8#"],
            difficulty="easy",
            description="White rook delivers classic back-rank mate. Black king trapped by own pawns.",
            tags=["back_rank", "rook", "basic"]
        ))
        
        # 2. Queen corner mate - VERIFIED WORKING ✓
        self.puzzles.append(MateIn1Puzzle(
            puzzle_id="queen_corner_001", 
            fen="6Qk/8/6K1/8/8/8/8/8 w - - 0 1",
            side_to_move="white",
            mate_moves=["Qh7#"],
            difficulty="easy",
            description="White queen delivers checkmate in the corner with king support.",
            tags=["queen", "corner", "basic"]
        ))
        
        # 3. Black queen mate - VERIFIED WORKING ✓
        self.puzzles.append(MateIn1Puzzle(
            puzzle_id="black_queen_001",
            fen="6qK/8/6k1/8/8/8/8/8 b - - 0 1", 
            side_to_move="black",
            mate_moves=["Qg7#"],
            difficulty="easy",
            description="Black queen delivers checkmate with king support.",
            tags=["queen", "black_to_move", "basic"]
        ))
    
    def get_puzzle(self, puzzle_id: str) -> Optional[MateIn1Puzzle]:
        """Get a specific puzzle by ID."""
        for puzzle in self.puzzles:
            if puzzle.puzzle_id == puzzle_id:
                return puzzle
        return None
    
    def get_random_puzzle(self, difficulty: Optional[str] = None) -> MateIn1Puzzle:
        """Get a random puzzle, optionally filtered by difficulty."""
        candidates = self.puzzles
        if difficulty:
            candidates = [p for p in self.puzzles if p.difficulty == difficulty]
        
        if not candidates:
            candidates = self.puzzles
            
        return random.choice(candidates)
    
    def get_puzzles_by_tag(self, tag: str) -> List[MateIn1Puzzle]:
        """Get all puzzles with a specific tag."""
        return [p for p in self.puzzles if tag in p.tags]
    
    def validate_puzzle(self, puzzle: MateIn1Puzzle) -> bool:
        """Validate that a puzzle is correctly configured."""
        try:
            board = chess.Board(puzzle.fen)
            
            # Check that it's the correct side to move
            expected_white_to_move = puzzle.side_to_move == "white"
            if board.turn != expected_white_to_move:
                return False
            
            # Check that each mate move actually delivers mate
            for move_san in puzzle.mate_moves:
                test_board = board.copy()
                move = test_board.parse_san(move_san)
                test_board.push(move)
                
                if not test_board.is_checkmate():
                    return False
            
            return True
            
        except Exception:
            return False
    
    def create_svg_board(self, puzzle: MateIn1Puzzle, size: int = 400) -> str:
        """Generate SVG representation of the puzzle position."""
        board = chess.Board(puzzle.fen)
        return chess.svg.board(board, size=size)
    
    def export_puzzles(self, filename: str):
        """Export all puzzles to JSON file."""
        puzzle_data = [asdict(puzzle) for puzzle in self.puzzles]
        with open(filename, 'w') as f:
            json.dump(puzzle_data, f, indent=2)
    
    def import_puzzles(self, filename: str):
        """Import puzzles from JSON file."""
        with open(filename, 'r') as f:
            puzzle_data = json.load(f)
        
        imported_puzzles = []
        for data in puzzle_data:
            puzzle = MateIn1Puzzle(**data)
            if self.validate_puzzle(puzzle):
                imported_puzzles.append(puzzle)
        
        self.puzzles.extend(imported_puzzles)
        return len(imported_puzzles)


class MateIn1Validator:
    """Validates solutions to mate-in-1 puzzles."""
    
    @staticmethod
    def validate_solution(puzzle: MateIn1Puzzle, solution_move: str) -> Dict[str, any]:
        """
        Validate a proposed solution to a mate-in-1 puzzle.
        
        Args:
            puzzle: The mate-in-1 puzzle
            solution_move: The move proposed as solution (in SAN notation)
            
        Returns:
            Dict with validation results:
            - is_correct: bool - Whether the move is correct
            - is_legal: bool - Whether the move is legal
            - is_mate: bool - Whether the move delivers checkmate
            - message: str - Explanation of the result
        """
        try:
            board = chess.Board(puzzle.fen)
            
            # Check if move is legal
            try:
                move = board.parse_san(solution_move)
                if move not in board.legal_moves:
                    return {
                        "is_correct": False,
                        "is_legal": False, 
                        "is_mate": False,
                        "message": f"Move {solution_move} is not legal in this position"
                    }
            except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError):
                return {
                    "is_correct": False,
                    "is_legal": False,
                    "is_mate": False,
                    "message": f"Move {solution_move} is invalid or illegal"
                }
            
            # Apply the move
            board.push(move)
            
            # Check if it's checkmate
            is_mate = board.is_checkmate()
            is_correct = solution_move in puzzle.mate_moves
            
            if is_correct and is_mate:
                message = f"Correct! {solution_move} delivers checkmate."
            elif is_mate and not is_correct:
                message = f"Good! {solution_move} delivers checkmate, but the expected solution was {puzzle.mate_moves[0]}."
            elif not is_mate:
                if board.is_check():
                    message = f"{solution_move} gives check but not checkmate."
                else:
                    message = f"{solution_move} is legal but doesn't deliver checkmate."
            else:
                message = f"Unexpected validation state for {solution_move}."
            
            return {
                "is_correct": is_correct,
                "is_legal": True,
                "is_mate": is_mate,
                "message": message
            }
            
        except Exception as e:
            return {
                "is_correct": False,
                "is_legal": False,
                "is_mate": False,
                "message": f"Error validating move: {str(e)}"
            }
    
    @staticmethod
    def analyze_position(puzzle: MateIn1Puzzle) -> Dict[str, any]:
        """Provide detailed analysis of a mate-in-1 position."""
        board = chess.Board(puzzle.fen)
        
        # Find all moves that deliver checkmate
        mate_moves = []
        for move in board.legal_moves:
            test_board = board.copy()
            test_board.push(move)
            if test_board.is_checkmate():
                mate_moves.append(board.san(move))
        
        # Find all moves that give check (but not mate)
        check_moves = []
        for move in board.legal_moves:
            test_board = board.copy()
            test_board.push(move)
            if test_board.is_check() and not test_board.is_checkmate():
                check_moves.append(board.san(move))
        
        return {
            "position_fen": puzzle.fen,
            "side_to_move": puzzle.side_to_move,
            "total_legal_moves": len(list(board.legal_moves)),
            "mate_moves": mate_moves,
            "check_moves": check_moves,
            "expected_mates": puzzle.mate_moves,
            "is_valid_puzzle": len(mate_moves) > 0,
            "has_multiple_mates": len(mate_moves) > 1
        }


def create_vmevalkit_task(puzzle: MateIn1Puzzle) -> Dict[str, any]:
    """
    Create a VMEvalKit task structure for a mate-in-1 puzzle.
    
    Returns a task dict that can be used by the VMEvalKit evaluation pipeline.
    """
    generator = MateIn1Generator()
    
    # Generate the input image (board position)
    svg_board = generator.create_svg_board(puzzle)
    
    # Create the text prompt
    side = "White" if puzzle.side_to_move == "white" else "Black"
    prompt = f"{side} to move. Find checkmate in one move."
    
    # Expected output description
    expected_output = f"Video showing the piece movement for {puzzle.mate_moves[0]} resulting in checkmate"
    
    return {
        "task_id": f"chess_mate_in_1_{puzzle.puzzle_id}",
        "task_type": "chess_reasoning",
        "difficulty": puzzle.difficulty,
        "input_image": svg_board,  # SVG representation of board
        "text_prompt": prompt,
        "expected_solution": puzzle.mate_moves,
        "evaluation_criteria": {
            "move_legality": "Move must be legal in the given position",
            "checkmate_delivery": "Move must result in checkmate", 
            "video_clarity": "Video must clearly show the piece movement",
            "sequence_completeness": "Must show the complete move from start to finish"
        },
        "metadata": {
            "fen": puzzle.fen,
            "description": puzzle.description,
            "tags": puzzle.tags,
            "side_to_move": puzzle.side_to_move
        }
    }


if __name__ == "__main__":
    # Demo the mate-in-1 system
    print("Chess Mate-in-1 Task Generator Demo")
    print("=" * 50)
    
    # Create generator
    generator = MateIn1Generator()
    validator = MateIn1Validator()
    
    print(f"Loaded {len(generator.puzzles)} mate-in-1 puzzles")
    print()
    
    # Show a few examples
    for difficulty in ["easy", "medium", "hard"]:
        puzzle = generator.get_random_puzzle(difficulty)
        print(f"{difficulty.upper()} PUZZLE: {puzzle.puzzle_id}")
        print(f"Description: {puzzle.description}")
        print(f"FEN: {puzzle.fen}")
        print(f"Side to move: {puzzle.side_to_move}")
        print(f"Mate moves: {puzzle.mate_moves}")
        print(f"Tags: {', '.join(puzzle.tags)}")
        
        # Validate the puzzle
        is_valid = generator.validate_puzzle(puzzle)
        print(f"Puzzle validation: {'✓ VALID' if is_valid else '✗ INVALID'}")
        
        # Show analysis
        analysis = validator.analyze_position(puzzle)
        print(f"Total legal moves: {analysis['total_legal_moves']}")
        print(f"Mate moves found: {analysis['mate_moves']}")
        print()
        
        # Test validation of correct solution
        if puzzle.mate_moves:
            result = validator.validate_solution(puzzle, puzzle.mate_moves[0])
            print(f"Solution validation: {result['message']}")
        print("-" * 30)
