#!/usr/bin/env python3
"""
Enhanced Chess Mate-in-1 Generator - Target: 100+ Positions

Advanced systematic generation using multiple approaches:
- Template multiplication
- Position transformations
- Database mining
- Combinatorial generation

Author: VMEvalKit Team  
"""

import sys
import os
import json
import random
import itertools
from typing import List, Dict, Tuple, Optional, Set

# Add python-chess to path
chess_path = os.path.join(os.path.dirname(__file__), '..', '..', 'submodules', 'python-chess')
if chess_path not in sys.path:
    sys.path.append(chess_path)

import chess

from chess_mate_in_1 import MateIn1Puzzle


class EnhancedMateGenerator:
    """Enhanced mate-in-1 generator with multiple strategies."""
    
    def __init__(self):
        self.generated_positions: List[MateIn1Puzzle] = []
        self.position_hashes: Set[str] = set()
        self.generation_stats = {
            "back_rank": 0,
            "queen_king": 0, 
            "rook_king": 0,
            "piece_mates": 0,
            "tactical": 0,
            "transformations": 0,
            "database": 0
        }
        
    def generate_comprehensive_collection(self) -> List[MateIn1Puzzle]:
        """Generate comprehensive collection using all strategies."""
        print("🚀 ENHANCED MATE-IN-1 GENERATION SYSTEM V2")
        print("Target: 100+ verified positions")
        print("=" * 70)
        
        # Strategy 1: Extended back-rank patterns
        self._generate_extended_back_rank_patterns()
        
        # Strategy 2: Systematic Queen+King mates
        self._generate_systematic_queen_king_mates()
        
        # Strategy 3: Working Rook+King patterns
        self._generate_working_rook_king_mates()
        
        # Strategy 4: Simple piece mates
        self._generate_simple_piece_mates()
        
        # Strategy 5: Position transformations
        self._generate_position_transformations()
        
        # Strategy 6: Combinatorial generation
        self._generate_combinatorial_positions()
        
        # Strategy 7: Known working patterns expansion
        self._generate_verified_pattern_expansions()
        
        self._print_final_statistics()
        return self.generated_positions
    
    def _add_verified_position(self, fen: str, expected_mates: List[str],
                             description: str, tags: List[str], 
                             difficulty: str = "easy") -> bool:
        """Add position only if verified to work."""
        try:
            # Skip duplicates
            if fen in self.position_hashes:
                return False
                
            board = chess.Board(fen)
            valid_mates = []
            
            # Test each expected mate move
            for mate_san in expected_mates:
                try:
                    move = board.parse_san(mate_san)
                    if move in board.legal_moves:
                        test_board = board.copy()
                        test_board.push(move)
                        if test_board.is_checkmate():
                            valid_mates.append(mate_san)
                except:
                    continue
            
            # Also find any additional mate moves
            for move in board.legal_moves:
                test_board = board.copy()
                test_board.push(move)
                if test_board.is_checkmate():
                    mate_san = board.san(move)
                    if mate_san not in valid_mates:
                        valid_mates.append(mate_san)
            
            if not valid_mates:
                return False
            
            # Create puzzle
            puzzle_id = f"v2_{len(self.generated_positions):03d}"
            side = "white" if board.turn else "black"
            
            puzzle = MateIn1Puzzle(
                puzzle_id=puzzle_id,
                fen=fen,
                side_to_move=side,
                mate_moves=valid_mates,
                difficulty=difficulty,
                description=description,
                tags=tags
            )
            
            self.generated_positions.append(puzzle)
            self.position_hashes.add(fen)
            return True
            
        except Exception:
            return False
    
    def _generate_extended_back_rank_patterns(self):
        """Generate extensive back-rank mate variations."""
        print("🔥 Strategy 1: Extended back-rank patterns")
        start_count = len(self.generated_positions)
        
        # Different king positions on back rank
        king_positions = ['k7', '1k6', '2k5', '3k4', '4k3', '5k2', '6k1', '7k']
        
        # Different pawn structures that trap the king
        pawn_structures = [
            'ppp5',   # 3 pawns on a,b,c
            '1ppp4',  # 3 pawns on b,c,d  
            '2ppp3',  # 3 pawns on c,d,e
            '3ppp2',  # 3 pawns on d,e,f
            '4ppp1',  # 3 pawns on e,f,g
            '5ppp',   # 3 pawns on f,g,h
            'pppp4',  # 4 pawns
            '1pppp3', # 4 pawns shifted
            '2pppp2', # 4 pawns centered
            'pp6',    # 2 pawns
            '1pp5',   # 2 pawns shifted
            '2pp4',   # etc.
            '3pp3',
            '4pp2', 
            '5pp1',
            '6pp',
        ]
        
        # Attacking pieces and their moves
        attackers = [
            ('R', 'a1', 'Ra8#'),
            ('Q', 'a1', 'Qa8#'),
            ('R', 'b1', 'Rb8#'),
            ('Q', 'b1', 'Qb8#'),
        ]
        
        for king_pos in king_positions:
            for pawn_struct in pawn_structures:
                for piece, piece_pos, mate_move in attackers:
                    # Construct FEN
                    rank8 = king_pos
                    rank7 = pawn_struct
                    rank1 = piece_pos.replace(piece, piece) + '6K'
                    
                    if piece_pos == 'a1':
                        rank1 = piece + '6K'
                    elif piece_pos == 'b1':
                        rank1 = '1' + piece + '5K'
                    
                    fen = f"{rank8}/{rank7}/8/8/8/8/8/{rank1} w - - 0 1"
                    
                    desc = f"{piece_name(piece)} back-rank mate, {king_pos} with {pawn_struct}"
                    tags = ["back_rank", piece.lower()]
                    
                    self._add_verified_position(fen, [mate_move], desc, tags)
        
        count = len(self.generated_positions) - start_count
        self.generation_stats["back_rank"] = count
        print(f"   Generated {count} back-rank positions")
    
    def _generate_systematic_queen_king_mates(self):
        """Generate systematic Queen+King vs King mates."""
        print("👑 Strategy 2: Systematic Queen+King mates")
        start_count = len(self.generated_positions)
        
        # Queen and King coordinated mate patterns
        patterns = [
            # Corner mates
            ("6Qk/8/6K1/8/8/8/8/8", "Qh7#", "Queen corner mate, K on g6"),
            ("6Qk/8/5K2/8/8/8/8/8", "Qh7#", "Queen corner mate, K on f6"),
            ("7k/6Q1/6K1/8/8/8/8/8", "Qg8#", "Queen back rank, K on g6"),
            ("7k/5Q2/6K1/8/8/8/8/8", "Qf8#", "Queen side mate, K on g6"),
            
            # Edge mates with different king positions
            ("3Q3k/8/6K1/8/8/8/8/8", "Qd8#", "Queen edge mate"),
            ("2Q4k/8/6K1/8/8/8/8/8", "Qc8#", "Queen c-file mate"),
            ("Q6k/8/6K1/8/8/8/8/8", "Qa8#", "Queen a-file mate"),
            
            # Queen on different files
            ("k7/Q7/K7/8/8/8/8/8", "Qa8#", "Queen file mate with K support"),
            ("1k6/Q7/K7/8/8/8/8/8", "Qa8#", "Queen file mate, K on b8"),
        ]
        
        for pattern, move, desc in patterns:
            self._add_verified_position(pattern + " w - - 0 1", [move], desc, ["queen", "endgame"])
        
        # Generate variants with king on different squares
        base_positions = [
            ("6Qk", "Queen threatening h7"),
            ("7k", "King on h8 corner"),
        ]
        
        king_support_squares = ["6K1", "5K2", "7K", "4K3"]
        
        for king_enemy, q_desc in base_positions:
            for king_support in king_support_squares:
                pattern = f"{king_enemy}/8/{king_support}/8/8/8/8/8"
                # Try different queen moves
                possible_moves = ["Qg8#", "Qh7#", "Qf8#"]
                for move in possible_moves:
                    desc = f"{q_desc}, white king on various squares"
                    self._add_verified_position(pattern + " w - - 0 1", [move], desc, ["queen", "systematic"])
        
        count = len(self.generated_positions) - start_count
        self.generation_stats["queen_king"] = count
        print(f"   Generated {count} Queen+King positions")
    
    def _generate_working_rook_king_mates(self):
        """Generate verified working Rook+King mates."""
        print("🏰 Strategy 3: Working Rook+King mates")
        start_count = len(self.generated_positions)
        
        # Start with known working patterns and expand
        working_patterns = [
            # Edge restriction mates
            ("7k/R7/K7/8/8/8/8/8", "Ra8#", "Rook file mate with king support"),
            ("6k1/R7/K7/8/8/8/8/8", "Ra8#", "Rook file mate, king on g8"),
            ("5k2/R7/K7/8/8/8/8/8", "Ra8#", "Rook file mate, king on f8"),
            
            # Different rook positions
            ("k7/1R6/K7/8/8/8/8/8", "Rb8#", "Rook b-file mate"),
            ("k7/2R5/K7/8/8/8/8/8", "Rc8#", "Rook c-file mate"),
            
            # King on different support squares
            ("7k/R7/1K6/8/8/8/8/8", "Ra8#", "Rook mate, K on b6"),
            ("7k/R7/2K5/8/8/8/8/8", "Ra8#", "Rook mate, K on c6"),
        ]
        
        for pattern, move, desc in working_patterns:
            self._add_verified_position(pattern + " w - - 0 1", [move], desc, ["rook", "endgame"])
        
        count = len(self.generated_positions) - start_count
        self.generation_stats["rook_king"] = count
        print(f"   Generated {count} Rook+King positions")
    
    def _generate_simple_piece_mates(self):
        """Generate simple single-piece mates."""
        print("⚡ Strategy 4: Simple piece mates")  
        start_count = len(self.generated_positions)
        
        # Simple queen mates (no second piece needed)
        simple_queen = [
            ("k7/8/8/8/8/8/8/Q6K", "Qa8#", "Simple queen mate on a-file"),
            ("1k6/8/8/8/8/8/8/Q6K", "Qa8#", "Queen mate, king on b8"),
            ("7k/8/8/8/8/8/8/6QK", "Qg8#", "Queen mate on g-file"),
            
            # Queen mates on different files
            ("k7/8/8/8/8/8/8/1Q5K", "Qb8#", "Queen b-file mate"),
            ("k7/8/8/8/8/8/8/2Q4K", "Qc8#", "Queen c-file mate"),
            ("k7/8/8/8/8/8/8/3Q3K", "Qd8#", "Queen d-file mate"),
        ]
        
        for pattern, move, desc in simple_queen:
            self._add_verified_position(pattern + " w - - 0 1", [move], desc, ["queen", "simple"])
        
        # Black to move variants
        black_mates = [
            ("6qK/8/6k1/8/8/8/8/8", "Qg7#", "Black queen mate"),
            ("5q1K/8/6k1/8/8/8/8/8", "Qf7#", "Black queen f-file mate"),
            ("q6K/8/6k1/8/8/8/8/8", "Qa7#", "Black queen a-file mate"),
            
            # Black rook mates
            ("r6K/8/6k1/8/8/8/8/8", "Ra7#", "Black rook mate"),
            ("6rK/8/6k1/8/8/8/8/8", "Rg7#", "Black rook g-file mate"),
        ]
        
        for pattern, move, desc in black_mates:
            self._add_verified_position(pattern + " b - - 0 1", [move], desc, ["black_to_move", move[0].lower()])
        
        count = len(self.generated_positions) - start_count
        self.generation_stats["piece_mates"] = count
        print(f"   Generated {count} simple piece mates")
    
    def _generate_position_transformations(self):
        """Generate positions through transformations of existing ones."""
        print("🔄 Strategy 5: Position transformations")
        start_count = len(self.generated_positions)
        
        original_positions = self.generated_positions[:]
        
        for puzzle in original_positions:
            # Try horizontal mirroring
            mirrored = self._try_mirror_horizontal(puzzle)
            if mirrored:
                pass  # Already added by _add_verified_position
            
            # Try vertical mirroring (ranks 1↔8)
            v_mirrored = self._try_mirror_vertical(puzzle)
            if v_mirrored:
                pass
                
            # Try 180-degree rotation
            rotated = self._try_rotate_180(puzzle)
            if rotated:
                pass
        
        count = len(self.generated_positions) - start_count
        self.generation_stats["transformations"] = count
        print(f"   Generated {count} transformed positions")
    
    def _try_mirror_horizontal(self, puzzle: MateIn1Puzzle) -> bool:
        """Try to create horizontal mirror of position."""
        try:
            mirrored_fen = self._mirror_fen_horizontal(puzzle.fen)
            if not mirrored_fen or mirrored_fen == puzzle.fen:
                return False
                
            mirrored_moves = [self._mirror_move_horizontal(m) for m in puzzle.mate_moves]
            desc = puzzle.description + " (mirrored)"
            tags = puzzle.tags + ["transformed"]
            
            return self._add_verified_position(mirrored_fen, mirrored_moves, desc, tags, puzzle.difficulty)
        except:
            return False
    
    def _mirror_fen_horizontal(self, fen: str) -> Optional[str]:
        """Mirror FEN horizontally (a↔h files)."""
        try:
            parts = fen.split()
            board = parts[0]
            
            ranks = board.split('/')
            mirrored_ranks = []
            
            for rank in ranks:
                # Reverse the rank string
                mirrored_rank = rank[::-1]  
                mirrored_ranks.append(mirrored_rank)
            
            mirrored_board = '/'.join(mirrored_ranks)
            return mirrored_board + " " + " ".join(parts[1:])
        except:
            return None
    
    def _mirror_move_horizontal(self, move: str) -> str:
        """Mirror move horizontally (a↔h)."""
        file_map = {'a': 'h', 'b': 'g', 'c': 'f', 'd': 'e',
                   'e': 'd', 'f': 'c', 'g': 'b', 'h': 'a'}
        
        result = ""
        for char in move:
            result += file_map.get(char, char)
        return result
    
    def _try_mirror_vertical(self, puzzle: MateIn1Puzzle) -> bool:
        """Try vertical mirroring (rank 1↔8).""" 
        try:
            parts = puzzle.fen.split()
            board = parts[0]
            
            # Reverse rank order and swap colors
            ranks = board.split('/')
            mirrored_ranks = []
            
            for rank in reversed(ranks):
                # Swap piece colors
                new_rank = ""
                for char in rank:
                    if char.isalpha():
                        new_rank += char.swapcase()  # Upper↔lower
                    else:
                        new_rank += char
                mirrored_ranks.append(new_rank)
            
            mirrored_board = '/'.join(mirrored_ranks)
            
            # Switch side to move
            side = 'b' if parts[1] == 'w' else 'w'
            mirrored_fen = mirrored_board + f" {side} " + " ".join(parts[2:])
            
            # Mirror moves (rank numbers)
            rank_map = {'1': '8', '2': '7', '3': '6', '4': '5',
                       '5': '4', '6': '3', '7': '2', '8': '1'}
            
            mirrored_moves = []
            for move in puzzle.mate_moves:
                new_move = ""
                for char in move:
                    new_move += rank_map.get(char, char)
                mirrored_moves.append(new_move)
            
            desc = puzzle.description + " (vertically mirrored)"
            tags = puzzle.tags + ["transformed", "color_swapped"]
            
            return self._add_verified_position(mirrored_fen, mirrored_moves, desc, tags, puzzle.difficulty)
        except:
            return False
    
    def _try_rotate_180(self, puzzle: MateIn1Puzzle) -> bool:
        """Try 180-degree rotation."""
        # Combination of horizontal + vertical mirroring
        return False  # Skip for now - complex implementation
    
    def _generate_combinatorial_positions(self):
        """Generate positions through combinatorial placement."""
        print("🧮 Strategy 6: Combinatorial generation")
        start_count = len(self.generated_positions)
        
        # Try systematic queen placement against cornered king
        king_corners = ['7k', '6k1', 'k7', '1k6']
        queen_squares = ['6Q1', '5Q2', '4Q3', '3Q4', '2Q5', '1Q6', 'Q7']
        
        for k_pos in king_corners:
            for q_pos in queen_squares:
                pattern = f"{k_pos}/8/{q_pos}/8/8/8/8/8"
                
                # Try common queen mate moves
                possible_moves = ["Qa8#", "Qb8#", "Qc8#", "Qd8#", "Qe8#", "Qf8#", "Qg8#", "Qh8#",
                                "Qa7#", "Qb7#", "Qc7#", "Qd7#", "Qe7#", "Qf7#", "Qg7#", "Qh7#"]
                
                for move in possible_moves:
                    desc = f"Combinatorial queen mate: {k_pos} vs {q_pos}"
                    self._add_verified_position(pattern + " w - - 0 1", [move], desc, ["queen", "combinatorial"])
        
        count = len(self.generated_positions) - start_count  
        self.generation_stats["tactical"] = count
        print(f"   Generated {count} combinatorial positions")
    
    def _generate_verified_pattern_expansions(self):
        """Expand known working patterns."""
        print("✅ Strategy 7: Verified pattern expansion")
        start_count = len(self.generated_positions)
        
        # Take the 3 original verified positions and create many variants
        base_patterns = [
            ("6k1/5ppp/8/8/8/8/8/R6K", "Ra8#", "back_rank"),
            ("6Qk/8/6K1/8/8/8/8/8", "Qh7#", "queen_corner"),  
            ("6qK/8/6k1/8/8/8/8/8", "Qg7#", "black_queen"),
        ]
        
        for base_fen, base_move, pattern_type in base_patterns:
            # Create systematic variations
            self._expand_verified_pattern(base_fen, base_move, pattern_type)
        
        count = len(self.generated_positions) - start_count
        self.generation_stats["database"] = count  
        print(f"   Generated {count} verified expansions")
    
    def _expand_verified_pattern(self, base_fen: str, base_move: str, pattern_type: str):
        """Create variations of a verified pattern."""
        
        if pattern_type == "back_rank":
            # Move attacking piece to different files
            for file_char, move_char in [('R', 'R'), ('Q', 'Q')]:
                for pos in ['R6K', '1R5K', '2R4K', '3R3K', '4R2K', '5R1K', '6RK']:
                    new_fen = base_fen.replace('R6K', pos.replace('R', file_char))
                    new_move = f"{file_char}a8#"
                    desc = f"Back-rank {file_char.lower()} mate variant"
                    self._add_verified_position(new_fen + " w - - 0 1", [new_move], desc, ["back_rank", file_char.lower()])
        
        elif pattern_type == "queen_corner":
            # Move white king to different support positions  
            king_positions = ['6K1', '5K2', '7K', '4K3', '3K4']
            for k_pos in king_positions:
                new_fen = base_fen.replace('6K1', k_pos) 
                desc = f"Queen corner mate with king on different square"
                self._add_verified_position(new_fen + " w - - 0 1", [base_move], desc, ["queen", "corner"])
    
    def _print_final_statistics(self):
        """Print comprehensive generation statistics."""
        total = len(self.generated_positions)
        print(f"\n🎯 FINAL RESULTS: {total} verified mate-in-1 positions")
        print("=" * 70)
        print("📊 Generation breakdown:")
        for strategy, count in self.generation_stats.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"   {strategy:<20}: {count:>3} positions ({percentage:4.1f}%)")
        
        # Difficulty distribution
        diff_counts = {}
        for puzzle in self.generated_positions:
            diff_counts[puzzle.difficulty] = diff_counts.get(puzzle.difficulty, 0) + 1
        
        print(f"\n📈 Difficulty distribution:")
        for diff, count in diff_counts.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"   {diff:<10}: {count:>3} positions ({percentage:4.1f}%)")


def piece_name(piece: str) -> str:
    """Get piece name from letter."""
    return {'R': 'Rook', 'Q': 'Queen', 'B': 'Bishop', 'N': 'Knight', 'K': 'King'}.get(piece, piece)


def main():
    """Run enhanced mate generation system.""" 
    generator = EnhancedMateGenerator()
    positions = generator.generate_comprehensive_collection()
    
    # Export to JSON
    output_file = "enhanced_mate_positions.json"
    export_data = []
    
    for puzzle in positions:
        export_data.append({
            "puzzle_id": puzzle.puzzle_id,
            "fen": puzzle.fen,
            "side_to_move": puzzle.side_to_move,
            "mate_moves": puzzle.mate_moves,
            "difficulty": puzzle.difficulty,
            "description": puzzle.description,
            "tags": puzzle.tags
        })
    
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\n💾 Exported to {output_file}")
    print("🚀 Enhanced generation complete!")
    
    return positions


if __name__ == "__main__":
    main()
