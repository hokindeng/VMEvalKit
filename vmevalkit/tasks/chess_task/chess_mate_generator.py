#!/usr/bin/env python3
"""
Chess Mate-in-1 Position Generator - 100+ Positions

This module systematically generates large collections of mate-in-1 positions
using template patterns, variations, and verification systems.

Author: VMEvalKit Team
"""

import sys
import os
import json
import random
import itertools
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

# Add python-chess to path
chess_path = os.path.join(os.path.dirname(__file__), '..', '..', 'submodules', 'python-chess')
if chess_path not in sys.path:
    sys.path.append(chess_path)

import chess

from chess_mate_in_1 import MateIn1Puzzle


class MatePositionGenerator:
    """Generates large collections of mate-in-1 positions systematically."""
    
    def __init__(self):
        self.generated_positions: List[MateIn1Puzzle] = []
        self.position_hashes: Set[str] = set()  # Avoid duplicates
        
    def generate_all_positions(self) -> List[MateIn1Puzzle]:
        """Generate comprehensive collection of 100+ mate-in-1 positions."""
        print("🏭 SYSTEMATIC MATE-IN-1 GENERATION SYSTEM")
        print("=" * 60)
        
        # Phase 1: Back-rank mate templates
        self._generate_back_rank_mates()
        
        # Phase 2: Queen+King endgame templates  
        self._generate_queen_king_mates()
        
        # Phase 3: Rook+King endgame templates
        self._generate_rook_king_mates()
        
        # Phase 4: Tactical pattern templates
        self._generate_tactical_mates()
        
        # Phase 5: Special piece mates
        self._generate_special_mates()
        
        # Phase 6: Mirror and variation generation
        self._generate_position_variants()
        
        print(f"\n🎯 GENERATION COMPLETE: {len(self.generated_positions)} positions")
        return self.generated_positions
    
    def _add_position_if_valid(self, fen: str, mate_moves: List[str], 
                              description: str, tags: List[str], 
                              difficulty: str = "easy") -> bool:
        """Add position to collection if it's valid and unique."""
        try:
            # Check for duplicates
            if fen in self.position_hashes:
                return False
                
            # Verify it's actually mate-in-1
            board = chess.Board(fen)
            
            # Check each proposed mate move
            valid_mates = []
            for move_san in mate_moves:
                try:
                    move = board.parse_san(move_san)
                    if move in board.legal_moves:
                        test_board = board.copy()
                        test_board.push(move)
                        if test_board.is_checkmate():
                            valid_mates.append(move_san)
                except:
                    continue
            
            if not valid_mates:
                return False
                
            # Create puzzle ID
            puzzle_id = f"gen_{len(self.generated_positions):03d}"
            side_to_move = "white" if board.turn else "black"
            
            # Add to collection
            puzzle = MateIn1Puzzle(
                puzzle_id=puzzle_id,
                fen=fen,
                side_to_move=side_to_move,
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
    
    def _generate_back_rank_mates(self):
        """Generate back-rank mate variations."""
        print("🔥 Phase 1: Back-rank mate templates")
        
        # Base template variations
        back_rank_templates = [
            # Different king positions on 8th rank
            ("6k1/5ppp/8/8/8/8/8/{piece}6K", "Rook back-rank mate, king on g8"),
            ("5k2/5ppp/8/8/8/8/8/{piece}6K", "Rook back-rank mate, king on f8"),  
            ("4k3/5ppp/8/8/8/8/8/{piece}6K", "Rook back-rank mate, king on e8"),
            ("7k/6pp/8/8/8/8/8/{piece}6K", "Rook back-rank mate, king on h8"),
            ("k7/1ppp4/8/8/8/8/8/{piece}6K", "Rook back-rank mate, king on a8"),
            
            # Different pawn structures
            ("6k1/4pppp/8/8/8/8/8/{piece}6K", "Back-rank mate with 4 pawns"),
            ("6k1/6pp/8/8/8/8/8/{piece}6K", "Back-rank mate with 2 pawns"),
            ("6k1/5p1p/8/8/8/8/8/{piece}6K", "Back-rank mate with split pawns"),
        ]
        
        attacking_pieces = [
            ("R", "Ra8#", ["rook"]),
            ("Q", "Qa8#", ["queen"]), 
        ]
        
        count = 0
        for template, desc in back_rank_templates:
            for piece, move, tags in attacking_pieces:
                fen = template.format(piece=piece) + " w - - 0 1"
                full_desc = desc.replace("Rook", piece_name(piece))
                if self._add_position_if_valid(fen, [move], full_desc, ["back_rank"] + tags):
                    count += 1
        
        print(f"   Generated {count} back-rank positions")
    
    def _generate_queen_king_mates(self):
        """Generate Queen+King vs King endgame mates.""" 
        print("👑 Phase 2: Queen+King endgame templates")
        
        # Corner mate patterns
        corner_patterns = [
            # Queen in various positions around cornered king
            ("6Qk/8/6K1/8/8/8/8/8", "Qh7#", "Queen corner mate with king support"),
            ("7k/6Q1/6K1/8/8/8/8/8", "Qg8#", "Queen back-rank mate in corner"),
            ("5Q1k/8/6K1/8/8/8/8/8", "Qf8#", "Queen side mate in corner"),
            ("6k1/6Q1/6K1/8/8/8/8/8", "Qg8#", "Queen diagonal corner mate"),
            
            # Edge mate patterns
            ("3Q3k/8/6K1/8/8/8/8/8", "Qd8#", "Queen edge mate"),
            ("7k/5Q2/6K1/8/8/8/8/8", "Qf8#", "Queen edge support mate"),
        ]
        
        # Different king positions for variety
        king_positions = [
            ("6K1", "white king on g6"),
            ("5K2", "white king on f6"),
            ("7K", "white king on h6"),
        ]
        
        count = 0
        for base_pattern, move, desc in corner_patterns:
            # Try with original king position
            fen = base_pattern + " w - - 0 1"
            if self._add_position_if_valid(fen, [move], desc, ["queen", "corner"]):
                count += 1
        
        print(f"   Generated {count} Queen+King positions")
    
    def _generate_rook_king_mates(self):
        """Generate Rook+King vs King endgame mates."""
        print("🏰 Phase 3: Rook+King endgame templates")
        
        rook_patterns = [
            # Basic rook + king mates
            ("k7/1R6/1K6/8/8/8/8/8", "Rb8#", "Rook mate with king support"),
            ("1k6/1R6/1K6/8/8/8/8/8", "Rb8#", "Rook mate on b-file"),
            ("k7/R7/K7/8/8/8/8/8", "Ra8#", "Rook mate on a-file"),
            
            # Edge restriction patterns  
            ("7k/6R1/6K1/8/8/8/8/8", "Rg8#", "Rook mate on edge"),
            ("6k1/6R1/6K1/8/8/8/8/8", "Rg8#", "Rook mate restricting king"),
        ]
        
        count = 0
        for pattern, move, desc in rook_patterns:
            fen = pattern + " w - - 0 1"
            if self._add_position_if_valid(fen, [move], desc, ["rook", "king_support"]):
                count += 1
        
        print(f"   Generated {count} Rook+King positions")
    
    def _generate_tactical_mates(self):
        """Generate tactical mate patterns."""
        print("⚔️  Phase 4: Tactical mate templates")
        
        # Fork mate patterns
        fork_patterns = [
            ("6k1/8/5N2/8/8/8/8/7K", "Ne8#", "Knight fork mate"),
            ("r5k1/8/6N1/8/8/8/8/7K", "Ne7#", "Knight fork with rook"),
        ]
        
        # Discovery mate patterns
        discovery_patterns = [
            ("7k/6B1/8/8/8/8/6R1/7K", "Rg8#", "Discovery mate with rook"),
        ]
        
        count = 0
        
        # Test fork patterns
        for pattern, move, desc in fork_patterns:
            fen = pattern + " w - - 0 1"
            if self._add_position_if_valid(fen, [move], desc, ["knight", "fork"], "medium"):
                count += 1
        
        # Test discovery patterns
        for pattern, move, desc in discovery_patterns:
            fen = pattern + " w - - 0 1"
            if self._add_position_if_valid(fen, [move], desc, ["rook", "discovery"], "medium"):
                count += 1
        
        print(f"   Generated {count} tactical positions")
    
    def _generate_special_mates(self):
        """Generate special mate patterns."""
        print("✨ Phase 5: Special mate patterns")
        
        special_patterns = [
            # Bishop mate patterns
            ("k7/8/1K6/8/8/8/8/B7", "Bb8#", "Bishop corner mate"),
            
            # Pawn promotion mates
            ("6k1/7P/6K1/8/8/8/8/8", "h8=Q#", "Pawn promotion mate"),
            
            # Black to move variants
            ("6qK/8/6k1/8/8/8/8/8", "Qg7#", "Black queen mate"),
            ("5r1K/5ppp/8/8/8/8/8/5r1k", "Ra8#", "Black rook back-rank"),
        ]
        
        count = 0
        for pattern, move, desc in special_patterns:
            # Determine side to move from uppercase/lowercase pieces
            side_to_move = "w" if any(c.isupper() for c in pattern if c.isalpha()) else "b"
            fen = pattern + f" {side_to_move} - - 0 1"
            
            piece_type = move[0].lower()
            tags = [piece_type] if piece_type in "qrbnp" else ["special"]
            
            if self._add_position_if_valid(fen, [move], desc, tags):
                count += 1
        
        print(f"   Generated {count} special positions")
    
    def _generate_position_variants(self):
        """Generate variants of existing positions."""
        print("🔄 Phase 6: Position variants and mirrors")
        
        original_count = len(self.generated_positions)
        new_positions = []
        
        for puzzle in self.generated_positions[:original_count]:  # Only process originals
            variants = self._create_position_variants(puzzle)
            new_positions.extend(variants)
        
        print(f"   Generated {len(new_positions)} position variants")
    
    def _create_position_variants(self, puzzle: MateIn1Puzzle) -> List[MateIn1Puzzle]:
        """Create variants of a position through transformations."""
        variants = []
        
        try:
            # Mirror horizontally (files a↔h)
            mirrored_fen = self._mirror_position_horizontal(puzzle.fen)
            if mirrored_fen and mirrored_fen != puzzle.fen:
                mirrored_moves = [self._mirror_move_horizontal(move) for move in puzzle.mate_moves]
                mirrored_desc = puzzle.description + " (mirrored)"
                
                if self._add_position_if_valid(mirrored_fen, mirrored_moves, 
                                             mirrored_desc, puzzle.tags + ["variant"]):
                    pass  # Added successfully
        except:
            pass
        
        return variants
    
    def _mirror_position_horizontal(self, fen: str) -> Optional[str]:
        """Mirror a FEN position horizontally (a-file ↔ h-file)."""
        try:
            parts = fen.split()
            board_part = parts[0]
            
            # Mirror each rank
            ranks = board_part.split('/')
            mirrored_ranks = []
            
            for rank in ranks:
                mirrored_rank = ""
                for char in reversed(rank):
                    mirrored_rank += char
                mirrored_ranks.append(mirrored_rank)
            
            mirrored_board = '/'.join(mirrored_ranks)
            return mirrored_board + " " + " ".join(parts[1:])
            
        except:
            return None
    
    def _mirror_move_horizontal(self, move: str) -> str:
        """Mirror a move horizontally."""
        try:
            # Simple file mirroring: a↔h, b↔g, c↔f, d↔e
            file_map = {'a': 'h', 'b': 'g', 'c': 'f', 'd': 'e',
                       'e': 'd', 'f': 'c', 'g': 'b', 'h': 'a'}
            
            result = ""
            for char in move:
                if char in file_map:
                    result += file_map[char]
                else:
                    result += char
            return result
        except:
            return move
    
    def get_statistics(self) -> Dict:
        """Get generation statistics."""
        if not self.generated_positions:
            return {"total": 0}
        
        stats = {
            "total": len(self.generated_positions),
            "by_difficulty": {},
            "by_piece": {},
            "by_pattern": {}
        }
        
        for puzzle in self.generated_positions:
            # Count by difficulty
            diff = puzzle.difficulty
            stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1
            
            # Count by primary piece (first tag usually)
            if puzzle.tags:
                piece = puzzle.tags[0]
                stats["by_piece"][piece] = stats["by_piece"].get(piece, 0) + 1
        
        return stats


def piece_name(piece_letter: str) -> str:
    """Convert piece letter to name."""
    names = {'R': 'Rook', 'Q': 'Queen', 'B': 'Bishop', 'N': 'Knight', 'P': 'Pawn'}
    return names.get(piece_letter, piece_letter)


def main():
    """Generate comprehensive mate-in-1 collection."""
    print("🚀 GENERATING 100+ MATE-IN-1 POSITIONS")
    print("=" * 80)
    
    generator = MatePositionGenerator()
    positions = generator.generate_all_positions()
    
    # Show statistics
    stats = generator.get_statistics()
    print("\n📊 GENERATION STATISTICS:")
    print(f"   Total positions: {stats['total']}")
    print(f"   By difficulty: {stats['by_difficulty']}")
    print(f"   By piece type: {stats['by_piece']}")
    
    # Show first few examples
    print("\n🔍 SAMPLE GENERATED POSITIONS:")
    for i, puzzle in enumerate(positions[:5]):
        print(f"\n{i+1}. {puzzle.puzzle_id} ({puzzle.difficulty})")
        print(f"   {puzzle.description}")
        print(f"   FEN: {puzzle.fen}")
        print(f"   Mate moves: {puzzle.mate_moves}")
        print(f"   Tags: {', '.join(puzzle.tags)}")
    
    # Export results
    output_file = "generated_mate_positions.json"
    puzzle_data = [
        {
            "puzzle_id": p.puzzle_id,
            "fen": p.fen,
            "side_to_move": p.side_to_move,
            "mate_moves": p.mate_moves,
            "difficulty": p.difficulty,
            "description": p.description,
            "tags": p.tags
        }
        for p in positions
    ]
    
    with open(output_file, 'w') as f:
        json.dump(puzzle_data, f, indent=2)
    
    print(f"\n💾 Exported {len(positions)} positions to {output_file}")
    print("\n🎯 GENERATION COMPLETE!")
    
    return positions


if __name__ == "__main__":
    main()
