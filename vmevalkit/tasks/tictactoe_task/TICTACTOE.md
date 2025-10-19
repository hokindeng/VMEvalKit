# Tic-Tac-Toe Reasoning Task

## Overview

The Tic-Tac-Toe task evaluates video models' ability to understand and solve strategic game scenarios. This task tests fundamental reasoning capabilities including strategic thinking, pattern recognition, game theory, and logical decision-making.

## Task Structure

Each tic-tac-toe task consists of:

- **Initial Board State**: A 3x3 tic-tac-toe board with some moves already played
- **Player to Move**: Either X or O, indicating whose turn it is
- **Objective**: Find the optimal move (winning move, blocking move, or strategic play)
- **Final Board State**: The board after making the optimal move

## Reasoning Capabilities Tested

### 1. Strategic Thinking
- Understanding game theory principles
- Recognizing winning patterns
- Identifying defensive moves
- Planning multiple moves ahead

### 2. Pattern Recognition
- Detecting potential winning lines
- Recognizing fork opportunities
- Identifying blocking threats
- Understanding board symmetries

### 3. Logical Reasoning
- Analyzing move consequences
- Evaluating multiple options
- Applying constraint satisfaction
- Making optimal decisions under uncertainty

### 4. Visual Understanding
- Interpreting board state from visual input
- Tracking piece positions
- Understanding spatial relationships
- Recognizing game progression

## Task Types

### Winning Move (40% of tasks)
- Player can win in one move
- Tests ability to recognize immediate winning opportunities
- Requires understanding of line completion

### Blocking Move (30% of tasks)
- Player must prevent opponent from winning
- Tests defensive strategic thinking
- Requires threat recognition

### Optimal Move (20% of tasks)
- Strategic positioning for long-term advantage
- Tests game theory understanding
- Requires evaluation of multiple factors

### Fork Setup (10% of tasks)
- Advanced scenarios creating multiple threats
- Tests sophisticated strategic thinking
- Requires complex pattern recognition

## Difficulty Levels

### Easy
- Clear winning or blocking moves
- Few pieces on the board
- Obvious strategic choices

### Medium
- Multiple valid moves to consider
- Moderate board complexity
- Some strategic depth required

### Hard
- Complex strategic scenarios
- Multiple competing objectives
- Advanced pattern recognition needed

## Visual Design

The tic-tac-toe boards are rendered with:
- **X**: Red diagonal lines forming an X
- **O**: Blue circles
- **Grid**: Black lines forming 3x3 grid
- **Highlighting**: Yellow background for winning lines
- **Clean Layout**: Professional appearance suitable for video generation

## Evaluation Criteria

Video models are evaluated on their ability to:

1. **Correctly Identify** the optimal move
2. **Demonstrate Understanding** of game state
3. **Show Reasoning Process** through video generation
4. **Apply Strategy** appropriate to the scenario
5. **Visualize Solution** clearly in the final frame

## Usage in VMEvalKit

The tic-tac-toe task integrates seamlessly with VMEvalKit's evaluation framework:

```python
# Generate tic-tac-toe dataset
python vmevalkit/runner/create_dataset.py --pairs-per-domain 50

# Run inference on tic-tac-toe tasks
python vmevalkit/runner/inference.py --model [model_name] --domain tictactoe
```

## Technical Implementation

- **Board Representation**: 3x3 list with "X", "O", or "" for empty
- **Move Generation**: Algorithmic generation of strategic scenarios
- **Image Creation**: Matplotlib-based visualization with clean styling
- **Metadata**: Rich task information including difficulty and scenario type

## Future Extensions

Potential enhancements to the tic-tac-toe task:

- **Larger Boards**: 4x4 or 5x5 variants for increased complexity
- **Different Rules**: Variants like "first to get 3 in a row" on larger boards
- **Multi-Move Sequences**: Tasks requiring multiple moves
- **Tournament Scenarios**: Complex endgame positions
- **Custom Symbols**: Different visual representations for X and O

## Research Applications

This task is valuable for research in:

- **Game AI**: Understanding strategic decision-making
- **Visual Reasoning**: Interpreting game states from images
- **Pattern Recognition**: Detecting winning patterns and threats
- **Strategic Planning**: Multi-step reasoning and planning
- **Constraint Satisfaction**: Applying rules and constraints in problem-solving
