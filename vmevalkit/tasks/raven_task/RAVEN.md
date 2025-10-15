# Raven's Progressive Matrices Task Documentation

## Overview

The Raven's Progressive Matrices (RPM) task evaluates video generation models' ability to demonstrate abstract reasoning and pattern recognition by completing visual matrix puzzles. This task tests fundamental cognitive capabilities including pattern detection, logical inference, and abstract rule application - core components of visual intelligence.

## Task Description

### Core Challenge
Models must:
1. **Pattern Recognition**: Identify the underlying rule governing the matrix
2. **Abstract Reasoning**: Apply the discovered rule to determine the missing element
3. **Visual Completion**: Generate a video showing the reasoning from incomplete to complete matrix
4. **Logical Consistency**: Ensure the solution follows the established pattern

### Visual Format
- **3x3 Matrix Grid**: Nine cells arranged in a square grid
- **Missing Element**: Bottom-right cell (position 9) is always the unknown
- **Pattern Types**: Shapes, numbers, rotations, colors, and combinations
- **Clean Visualization**: High-contrast black and white design with clear geometric shapes

## Data Structure

### RavenTaskPair
Each task consists of a pair of matrix images and a text prompt:

```python
{
    "id": str,                     # Unique identifier (e.g., "raven_0001")
    "prompt": str,                  # Instructions for the video model
    "first_image_path": str,        # Path to incomplete matrix image
    "final_image_path": str,        # Path to completed matrix image
    "domain": str,                  # "raven"
    "task_category": str,           # "AbstractReasoning"
    "difficulty": str,              # "easy", "medium", or "hard"
    "raven_data": {                # Task-specific metadata
        "rule": str,                        # Description of the pattern rule
        "rule_type": str,                   # Type of pattern (shape/number/rotation/etc.)
        "matrix_size": str,                 # "3x3"
        "seed": int                         # Random seed for reproducibility
    },
    "created_at": str              # ISO timestamp
}
```

### RavenDataset
A collection of Raven task pairs with metadata:

```python
{
    "name": "raven_tasks",
    "description": "Raven Progressive Matrices visual reasoning tasks (N pairs)",
    "created_at": str,
    "total_pairs": int,
    "pairs": [RavenTaskPair, ...],
    "generation_info": {
        "generator": "RPMPuzzleGenerator",
        "tile_size": 150,
        "matrix_size": "3x3",
        "difficulty_distribution": {
            "easy": int,
            "medium": int,
            "hard": int
        }
    }
}
```

## Pattern Types

### 1. Shape Progression
- **Description**: Shapes change systematically across rows or columns
- **Example**: Circle → Square → Triangle pattern repeated
- **Difficulty**: Easy
- **Rule Application**: Identify sequence and continue pattern

### 2. Number Progression
- **Description**: Quantity of elements increases or decreases systematically
- **Example**: 1 shape → 2 shapes → 3 shapes per cell
- **Difficulty**: Medium
- **Rule Application**: Count elements and apply arithmetic progression

### 3. Rotation Pattern
- **Description**: Elements rotate by consistent angles
- **Example**: Shapes rotate 90° clockwise across the matrix
- **Difficulty**: Medium
- **Rule Application**: Identify rotation direction and angle

### 4. Color/Fill Pattern
- **Description**: Colors or fill patterns change systematically
- **Example**: Empty → Filled → Shaded progression
- **Difficulty**: Easy
- **Rule Application**: Identify fill sequence and continue

### 5. Combination Pattern
- **Description**: Multiple rules apply simultaneously
- **Example**: Shape changes AND number increases
- **Difficulty**: Hard
- **Rule Application**: Identify and apply multiple rules together

## Visual Elements

### Shapes Used
- **Basic Geometrics**: Circle, Square, Triangle, Diamond
- **Complex Forms**: Cross, Star
- **Variations**: Different sizes, orientations, and positions

### Visual Properties
- **Colors**: Black, gray, darkgray
- **Fills**: None (outline only), lightgray, white
- **Line Width**: Consistent 2-3 pixel borders
- **Tile Size**: 150x150 pixels per matrix cell
- **Total Size**: 450x450 pixels for complete 3x3 matrix

## Prompts

### Standardized Prompt
The Raven task uses a single standardized prompt:

**"This is Raven's Progressive Matrices like task. Complete the missing pattern in this 3x3 matrix."**

This prompt:
- Clearly identifies the task type
- Specifies the goal (complete the pattern)
- References the familiar RPM framework
- Indicates matrix size for clarity

## Difficulty Levels

### Easy
- **Pattern Types**: Shape progression, color patterns
- **Characteristics**: Single rule, clear progression
- **Cognitive Load**: Low - pattern immediately apparent
- **Example**: Simple left-to-right shape sequence

### Medium
- **Pattern Types**: Number progression, rotation
- **Characteristics**: Requires counting or spatial transformation
- **Cognitive Load**: Moderate - needs systematic analysis
- **Example**: 90° rotation pattern across rows

### Hard
- **Pattern Types**: Combination patterns
- **Characteristics**: Multiple simultaneous rules
- **Cognitive Load**: High - requires identifying and coordinating multiple patterns
- **Example**: Shape changes while number increases and rotation occurs

## Generation Process

### 1. Pattern Selection
```python
rule_types = [
    "shape_progression",
    "number_progression",
    "rotation",
    "color_pattern",
    "combination"
]
```

### 2. Matrix Construction
- Generate 8 cells following the selected rule
- Leave position 9 (bottom-right) for completion
- Ensure pattern is unambiguous and has unique solution

### 3. Image Rendering
- Create incomplete matrix (first frame)
- Generate complete matrix with solution (final frame)
- Use consistent 150x150 pixel tiles
- Apply clean, high-contrast styling

### 4. Quality Validation
- Verify pattern consistency
- Ensure solution uniqueness
- Check visual clarity
- Validate difficulty classification

## Usage Examples

### Basic Dataset Generation
```python
from vmevalkit.tasks.raven_task import create_dataset

# Generate 50 Raven Progressive Matrices tasks
dataset = create_dataset(num_samples=50)
print(f"Created {len(dataset['pairs'])} RPM tasks")

# Check difficulty distribution
for pair in dataset['pairs']:
    print(f"Task {pair['id']}: {pair['raven_data']['rule_type']} - {pair['difficulty']}")
```

### Visualizing Solutions
```python
from vmevalkit.tasks.raven_task import visualize_solution_process

# Create visualization showing solution process
viz_path = visualize_solution_process(
    task_id="raven_0001",
    output_dir="output/raven_solutions"
)
```

### Custom Pattern Generation
```python
from vmevalkit.tasks.raven_task.rpm_generator import RPMPuzzleGenerator

# Create generator with custom settings
generator = RPMPuzzleGenerator(tile_size=150, seed=42)

# Generate specific pattern type
matrix, rule = generator.generate_pattern_matrix()
print(f"Generated matrix with rule: {rule}")
```

## Evaluation Criteria

### Pattern Recognition
- **Correct Rule Identification**: Does the model identify the underlying pattern?
- **Rule Application**: Can the model apply the rule to find the missing element?
- **Consistency**: Is the solution consistent with all matrix elements?

### Visual Reasoning
- **Spatial Understanding**: Accurate interpretation of 2D arrangements
- **Abstract Thinking**: Going beyond surface features to find rules
- **Systematic Analysis**: Checking rows, columns, and diagonals

### Solution Quality
- **Correctness**: The completed pattern follows the established rule
- **Uniqueness**: Solution is the only valid completion
- **Visual Clarity**: Generated video clearly shows the reasoning

## Expected Model Behavior

```
INPUT:  Incomplete 3x3 matrix + "This is Raven's Progressive Matrices like task. 
        Complete the missing pattern in this 3x3 matrix."

ANALYSIS: Model should:
          - Examine rows for horizontal patterns
          - Check columns for vertical patterns  
          - Consider diagonal relationships
          - Identify the governing rule

OUTPUT: Video showing:
        - Initial incomplete matrix (8 cells filled)
        - Reasoning process (optional intermediate frames)
        - Final complete matrix (all 9 cells filled correctly)

VALIDATION: ✅ Pattern rule maintained  ✅ Logical consistency  ✅ Visual clarity
```

## File Organization

```
vmevalkit/tasks/raven_task/
├── __init__.py                  # Module exports
├── raven_reasoning.py          # Main task generation logic
├── rpm_generator.py            # RPM puzzle generator
└── RAVEN.md                    # This documentation

data/
├── raven_tasks/
│   └── raven_tasks.json       # Dataset metadata
└── generated_raven/
    ├── raven_0000_first.png   # Incomplete matrices
    ├── raven_0000_final.png   # Complete solutions
    └── ...                     # Additional pairs
```

## Research Applications

### Cognitive Assessment
- **Abstract Reasoning**: Fundamental pattern recognition abilities
- **Logical Inference**: Rule discovery and application
- **Visual Intelligence**: Non-verbal reasoning capabilities

### Model Capabilities
- **Systematic Thinking**: Methodical pattern analysis
- **Generalization**: Applying learned rules to new instances
- **Multi-step Reasoning**: Building from pattern to solution

### Benchmarking Value
- **Standardized Format**: Consistent with classical RPM tests
- **Difficulty Scaling**: Progressive complexity levels
- **Clear Metrics**: Objective success/failure criteria

## Technical Specifications

### Dependencies
```python
# Core requirements
PIL (Pillow)>=8.3.0     # Image generation and manipulation
numpy>=1.21.0           # Numerical operations
random                  # Pattern randomization
```

### Performance Characteristics
- **Generation Speed**: ~0.5-1 second per puzzle
- **Memory Usage**: ~10MB peak during generation
- **Image Quality**: 450x450 RGB, ~50KB per image
- **Success Rate**: 100% valid puzzle generation

## Related Work

Inspired by:
- **Raven's Progressive Matrices (1938)**: Original non-verbal intelligence test
- **Abstract Reasoning Corpus (ARC)**: Modern AI reasoning benchmark
- **PGM Dataset**: Procedurally generated matrices for machine learning

## Future Extensions

### Enhanced Patterns
- **Analogy Patterns**: A:B::C:? relationships
- **Distributed Rules**: Patterns spanning multiple cells
- **Hidden Patterns**: Rules requiring deeper analysis

### Increased Complexity
- **Larger Matrices**: 4x4 or 5x5 grids
- **Multiple Missing Cells**: More than one unknown
- **Conditional Rules**: If-then pattern logic

### Interactive Features
- **Hint System**: Progressive clues for difficult patterns
- **Multiple Choice**: Selecting from candidate solutions
- **Explanation Generation**: Models describe discovered rules

## Conclusion

The Raven's Progressive Matrices task provides a rigorous evaluation of abstract reasoning capabilities in video generation models. By requiring pattern recognition, rule application, and visual completion, this task tests fundamental aspects of visual intelligence that are crucial for advanced AI systems.

The standardized format, clear evaluation criteria, and scalable difficulty make this an essential component of the VMEvalKit reasoning evaluation suite, complementing spatial (maze, rotation) and strategic (chess, sudoku) reasoning tasks.
