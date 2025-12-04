# Traffic Light Reasoning Task for VMEvalKit

## 📊 Overview

The Traffic Light task evaluates video generation models' capacity for **temporal reasoning**, **rule application**, and **coordination understanding**. This task tests whether models can:

1. **Understand temporal concepts** - Comprehend countdown timers and time progression
2. **Apply relative rules** - Understand that two traffic lights are opposite (one red, the other green)
3. **Generate videos with changing numbers** - Show countdown numbers decrementing over time
4. **Demonstrate coordination reasoning** - Understand how one system's state change affects another

The traffic light tasks test temporal perception, rule application, and coordination reasoning capabilities in video models.

## 🚀 Usage

### Generate Traffic Light Tasks

Use the `create_questions.py` script to generate traffic light reasoning tasks:

```bash
# Generate traffic light tasks (default)
python examples/create_questions.py --task traffic_light

# Generate specific number of tasks (random sampling)
python examples/create_questions.py --task traffic_light --pairs-per-domain 100

python examples/generate_videos.py --model svd --task traffic_light
```

## 🎯 Task Description

### Input Components
- **First Frame**: A crossroad scene with two traffic lights showing their current states and countdown timers
- **Prompt**: Text instruction explaining the relative rule and asking to show countdown decrement and final state
- **Format**: 600×600px PNG image at 150 DPI with clear traffic light representation

### Expected Output
- **Video Sequence**: Animation showing countdown numbers decrementing (e.g., 5→4→3→2→1→0)
- **Final Frame**: Traffic lights in the correct final state after countdown reaches 0, with relative rule applied
- **Reasoning**: Proper understanding of temporal progression, countdown mechanics, and relative state coordination

### Core Features
- **Two traffic lights**: Traffic Light A and Traffic Light B
- **Relative rule**: When one is red, the other is green, and vice versa
- **Countdown timers**: Numbers displayed on traffic lights that decrement over time
- **State switching**: When countdown reaches 0, the light switches state and triggers relative rule
- **4 task types**: Different reasoning challenges from simple to complex

## 📋 Task Types

### Type 1: Basic Countdown Decrement (Simple)
Tests basic understanding of number change, time concept, and rule application.

**Scenario:**
- Traffic Light A: Red with countdown 5
- Traffic Light B: Green

**Prompt:**
```
This scene shows a crossroad with two traffic lights. The two traffic lights are opposite to each other: 
when one is red, the other is green, and vice versa. 

Currently, Traffic Light A shows red with countdown 5. Traffic Light B shows green. 

Generate a video showing the countdown number decrementing from 5 to 0, then show the final state of both traffic lights.
```

**Expected Final State:**
- Traffic Light A: Green (countdown reached 0, state switched)
- Traffic Light B: Red (relative rule applied)
- Countdown numbers: 0 or disappeared

**Cognitive Focus:**
- Number change understanding (5→4→3→2→1→0)
- Time concept (countdown represents time)
- Rule application (relative state switching)

---

### Type 2: Number Change + Time Understanding (Medium)
Tests understanding of larger countdown numbers and complete decrement process.

**Scenario:**
- Traffic Light A: Red with countdown 7 (or larger numbers: 8, 9, 10)
- Traffic Light B: Green

**Prompt:**
```
This scene shows a crossroad with two traffic lights. The two traffic lights are opposite to each other: 
when one is red, the other is green, and vice versa. 

Currently, Traffic Light A shows red with countdown 7. Traffic Light B shows green. 

Generate a video showing the countdown number decrementing from 7 to 0, then show the final state of both traffic lights.
```

**Expected Final State:**
- Traffic Light A: Green
- Traffic Light B: Red
- Countdown numbers: 0

**Cognitive Focus:**
- Larger number handling (7+ seconds)
- Complete decrement process understanding
- Enhanced time concept understanding

---

### Type 3: Dual Countdown Coordination (Hard)
Tests understanding of two countdowns simultaneously, determining which reaches zero first, and coordination understanding.

**Scenario:**
- Traffic Light A: Red with countdown 10
- Traffic Light B: Green with countdown 3

**Prompt:**
```
This scene shows a crossroad with two traffic lights. The two traffic lights are opposite to each other: 
when one is red, the other is green, and vice versa. 

Currently, Traffic Light A shows red with countdown 10. Traffic Light B shows green with countdown 3. 

Generate a video showing both countdown numbers decrementing simultaneously. When any countdown reaches 0, 
apply the relative rule to switch states. Then show the final state of both traffic lights.
```

**Expected Final State:**
- Traffic Light A: Green (because B reached 0 first, triggering relative rule)
- Traffic Light B: Red (countdown reached 0, state switched)
- Traffic Light A countdown: 7 (10-3=7) or disappeared
- Traffic Light B countdown: 0

**Key Challenge:**
The model must understand that even though Traffic Light A's countdown hasn't reached 0, it will become green because Traffic Light B reached 0 first, triggering the relative rule.

**Cognitive Focus:**
- Simultaneous countdown tracking
- Determining which countdown reaches 0 first (3 < 10)
- Coordination understanding (one change affects the other)
- Remaining time calculation (10-3=7)

---

### Type 4: Complex Time Calculation (Hard)
Tests understanding of multiple state switches and complex time sequence calculation.

**Scenario:**
- Traffic Light A: Red with countdown 8
- Traffic Light B: Green with countdown 5
- Time elapsed: 10 seconds

**Prompt:**
```
This scene shows a crossroad with two traffic lights. The two traffic lights are opposite to each other: 
when one is red, the other is green, and vice versa. 

Currently, Traffic Light A shows red with countdown 8. Traffic Light B shows green with countdown 5. 

Generate a video showing countdown numbers decrementing. When countdown reaches 0, apply the relative rule 
to switch states. Then show the final state of both traffic lights after 10 seconds.
```

**Expected Calculation:**
1. At 5 seconds: Traffic Light B countdown reaches 0 → B becomes red, A becomes green (A countdown: 3 remaining)
2. At 8 seconds (from initial): Traffic Light A countdown reaches 0 → A becomes red, B becomes green (B countdown: 2 remaining)
3. At 10 seconds: Traffic Light B countdown reaches 0 → B becomes red, A becomes green

**Expected Final State:**
- Traffic Light A: Green
- Traffic Light B: Red
- Countdown numbers: 0 or disappeared

**Cognitive Focus:**
- Multiple state switch tracking
- Complex time sequence calculation
- State change chain reasoning

---

## 🎨 Visual Design

### Traffic Light Representation

**Layout:**
- Two traffic lights positioned on a crossroad scene (left and right, or top and bottom)
- Simple crossroad background for context

**Traffic Light Structure:**
- Colored circle: Red (🔴) or Green (🟢)
- Countdown number: Displayed below or inside the light
- Clear visual distinction between red and green states

**Countdown Display:**
- Numbers displayed clearly (e.g., "5", "3", "10")
- Position: Below the traffic light or integrated into the light design
- Font: Large, readable, contrasting with background

### Example Visual States

**Initial State (Type 1):**
```
┌─────────────┐     ┌─────────────┐
│   🔴        │     │   🟢        │
│     5       │     │             │
│ Traffic A   │     │ Traffic B   │
└─────────────┘     └─────────────┘
```

**Intermediate State (Countdown Decrementing):**
```
┌─────────────┐     ┌─────────────┐
│   🔴        │     │   🟢        │
│     3       │     │             │
│ Traffic A   │     │ Traffic B   │
└─────────────┘     └─────────────┘
```

**Final State (After Countdown Reaches 0):**
```
┌─────────────┐     ┌─────────────┐
│   🟢        │     │   🔴        │
│     0       │     │     0       │
│ Traffic A   │     │ Traffic B   │
└─────────────┘     └─────────────┘
```

Or countdown numbers may disappear:
```
┌─────────────┐     ┌─────────────┐
│   🟢        │     │   🔴        │
│             │     │             │
│ Traffic A   │     │ Traffic B   │
└─────────────┘     └─────────────┘
```

## 🧠 Cognitive Abilities Tested

### 1. Temporal Reasoning
- **Countdown Understanding**: Comprehending that countdown numbers represent time remaining
- **Time Progression**: Understanding that numbers decrementing (5→4→3→2→1→0) represents time passing
- **Time Calculation**: Calculating remaining time and future states

### 2. Rule Application
- **Relative Rule**: Understanding that two traffic lights are opposite (one red, the other green)
- **State Switching**: Understanding that when countdown reaches 0, the light switches state
- **Rule Triggering**: Understanding that state switching triggers the relative rule

### 3. Coordination Understanding
- **System Interdependence**: Understanding that one system's state change affects another
- **Simultaneous Tracking**: Tracking multiple countdowns simultaneously
- **Priority Determination**: Determining which countdown reaches 0 first

### 4. Video Generation with Numbers
- **Number Animation**: Generating videos where numbers change over time
- **Visual Consistency**: Maintaining visual consistency while numbers change
- **State Synchronization**: Synchronizing number changes with state changes

## 📊 Task Generation Strategy

### Countdown Range
- **Type 1**: Small countdowns (3-7 seconds)
- **Type 2**: Medium to large countdowns (7-15 seconds)
- **Type 3**: Two countdowns with different values (one smaller, one larger)
- **Type 4**: Two countdowns with complex time calculations

### State Combinations
- **Initial States**: 
  - Light A: Red, Light B: Green (most common)
  - Light A: Green, Light B: Red (less common, for variety)
- **Countdown Values**: 
  - Type 1/2: Single countdown (3-15 seconds)
  - Type 3: Two countdowns (e.g., 3 and 10, 5 and 12)
  - Type 4: Two countdowns with time elapsed > both countdowns

## 🔗 Related Resources

- [VMEvalKit Documentation](../../../README.md)
- [Adding Tasks Guide](../../../docs/ADDING_TASKS.md)
- Other reasoning tasks: Clock, Light Sequence, Chess, Maze, Sudoku

## 📝 Design Notes

### Key Design Decisions

1. **Rule Explicit in Prompt**: The relative rule is explicitly stated in the prompt, so the task tests rule application rather than rule discovery.

2. **Focus on Final Frame**: The evaluation focuses on the final frame state, not the intermediate process, allowing models to generate videos with number changes.

3. **Number Change Emphasis**: The task specifically tests whether models can generate videos with changing numbers (countdown decrement), which is a challenging aspect of video generation.

4. **Coordination Testing**: Type 3 and 4 specifically test coordination understanding - how one system's change affects another.

5. **Temporal Reasoning**: All types test temporal reasoning, but with increasing complexity from Type 1 to Type 4.

### Future Extensions

- **Yellow Light**: Add yellow/amber light as transition state
- **Multiple Directions**: Add traffic lights for multiple directions (4-way intersection)
- **Variable Countdown Speeds**: Different countdown speeds for different lights
- **Traffic Flow Simulation**: More complex traffic scenarios

