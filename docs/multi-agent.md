# Multi-Agent Implementation Plan for Agent Arcade

## Quick Start Guide

### Training Your First Multi-Agent Game

Agent Arcade supports training AI agents to play against each other in versus games. Here's how to get started with two-player Pong:

```bash
# 1. List available versus (multi-agent) games
agent-arcade list-games --type versus

# 2. Train an AI for the left paddle
agent-arcade train pong-versus --train-paddle left --render

# 3. Train an AI for the right paddle
agent-arcade train pong-versus --train-paddle right --render

# 4. Have them compete against each other
agent-arcade compete pong-versus \
    --left-ai models/pong_left_final.zip \
    --right-ai models/pong_right_final.zip \
    --render
```

### Understanding Multi-Agent Games

In multi-agent games:
- Each position (e.g., left paddle, right paddle) needs its own trained AI
- During training, one position is controlled by the learning AI while others use random actions
- In competition, each position can be controlled by a trained AI
- Models are saved with position-specific names for easy identification

### Available Multi-Agent Games

1. **Pong Versus** (`pong-versus`)
   - Two-player competitive game
   - Positions: left paddle, right paddle
   - Scoring: +1 for scoring, -1 for opponent scoring

2. **Boxing** (`boxing-versus`) - Coming soon
   - Two-player competitive game
   - Positions: blue boxer, red boxer
   - Scoring: Points for successful hits

3. **Space Invaders Co-op** (`space-invaders-coop`) - Coming soon
   - Two-player cooperative game
   - Positions: left ship, right ship
   - Scoring: Shared points for destroying aliens

## Implementation Plan

## Overview
This document outlines the focused implementation of multi-agent support in Agent Arcade, starting with three classic Atari games from PettingZoo.

## Phase 1: Core Multi-Agent Support

### 1.1 Infrastructure Updates
- [ ] Add PettingZoo dependency and environment registration
- [ ] Extend existing `GameInterface` for multi-agent support
  ```python
  class GameInterface:
      # Add to existing interface
      @property
      def num_players(self) -> int:
          """Number of players supported."""
          return 1  # Default to single player
      
      @property
      def is_multi_agent(self) -> bool:
          """Whether game supports multiple agents."""
          return False  # Default to single agent
      
      @property
      def player_roles(self) -> List[str]:
          """Available player roles for multi-agent games."""
          return ["first_0"] if self.num_players == 1 else ["first_0", "second_0"]
  ```

### 1.2 Game Implementations

#### Pong (2-player)
- [ ] Implement two-player Pong using PettingZoo's `pong_v3`
  - Action space: 6 actions (No-op, Fire, Move right/left, Fire right/left)
  - Observation shape: (210, 160, 3)
  - Scoring: +1 for scoring, -1 for opponent scoring
  - Player roles: "first_0" (left paddle), "second_0" (right paddle)

#### Boxing (2-player)
- [ ] Implement two-player Boxing using PettingZoo's `boxing_v2`
  - Action space: 18 actions (movement and punch combinations)
  - Observation shape: (210, 160, 3)
  - Scoring: Points for successful punches (1-2 points, 100 for KO)
  - Player roles: "first_0" (blue boxer), "second_0" (red boxer)

#### Space Invaders (2-player)
- [ ] Implement two-player Space Invaders using PettingZoo's `space_invaders_v2`
  - Action space: 6 actions (standard Space Invaders controls)
  - Observation shape: (210, 160, 3)
  - Scoring: Points for destroying aliens, cooperative play
  - Player roles: "first_0" (left ship), "second_0" (right ship)

## Phase 2: Integration with Existing Systems

### 2.1 Evaluation Pipeline
- [ ] Update evaluation to handle multiple agents
  ```python
  @dataclass 
  class EvaluationResult:
      """Result of an evaluation."""
      score: float
      player_scores: Optional[Dict[str, float]] = None
      winner: Optional[str] = None
      episode_length: int = 0
      additional_metrics: Optional[Dict[str, Any]] = None
  ```
- [ ] Maintain compatibility with single-agent games
- [ ] Add multi-agent specific metrics (win rates, player scores)

### 2.2 Staking System
- [ ] Update `StakeRecord` to support multi-agent games:
  ```python
  @dataclass
  class StakeRecord:
      """Record of a stake placed on a game."""
      game: str
      model_path: str
      amount: float
      target_score: float
      status: str  # pending, completed, failed
      player_role: Optional[str] = None
      is_multi_agent: bool = False
  ```
- [ ] Extend staking function to handle player roles:
  ```python
  def stake_on_game(
      wallet: Optional['NEARWallet'], 
      game_name: str,
      model_path: Path, 
      amount: float, 
      target_score: float,
      player_role: Optional[str] = None,
      is_multi_agent: bool = False
  ) -> None:
  ```
- [ ] Keep individual performance tracking in multi-agent games
- [ ] Maintain backward compatibility with single-agent stakes

### 2.3 CLI Updates
- [ ] Add multi-agent specific commands:
  ```bash
  # Training with role specification
  agent-arcade train pong-2p --role first_0 --render
  
  # Evaluation with role
  agent-arcade evaluate boxing-2p --role second_0 --model models/boxing_final.zip
  
  # Staking with role
  agent-arcade stake space-invaders-2p --role first_0 --amount 10 --target-score 1000
  ```
- [ ] Update help documentation for multi-agent features
- [ ] Add role validation in CLI commands

## Phase 3: Documentation and Testing

### 3.1 Documentation
- [ ] Update game addition guide for multi-agent support
- [ ] Add multi-agent game examples
- [ ] Document PettingZoo integration
- [ ] Add multi-agent staking documentation

### 3.2 Testing
- [ ] Add multi-agent specific tests
- [ ] Verify compatibility with existing games
- [ ] Test staking system with multi-agent games
- [ ] Validate player role handling

## Implementation Notes

### Environment Setup
```python
# Standard preprocessing for all games
env = supersuit.max_observation_v0(env, 2)  # Frame flickering fix
env = supersuit.frame_skip_v0(env, 4)  # Performance
env = supersuit.resize_v1(env, 84, 84)  # Standard size
env = supersuit.frame_stack_v1(env, 4)  # Temporal info
```

### Game-Specific Details

#### Pong
- Simple scoring system (+1/-1)
- Two-player competitive
- 6 basic actions
- Good starting point for multi-agent implementation

#### Boxing
- More complex action space (18 actions)
- Direct competition
- Points-based scoring
- Natural progression from Pong

#### Space Invaders
- Cooperative gameplay option
- Shared scoring system
- Familiar mechanics from single-player version
- Bridge between single and multi-agent implementations

## Benefits of This Approach
1. Builds on existing infrastructure
2. Minimal changes to core systems
3. Clear progression path
4. Maintains familiar user experience
5. Leverages proven game implementations
6. Easy to extend later

## Future Considerations
- Additional multi-agent games
- Team-based features
- Tournament support
- Advanced training algorithms

## Testing Commands
```bash
# Training with role specification
agent-arcade train pong-2p --role first_0 --render

# Evaluation with role
agent-arcade evaluate boxing-2p --role second_0 --model models/boxing_final.zip

# Staking with role
agent-arcade stake space-invaders-2p --role first_0 --amount 10 --target-score 1000
```
