import numpy as np
import math
from typing import Optional
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import BALL_RADIUS, CAR_MAX_SPEED

BALL_MIN_HEIGHT = 300

class AerialReward(RewardFunction):
    def __init__(self):
        super().__init__()
    
    def reset(self, initial_state: GameState):
        pass
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ballPos = state.ball.position
        carPos = player.car_data.position
        
        vector_diff = ballPos - carPos
        distance = np.linalg.norm(vector_diff)
        unit_direction = vector_diff / distance if distance != 0 else np.zeros_like(vector_diff)
        
        distance_reward = math.exp(-.5 * (distance - BALL_RADIUS) / CAR_MAX_SPEED)
        
        speed_towards_ball = np.dot(unit_direction, player.car_data.velocity)
        speed_reward = speed_towards_ball / CAR_MAX_SPEED
        
        if speed_towards_ball < 0:
            return 0

        if player.on_ground:
            return 0
        
        if ballPos[2] < BALL_MIN_HEIGHT:
            return 0
        
        reward = distance_reward * speed_reward
        return max(0, reward)