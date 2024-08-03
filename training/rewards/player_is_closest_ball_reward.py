import numpy as np
import math
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData

from rlgym_sim.utils.common_values import CAR_MAX_SPEED, BALL_MAX_SPEED

class PlayerIsClosestBallReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        closest_player = None
        closest_distance = math.inf

        for p in state.players:
            distance = np.linalg.norm(p.car_data.position - state.ball.position)
            if distance >= closest_distance:
                continue
            
            closest_distance = distance
            closest_player = p
        
        return 1 if closest_player == player else 0