import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import CAR_MAX_SPEED

class PlayerVelocityReward(RewardFunction):
    def __init__(self, player_index: int):
        super().__init__()
        self.player_index = player_index

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        player_velocity = np.linalg.norm(player.car_data.linear_velocity)
        reward = player_velocity / CAR_MAX_SPEED
        if np.isnan(reward):
            return 0
        
        return reward