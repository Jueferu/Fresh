import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import BLUE_TEAM

class PlayerBehindBallReward(RewardFunction):
    def __init__(self):
        super().__init__()
    
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM:
            return player.car_data.position[1] < state.ball.position[1]
        
        return player.car_data.position[1] > state.ball.position[1]