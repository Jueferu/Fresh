import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData

class SaveBoostReward(RewardFunction):
    def __init__(self):
        super().__init__()
    
    def reset(self, initial_state: GameState):
        pass
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return np.sqrt(player.boost_amount)