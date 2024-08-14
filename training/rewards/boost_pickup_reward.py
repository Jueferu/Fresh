import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData

#https://github.com/redd-rl/apollo-cpp/blob/main/CustomRewards.h#L204
class BoostPickupReward(RewardFunction):
    def __init__(self, small=3, big=10):
        super().__init__()
        self.last_state = None
        self.small = small
        self.big = big

    def reset(self, initial_state: GameState):
        self.last_state = initial_state

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0
        last_player = None
        
        for old_player in self.last_state.players:
            if old_player.car_id == player.car_id:
                last_player = old_player
                
        if not last_player:
            return 0
        
        old_boost = last_player.boost_amount
        new_boost = player.boost_amount
        if new_boost <= 0:
            return 0
        
        reward += new_boost * self.big
        if new_boost < 0.98 and old_boost < 0.88:
            reward += new_boost * self.small
        
        self.last_state = state
        return reward