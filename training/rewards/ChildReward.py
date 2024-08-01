# Reward for child bots (can hit the ball)

import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData

from rlgym_sim.utils.reward_functions.common_rewards import EventReward, VelocityBallToGoalReward, LiuDistanceBallToGoalReward
from rlgym_sim.utils.reward_functions import CombinedReward

class InAirReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass 

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        return 0 if player.on_ground else 1
    
class FaceTowardBallReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass 

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        player_forward = player.car_data.forward()
        pos_diff = state.ball.position - player.car_data.position
        dir_to_ball = pos_diff / np.linalg.norm(pos_diff)
        
        reward = np.dot(player_forward, dir_to_ball)
        if np.isnan(reward):
            reward = 0
        return reward

class SaveBoostReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return np.sqrt(player.boost_amount)

class PossesionReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # get closest player to the ball
        closest_player = min(state.players, key=lambda x: np.linalg.norm(x.car_data.position - state.ball.position))
        return 1 if closest_player.team_num == player.team_num else 0

class DistanceFromTeammatesReward(RewardFunction):
    def __init__(self, max_distance=100):
        self.max_distance = max_distance

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        teammates = [p for p in state.players if p.team_num == player.team_num and p.car_id != player.car_id]
        closest_teamate = min(teammates, key=lambda x: np.linalg.norm(x.car_data.position - player.car_data.position))
        return 1 - np.linalg.norm(closest_teamate.car_data.position - player.car_data.position) / self.max_distance
class ChildReward(RewardFunction):
    def __init__(self, agression_bias=.2) -> None:
        super().__init__()

        goal_reward = 1
        concede_reward = -goal_reward * (1 - agression_bias)

        # why bother doing everything manually when you can just use CombinedReward
        self.combined_reward = CombinedReward.from_zipped(
            (InAirReward(), .05),
            (VelocityBallToGoalReward(), 1),
            (FaceTowardBallReward(), .5),
            (SaveBoostReward(), .5),
            (PossesionReward(), .5),
            (LiuDistanceBallToGoalReward(), 1.5),
            (DistanceFromTeammatesReward(), 1)
            (EventReward(team_goal=goal_reward, concede=concede_reward), 10)
        )
    
    def reset(self, initial_state: GameState):
        self.combined_reward.reset(initial_state)
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        return self.combined_reward.get_reward(player, state, previous_action)
    
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.combined_reward.get_final_reward(player, state, previous_action)