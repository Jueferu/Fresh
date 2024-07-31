# Reward for child bots (just started training)
import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import CAR_MAX_SPEED

from rlgym_sim.utils.reward_functions.common_rewards import EventReward
from rlgym_sim.utils.reward_functions import CombinedReward

class InAirReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass 

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        return 0 if player.on_ground else 1

class SpeedTowardBallReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass 

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        player_vel = player.car_data.linear_velocity
        pos_diff = state.ball.position - player.car_data.position
        dist_to_ball = np.linalg.norm(pos_diff)
        dir_to_ball = pos_diff / dist_to_ball
        speed_toward_ball = np.dot(player_vel, dir_to_ball)
        
        speed_toward_ball = max(0, speed_toward_ball)
        reward = speed_toward_ball / CAR_MAX_SPEED

        if np.isnan(reward):
            reward = 0
        return reward
    
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

class BabyReward(RewardFunction):
    def __init__(self):
        super().__init()
        self.in_air_reward = InAirReward()
        self.speed_toward_ball_reward = SpeedTowardBallReward()
        self.face_toward_ball_reward = FaceTowardBallReward()
        self.goals_reward = EventReward(team_goal=1, concede=-1)
    
    def reset(self, initial_state: GameState):
        self.in_air_reward.reset(initial_state)
        self.speed_toward_ball_reward.reset(initial_state)
        self.face_toward_ball_reward.reset(initial_state)
        self.goals_reward.reset(initial_state)
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        in_air_reward = self.in_air_reward.get_reward(player, state, previous_action)
        speed_toward_ball_reward = self.speed_toward_ball_reward.get_reward(player, state, previous_action)
        face_toward_ball_reward = self.face_toward_ball_reward.get_reward(player, state, previous_action)
        goals_reward = self.goals_reward.get_reward(player, state, previous_action)
        
        reward = in_air_reward + speed_toward_ball_reward + face_toward_ball_reward + goals_reward
        return reward