# Reward for child bots (can hit the ball)

import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import CAR_MAX_SPEED

from rlgym_sim.utils.reward_functions.common_rewards import EventReward
from rlgym_sim.utils.reward_functions import CombinedReward

KPH_TO_VEL = 250/9

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

class TouchBallScaledByHitForceReward(RewardFunction):

    def __init__(self):
        super().__init__()
        self.max_hit_speed = 130 * KPH_TO_VEL
        self.last_ball_vel = None
        self.cur_ball_vel = None

    # game reset, after terminal condition
    def reset(self, initial_state: GameState):
        self.last_ball_vel = initial_state.ball.linear_velocity
        self.cur_ball_vel = initial_state.ball.linear_velocity

    # happens 
    def pre_step(self, state: GameState):
        self.last_ball_vel = self.cur_ball_vel
        self.cur_ball_vel = state.ball.linear_velocity

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched:
            reward = np.linalg.norm(self.cur_ball_vel - self.last_ball_vel) / self.max_hit_speed
            return reward
        return 0
    
class ChildReward(RewardFunction):
    def __init__(self, agression_bias=.2) -> None:
        super().__init__()

        goal_reward = 1
        concede_reward = -goal_reward * (1 - agression_bias)

        # why bother doing everything manually when you can just use CombinedReward
        self.combined_reward = CombinedReward.from_zipped(
            (InAirReward(), .05),
            (SpeedTowardBallReward(), 1),
            (FaceTowardBallReward(), .5),
            (SaveBoostReward(), 1),
            (PossesionReward(), 2),
            (TouchBallScaledByHitForceReward(), 2),
            (EventReward(team_goal=goal_reward, concede=concede_reward), 10)
        )
    
    def reset(self, initial_state: GameState):
        self.combined_reward.reset(initial_state)
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        return self.combined_reward.get_reward(player, state, previous_action)
    
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.combined_reward.get_final_reward(player, state, previous_action)