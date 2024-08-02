import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData

KPH_TO_VEL = 250/9

class TouchBallRewardScaledByHitForce(RewardFunction):
    def __init__(self):
        super().__init__()
        self.max_hit_speed = 130 * KPH_TO_VEL
        self.last_ball_vel = None
        self.cur_ball_vel = None

    def reset(self, initial_state: GameState):
        self.last_ball_vel = initial_state.ball.linear_velocity
        self.cur_ball_vel = initial_state.ball.linear_velocity

    def pre_step(self, state: GameState):
        self.last_ball_vel = self.cur_ball_vel
        self.cur_ball_vel = state.ball.linear_velocity

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched:
            reward = np.linalg.norm(self.cur_ball_vel - self.last_ball_vel) / self.max_hit_speed
            if np.isnan(reward):
                return 0
            return reward
        return 0