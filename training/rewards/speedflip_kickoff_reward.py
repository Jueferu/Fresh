import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import CAR_MAX_SPEED

def clamp(n, smallest, largest): return max(smallest, min(n, largest))
class SpeedflipKickoffReward(RewardFunction):
    def __init__(self, goal_speed=0.5):
        super().__init__()
        self.goal_speed = goal_speed

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state.ball.position[0] == 0:
            return 0
        if state.ball.position[1] == 0:
            return 0
        if player.boost_amount < 2:
            return 0
        
        vel = player.car_data.linear_velocity
        pos_diff = state.ball.position - player.car_data.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        norm_vel = vel / CAR_MAX_SPEED
        speed_rew = self.goal_speed * max(float(np.dot(norm_pos_diff, norm_vel)), 0.025)
        return speed_rew