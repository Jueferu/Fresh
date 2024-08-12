import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import CAR_MAX_SPEED

def clamp(n, smallest, largest): return max(smallest, min(n, largest))

#https://github.com/redd-rl/apollo-cpp/blob/main/CustomRewards.h#L18
class SpeedflipKickoffReward(RewardFunction):
    def __init__(self, goal_speed=0.5):
        super().__init__()
        self.goal_speed = goal_speed

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball_vel = state.ball.linear_velocity
        player_to_ball_norm = np.linalg.norm(state.ball.position - player.car_data.position)
        player_vel = player.car_data.linear_velocity
        
        if not (ball_vel[0] == 0 and ball_vel[1] == 0):
            return 0
        
        boost = player.boost_amount / 100
        if boost >= 0.02:
            return 0
        
        reward = max(0, np.dot(player_vel, player_to_ball_norm / CAR_MAX_SPEED))
        return reward