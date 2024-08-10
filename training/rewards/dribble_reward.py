import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import CAR_MAX_SPEED

MIN_BALL_HEIGHT = 109.0
MAX_BALL_HEIGHT = 180.0
MAX_DISTANCE = 197.0
SPEED_MATCH_FACTOR = 2.0

class DribbleReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        player_pos = player.car_data.position
        ball_pos = state.ball.position
        ball_height = ball_pos[2]

        if not player.on_ground:
            return 0
        
        if not (ball_height >= MIN_BALL_HEIGHT):
            return 0
        
        if not (ball_height <= MAX_BALL_HEIGHT):
            return 0
        
        if np.linalg.norm(player_pos - ball_pos) > MAX_DISTANCE:
            return 0

        player_speed = np.linalg.norm(player.car_data.linear_velocity)
        ball_speed = np.linalg.norm(state.ball.linear_velocity)
        speed_match_reward = 1 - ((player_speed / CAR_MAX_SPEED) + SPEED_MATCH_FACTOR * (1 - abs(player_speed - ball_speed) / (player_speed + ball_speed)))
        if np.isnan(speed_match_reward):
            return 0
        
        return speed_match_reward