import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData

class PlayerFaceBallReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        player_foward = player.car_data.forward()
        ball_pos = state.ball.position
        player_pos = player.car_data.position

        ball_to_player = ball_pos - player_pos
        ball_to_player_norm = np.linalg.norm(ball_to_player)
        ball_to_player /= ball_to_player_norm

        player_foward_dot = np.dot(player_foward, ball_to_player)
        reward = max(0, player_foward_dot)
        if np.isnan(reward):
            return 0
        
        return player_foward_dot