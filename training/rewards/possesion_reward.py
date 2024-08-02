import numpy as np
from rlgym_sim.utils import RewardFunction, math
from rlgym_sim.utils.gamestates import GameState, PlayerData

from rlgym_sim.utils.common_values import CAR_MAX_SPEED, BALL_MAX_SPEED

class PossesionReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        allies = [player for player in state.players if player.team_num == player.team_num]
        enemies = [player for player in state.players if player.team_num != player.team_num]

        ball_pos = state.ball.position
        closest_ally = min(allies, key=lambda x: np.linalg.norm(x.car_data.position - ball_pos))
        closest_enemy = min(enemies, key=lambda x: np.linalg.norm(x.car_data.position - ball_pos))

        ally_dist = np.linalg.norm(closest_ally.car_data.position - ball_pos)
        enemy_dist = np.linalg.norm(closest_enemy.car_data.position - ball_pos)
        total_dist = ally_dist + enemy_dist
        diff_ratio = (ally_dist - enemy_dist) / total_dist

        reward = max(0, diff_ratio)
        if np.isnan(reward):
            return 0
        
        return reward
        