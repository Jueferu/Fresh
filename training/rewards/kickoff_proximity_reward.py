import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData

class KickoffProximityReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball_vel = state.ball.linear_velocity
        if ball_vel[0] != 0 or ball_vel[1] != 0:
            return 0
        
        ball_pos = state.ball.position
        player_pos = player.car_data.position
        player_dist = np.linalg.norm(ball_pos - player_pos)
        
        nearest_enemy_dist = np.inf
        
        for other_player in state.players:
            if other_player.team_num == player.team_num:
                continue
            
            enemy_dist = np.linalg.norm(ball_pos - other_player.car_data.position)
            if enemy_dist >= nearest_enemy_dist:
                continue
            
            nearest_enemy_dist = enemy_dist

        return 1 if player_dist < nearest_enemy_dist else -.5