import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData

class KickoffProximityReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if not (state.ball.position[0] == 0 and state.ball.position[1] == 0):
            return 0
        
        player_pos = np.array(player.car_data.position)
        ball_pos = np.array(state.ball.position)
        player_dist_to_ball = np.linalg.norm(player_pos - ball_pos)

        opponent_distances = []
        for p in state.players:
            if p.team_num != player.team_num:
                opponent_pos = np.array(p.car_data.position)
                opponent_dist_to_ball = np.linalg.norm(opponent_pos - ball_pos)
                opponent_distances.append(opponent_dist_to_ball)

        if opponent_distances and player_dist_to_ball < min(opponent_distances):
            return 1
        return -1