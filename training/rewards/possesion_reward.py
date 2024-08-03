import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import BALL_RADIUS, BLUE_TEAM, BALL_MAX_SPEED

class PossesionReward(RewardFunction):
    def __init__(self, min_dist: float = 200):
        super().__init__()
        self.prevTeamTouch = -1
        self.stacking = 0
        self.min_dist = min_dist
    
    def reset(self, initial_state: GameState):
        self.prevTeamTouch = -1
        self.stacking = 0
    
    def pre_step(self, state: GameState):
        for player in state.players:
            if not player.ball_touched:
                continue

            if self.prevTeamTouch != player.team_num:
                self.prevTeamTouch = player.team_num
                self.stacking = 0
                continue

            self.stacking += 1

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        dist_to_ball = np.linalg.norm(player.car_data.position - state.ball.position)
        if not self.prevTeamTouch == player.team_num:
            return 0
        if not dist_to_ball < self.min_dist:
            return 0
        
        for enemy in state.players:
            if enemy.team_num == player.team_num:
                continue

            enemy_dist_to_ball = np.linalg.norm(enemy.car_data.position - state.ball.position)
            if enemy_dist_to_ball < dist_to_ball:
                return 0
        
        self.stacking += 1
        reward = self.stacking / 10
        return reward