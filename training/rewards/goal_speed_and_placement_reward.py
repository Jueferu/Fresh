import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import BALL_RADIUS, BLUE_TEAM, BALL_MAX_SPEED

class GoalSpeedAndPlacementReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.prevScoreBlue = 0
        self.prevScoreOrange = 0
        self.prevStateBlue = None
        self.prevStateOrange = None
        self.min_height = BALL_RADIUS + 10
        self.height_reward = 1.75

    def reset(self, initial_state: GameState):
        self.prevScoreBlue = 0
        self.prevScoreOrange = 0
        self.prevStateBlue = initial_state
        self.prevStateOrange = initial_state

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        score = 0
        previous_score = 0
        previous_state = None

        if player.team_num == BLUE_TEAM:
            score = state.blue_score
            previous_state = self.prevStateBlue
            previous_score = self.prevScoreBlue

            self.prevScoreBlue = score
            self.prevStateBlue = state
        else:
            score = state.orange_score
            previous_state = self.prevStateOrange
            previous_score = self.prevScoreOrange

            self.prevScoreOrange = score
            self.prevStateOrange = state

        if (score - previous_score) <= 0:
            return 0
        
        ball_speed = np.linalg.norm(previous_state.ball.linear_velocity) / BALL_MAX_SPEED
        ball_height = previous_state.ball.position[2]

        reward = ball_speed
        if ball_height > self.min_height:
            reward *= self.height_reward
        
        return reward