import math
import numpy as np
from typing import Any, List
from rlgym_sim.utils import common_values
from rlgym_sim.utils.gamestates import PlayerData, GameState, PhysicsObject
from rlgym_sim.utils.obs_builders import ObsBuilder

class AdvancedAdaptedObs(ObsBuilder):
    def __init__(self, pos_coef=1/2300, ang_coef=1/math.pi, lin_vel_coef=1/2300, ang_vel_coef=1/math.pi, player_padding=2, expanding=False):
        """
        :param pos_coef: Position normalization coefficient
        :param ang_coef: Rotation angle normalization coefficient
        :param lin_vel_coef: Linear velocity normalization coefficient
        :param ang_vel_coef: Angular velocity normalization coefficient
        """
        super().__init__()
        self.POS_COEF = pos_coef
        self.ANG_COEF = ang_coef
        self.LIN_VEL_COEF = lin_vel_coef
        self.ANG_VEL_COEF = ang_vel_coef
        self.PLAYER_PADDING = player_padding
        self.EXPANDING = expanding

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:

        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads

        obs = [ball.position * self.POS_COEF,
               ball.linear_velocity * self.LIN_VEL_COEF,
               ball.angular_velocity * self.ANG_VEL_COEF,
               previous_action,
               pads]

        player_car = self._add_player_to_obs(obs, player, ball, inverted)

        allies = []
        enemies = []
        ally_count = 0
        enemy_count = 0

        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
                ally_count += 1
                if ally_count > self.PLAYER_PADDING-1:
                    continue
            else:
                team_obs = enemies
                enemy_count += 1
                if enemy_count > self.PLAYER_PADDING:
                    continue

            other_car = self._add_player_to_obs(team_obs, other, ball, inverted)

            # Extra info
            team_obs.extend([
                (other_car.position - player_car.position) * self.POS_COEF,
                (other_car.linear_velocity - player_car.linear_velocity) * self.POS_COEF
            ])

        while ally_count < self.PLAYER_PADDING-1:
            self._add_dummy(allies)
            ally_count += 1
        
        while enemy_count < self.PLAYER_PADDING:
            self._add_dummy(enemies)
            enemy_count += 1

        obs.extend(allies)
        obs.extend(enemies)
        if self.EXPANDING:
            return np.expand_dims(np.concatenate(obs), 0)
        return np.concatenate(obs)

    def _add_dummy(self, obs: List):
        obs.extend([
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            [0, 0, 0, 0, 0]])
        obs.extend([np.zeros(3), np.zeros(3)])

    def _add_player_to_obs(self, obs: List, player: PlayerData, ball: PhysicsObject, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        rel_pos = ball.position - player_car.position
        rel_vel = ball.linear_velocity - player_car.linear_velocity

        obs.extend([
            rel_pos * self.POS_COEF,
            rel_vel * self.LIN_VEL_COEF,
            player_car.position * self.POS_COEF,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity * self.LIN_VEL_COEF,
            player_car.angular_velocity * self.ANG_VEL_COEF,
            [player.boost_amount,
             int(player.on_ground),
             int(player.has_flip),
             int(player.is_demoed),
             int(player.has_jump)]])

        return player_car

if __name__ == '__main__':
    pass