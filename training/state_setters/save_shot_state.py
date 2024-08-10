import numpy as np
import random

from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters import StateWrapper
from rlgym_sim.utils.math import rand_vec3
from rlgym_sim.utils.common_values import (
    BLUE_TEAM,
    CAR_MAX_SPEED,
    BALL_RADIUS,
    SIDE_WALL_X,
    BACK_WALL_Y,
    CAR_MAX_ANG_VEL,
    CEILING_Z,
)

LIM_X = SIDE_WALL_X - 1152 / 2 - BALL_RADIUS * 2**0.5
LIM_Y = BACK_WALL_Y - 1152 / 2 - BALL_RADIUS * 2
LIM_Z = CEILING_Z - BALL_RADIUS

PITCH_LIM = np.pi / 2
YAW_LIM = np.pi
ROLL_LIM = np.pi

DEG_TO_RAD = np.pi / 180

class SaveShot(StateSetter):
    def __init__(self, ball_speed=1500):
        self.ball_speed = ball_speed
        self.rng = np.random.default_rng()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            if car.team_num == BLUE_TEAM:
                if random.uniform(True, False):
                    car_x = -900
                    car_rot_yaw = 0
                else:
                    car_x = 900
                    car_rot_yaw = 180 * DEG_TO_RAD
                car_y = -5030
                car_z = 17
                car.set_pos(car_x, car_y, car_z)

                car_rot_pitch = 0
                car_rot_roll = 0
                car.set_rot(car_rot_pitch, car_rot_yaw, car_rot_roll)

                car.set_lin_vel(0.1, 0, 0)
                car.set_ang_vel(0.1, 0, 0)

                car.boost = self.rng.uniform(0.3, 0.4)

            else:
                car.set_pos(
                    random.randint(-2900, 2900),
                    random.randint(3000, 5120),
                    17
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)

        ball_x = self.rng.uniform(-800, 800)
        ball_y = -10240 / 2 + 2500
        ball_z = self.rng.uniform(500, 700)
        state_wrapper.ball.set_pos(ball_x, ball_y, ball_z)

        ball_vel_x = 0
        ball_vel_y = -self.ball_speed * 1.5
        ball_vel_z = 150
        state_wrapper.ball.set_lin_vel(ball_vel_x, ball_vel_y, ball_vel_z)
        state_wrapper.ball.set_ang_vel(0, 0, 0)