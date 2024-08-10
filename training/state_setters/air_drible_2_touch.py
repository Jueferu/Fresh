import random
import math
import numpy as np

from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters import StateWrapper
from rlgym_sim.utils.math import rand_vec3

from rlgym_sim.utils.common_values import BLUE_TEAM, SIDE_WALL_X, BALL_RADIUS, CEILING_Z, BACK_WALL_Y, CAR_MAX_SPEED, CAR_MAX_ANG_VEL

LIM_X = SIDE_WALL_X - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Y = BACK_WALL_Y - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Z = CEILING_Z - BALL_RADIUS

PITCH_LIM = np.pi / 2
YAW_LIM = np.pi
ROLL_LIM = np.pi

class AirDribble2Touch(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):

        for car in state_wrapper.cars:
            if car.team_num == BLUE_TEAM:

                car_x = random.randint(-3500, 3500)
                car_y = random.randint(-2000, 4000)
                car_z = np.random.uniform(500, LIM_Z)

                car.set_pos(
                    car_x,
                    car_y,
                    car_z
                )

                car_pitch_rot = random.uniform(-1, 1) * math.pi
                car_yaw_rot = random.uniform(-1, 1) * math.pi
                car_roll_rot = random.uniform(-1, 1) * math.pi

                car.set_rot(
                    car_pitch_rot,
                    car_yaw_rot,
                    car_roll_rot
                )

                car_lin_y = np.random.uniform(300, CAR_MAX_SPEED)

                car.set_lin_vel(
                    0 + random.uniform(-150, 150),
                    car_lin_y,
                    0 + random.uniform(-150, 150)
                )

                state_wrapper.ball.set_pos(
                    car_x + random.uniform(-150, 150),
                    car_y + random.uniform(0, 150),
                    car_z + random.uniform(0, 150)
                )

                ball_lin_y = car_lin_y + random.uniform(-150, 150)

                state_wrapper.ball.set_lin_vel(
                    0 + random.uniform(-150, 150),
                    ball_lin_y,
                    0 + random.uniform(-150, 150)
                )
                car.boost = np.random.uniform(0.4, 1)

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