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
    BALL_MAX_SPEED,
    CAR_MAX_ANG_VEL,
    CEILING_Z,
    ORANGE_TEAM,
)

LIM_X = SIDE_WALL_X - 1152 / 2 - BALL_RADIUS * 2**0.5
LIM_Y = BACK_WALL_Y - 1152 / 2 - BALL_RADIUS * 2
LIM_Z = CEILING_Z - BALL_RADIUS

PITCH_LIM = np.pi / 2
YAW_LIM = np.pi
ROLL_LIM = np.pi

class ShotState(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            if car.team_num == BLUE_TEAM:
                car.set_pos(random.uniform(-4096, 4096), random.uniform(0, 3000), 17)

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

                state_wrapper.ball.set_pos(
                    x=np.random.uniform(
                        max(car.position.item(0) - 1000, -LIM_X),
                        min(car.position.item(0) + 1000, LIM_X),
                    ),
                    y=np.random.uniform(
                        car.position.item(1) + 1000, car.position.item(1) + 100
                    ),
                    z=np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z / 2),
                )

                ball_speed = np.random.exponential(
                    -(BALL_MAX_SPEED / 3) / np.log(1 - 0.999)
                )
                vel = rand_vec3(min(ball_speed, BALL_MAX_SPEED / 3))
                state_wrapper.ball.set_lin_vel(*vel)

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL + 0.5))
                state_wrapper.ball.set_ang_vel(*ang_vel)

            if car.team_num == ORANGE_TEAM:
                car.set_pos(random.randint(-2900, 2900), random.randint(3000, 5120), 17)

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
