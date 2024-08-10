import numpy as np
import random

from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters import StateWrapper
from rlgym_sim.utils.common_values import (
    BALL_RADIUS,
    SIDE_WALL_X,
    BACK_WALL_Y,
    CEILING_Z,
)

LIM_X = SIDE_WALL_X - 1152 / 2 - BALL_RADIUS * 2**0.5
LIM_Y = BACK_WALL_Y - 1152 / 2 - BALL_RADIUS * 2
LIM_Z = CEILING_Z - BALL_RADIUS

PITCH_LIM = np.pi / 2
YAW_LIM = np.pi
ROLL_LIM = np.pi

DEG_TO_RAD = np.pi / 180

class SideHighRoll(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        sidepick = random.randrange(2)

        side_inverter = 1
        if sidepick == 1:
            # change side
            side_inverter = -1

        # MAGIC NUMBERS ARE FROM MANUAL CALIBRATION AND WHAT FEELS RIGHT

        ball_x_pos = 3000 * side_inverter
        ball_y_pos = random.randrange(1500) - 750
        ball_z_pos = BALL_RADIUS
        state_wrapper.ball.set_pos(ball_x_pos, ball_y_pos, ball_z_pos)

        ball_x_vel = (2000 + random.randrange(1000) - 500) * side_inverter
        ball_y_vel = random.randrange(1500) - 750
        ball_z_vel = random.randrange(300)
        state_wrapper.ball.set_lin_vel(ball_x_vel, ball_y_vel, ball_z_vel)

        wall_car_blue = [car for car in state_wrapper.cars if car.team_num == 0][0]

        # blue car setup
        blue_pitch_rot = 0 * DEG_TO_RAD
        blue_yaw_rot = 90 * DEG_TO_RAD
        blue_roll_rot = 90 * side_inverter * DEG_TO_RAD
        wall_car_blue.set_rot(blue_pitch_rot, blue_yaw_rot, blue_roll_rot)

        blue_x = 4096 * side_inverter
        blue_y = -2500 + (random.randrange(500) - 250)
        blue_z = 600 + (random.randrange(400) - 200)
        wall_car_blue.set_pos(blue_x, blue_y, blue_z)
        wall_car_blue.boost = 100

        # orange car setup
        wall_car_orange = None
        if len(state_wrapper.cars) > 1:
            wall_car_orange = [car for car in state_wrapper.cars if car.team_num == 1][0]
            # orange car setup
            orange_pitch_rot = 0 * DEG_TO_RAD
            orange_yaw_rot = -90 * DEG_TO_RAD
            orange_roll_rot = -90 * side_inverter * DEG_TO_RAD
            wall_car_orange.set_rot(orange_pitch_rot, orange_yaw_rot, orange_roll_rot)

            orange_x = 4096 * side_inverter
            orange_y = 2500 + (random.randrange(500) - 250)
            orange_z = 400 + (random.randrange(400) - 200)
            wall_car_orange.set_pos(orange_x, orange_y, orange_z)
            wall_car_orange.boost = 100

        for car in state_wrapper.cars:
            if len(state_wrapper.cars) == 1 or car is wall_car_orange or car is wall_car_blue:
                continue

            # set all other cars randomly in the field
            car.set_pos(random.randrange(2944) - 1472, random.randrange(3968) - 1984, 0)
            car.set_rot(0, (random.randrange(360) - 180) * (3.1415927 / 180), 0)