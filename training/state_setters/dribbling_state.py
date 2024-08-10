import random
import numpy as np

from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters import StateWrapper

from rlgym_sim.utils.common_values import BALL_RADIUS, SIDE_WALL_X, BACK_WALL_Y

LIM_X = SIDE_WALL_X - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Y = BACK_WALL_Y - 1152 / 2 - BALL_RADIUS * 2 ** 0.5

YAW_LIM = np.pi

class DribblingStateSetter(StateSetter):
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            selected_car = random.choice(state_wrapper.blue_cars())

            # Place the ball on top of the selected car
            state_wrapper.ball.set_pos(
                x=selected_car.position[0],
                y=selected_car.position[1],
                z=selected_car.position[2] + BALL_RADIUS + 20  # Car height + ball radius
            )

            # Set the ball velocity to zero
            state_wrapper.ball.set_lin_vel(0, 0, -0.1)
            state_wrapper.ball.set_ang_vel(0, 0, -0.1)

            # Set the car's position and velocity
            selected_car.set_pos(
                x=random.uniform(-LIM_X, LIM_X),
                y=random.uniform(-LIM_Y, LIM_Y),
                z=17
            )

            selected_car.set_lin_vel(0, 0, 0)
            selected_car.set_ang_vel(0, 0, 0)

            selected_car.set_rot(
                pitch=0,
                yaw=random.uniform(-YAW_LIM, YAW_LIM),
                roll=0
            )

            selected_car.boost = .52