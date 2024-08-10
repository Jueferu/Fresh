import numpy as np
from collections import namedtuple

from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters import StateWrapper
from rlgym_sim.utils.state_setters.wrappers import CarWrapper
from rlgym_sim.utils.math import rand_vec3

def mirror(car: CarWrapper, ball_x, ball_y):
    my_car = namedtuple('my_car', 'pos lin_vel rot ang_vel')
    if ball_x == ball_y == 0:
        my_car.pos = -car.position[0], -car.position[1], car.position[2]
        my_car.lin_vel = -car.linear_velocity[0], -car.linear_velocity[1], car.linear_velocity[2]
        my_car.rot = car.rotation[0], -car.rotation[1], car.rotation[2]
        my_car.ang_vel = -car.angular_velocity[0], -car.angular_velocity[1], car.angular_velocity[2]
    elif ball_x == 0:
        my_car.pos = -car.position[0], car.position[1], car.position[2]
        my_car.lin_vel = -car.linear_velocity[0], car.linear_velocity[1], car.linear_velocity[2]
        my_car.rot = car.rotation[0], -car.rotation[1], car.rotation[2]
        my_car.ang_vel = -car.angular_velocity[0], -car.angular_velocity[1], car.angular_velocity[2]
    elif ball_y == 0:
        my_car.pos = car.position[0], -car.position[1], car.position[2]
        my_car.lin_vel = -car.linear_velocity[0], car.linear_velocity[1], car.linear_velocity[2]
        my_car.rot = car.rotation[0], -car.rotation[1], car.rotation[2]
        my_car.ang_vel = -car.angular_velocity[0], -car.angular_velocity[1], car.angular_velocity[2]
    elif ball_x == ball_y and car.position[0] > car.position[1]:
        my_car.pos = -car.position[0], -car.position[1], car.position[2]
        my_car.lin_vel = car.linear_velocity[1], car.linear_velocity[0], car.linear_velocity[2]
        my_car.rot = car.rotation[0] - np.pi / 2, car.rotation[1], car.rotation[2]
        my_car.ang_vel = -car.angular_velocity[0], -car.angular_velocity[1], car.angular_velocity[2]
    else:
        return None
    return my_car

class RandomEvenRecovery(StateSetter):
    def __init__(self, zero_boost_weight=0, zero_ball_vel_weight=0, ball_vel_mult=1, ball_zero_z=False):
        self.ball_zero_z = ball_zero_z
        self.ball_vel_mult = ball_vel_mult
        self.zero_boost_weight = zero_boost_weight
        self.zero_ball_vel_weight = zero_ball_vel_weight
        self.rng = np.random.default_rng()

    def reset(self, state_wrapper: StateWrapper):
        assert len(state_wrapper.cars) < 3
        zero_ball_vel = True
        if self.rng.uniform() > self.zero_ball_vel_weight:
            zero_ball_vel = False
        if self.rng.choice([False, True]):
            y = self.rng.uniform(-1500, 1500)
            x = 0
        else:
            y = 0
            x = self.rng.uniform(-1500, 1500)
        if y >= 0:
            ball_sign = 1
        else:
            ball_sign = -1
        state_wrapper.ball.set_pos(x, y, 94)
        if zero_ball_vel:
            state_wrapper.ball.set_lin_vel(0, 0, 0)
        else:
            state_wrapper.ball.set_lin_vel(self.ball_vel_mult * self.rng.uniform(-600, 600) if y == 0 and x != 0 else 0,
                                           self.ball_vel_mult * self.rng.uniform(-600, 600) if x == 0 and y != 0 else 0,
                                           0 if self.zero_ball_vel_weight else self.rng.uniform(-200, 200))
        state_wrapper.ball.set_ang_vel(0, 0, 0)
        if self.rng.uniform() > self.zero_boost_weight:
            boost = self.rng.uniform(0, 1.000001)
        else:
            boost = 0
        for car in state_wrapper.cars:
            if car.id == 1:
                car.set_pos(self.rng.uniform(-1000, 1000), y - 2500, self.rng.uniform(50, 350))
                car.set_rot(self.rng.uniform(-np.pi / 2, np.pi / 2),
                            self.rng.uniform(-np.pi, np.pi),
                            self.rng.uniform(-np.pi / 2, np.pi / 2))
                car.set_lin_vel(self.rng.uniform(-1500, 1500),
                                ball_sign * self.rng.uniform(-1500, 1500),
                                self.rng.uniform(-50, -1))
                car.set_ang_vel(self.rng.uniform(-4, 4), self.rng.uniform(-4, 4), self.rng.uniform(-4, 4))
            else:
                values = mirror(state_wrapper.cars[0], x, y)
                car.set_pos(*values.pos)
                car.set_rot(*values.rot)
                car.set_lin_vel(*values.lin_vel)
                car.set_ang_vel(*values.ang_vel)
            car.boost = boost