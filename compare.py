import rlgym_sim
from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition
from rlgym_sim.utils.state_setters import DefaultState
import rlgym_sim.utils.common_values as common_values

import numpy as np
import torch

# read the agents' BOOK_KEEPING_VARS.json files to get the agent's "cumulative_model_updates" value
import json
with open("rlbot1/BOOK_KEEPING_VARS.json", "r") as f:
    bot1_cumulative_model_updates = json.load(f)["cumulative_model_updates"]
with open("rlbot2/BOOK_KEEPING_VARS.json", "r") as f:
    bot2_cumulative_model_updates = json.load(f)["cumulative_model_updates"]
    
bot1_name = f"Fresh V2.{bot1_cumulative_model_updates}"
bot2_name = f"Fresh V2.{bot2_cumulative_model_updates}"

from rlbot1.discrete_policy import DiscreteFF

from training.lookup_act import LookupAction
from training.advanced_adapted_obs import AdvancedAdaptedObs

action_parser = LookupAction()
device = torch.device("cpu")
num_actions = len(action_parser._lookup_table)

OBS_SIZE = 293
POLICY_LAYER_SIZES = [2048, 2048, 1024, 1024]

bot1_policy = DiscreteFF(OBS_SIZE, num_actions, POLICY_LAYER_SIZES, device)
bot2_policy = DiscreteFF(OBS_SIZE, num_actions, POLICY_LAYER_SIZES, device)

bot1_policy.load_state_dict(torch.load("rlbot1\PPO_POLICY.pt", map_location=device))
bot2_policy.load_state_dict(torch.load("rlbot2\PPO_POLICY.pt", map_location=device))

torch.set_num_threads(2)

env = rlgym_sim.make(
    tick_skip=8,
    spawn_opponents=True,
    action_parser=action_parser,
    obs_builder=AdvancedAdaptedObs(
        pos_coef=np.asarray(
            [
                1 / common_values.SIDE_WALL_X,
                1 / common_values.BACK_NET_Y,
                1 / common_values.CEILING_Z,
            ]
        ),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        player_padding=4,
        expanding=False,
    ),
    terminal_conditions=[GoalScoredCondition()],
    state_setter=DefaultState(),
)
import training.rocketsimvis_rlgym_sim_client as rsv
type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)

obs = env.reset()
done = False

bot1_score = 0
bot2_score = 0
max_score = 500

try:
    while (bot1_score + bot2_score) < max_score:
        obs = env.reset()
        done = False
        last_info = {'result': 0}
        
        while not done:
            action1, action2 = None, None
            with torch.no_grad():
                action1, _ = bot1_policy.get_action(obs[0], True)
                action2, _ = bot2_policy.get_action(obs[1], True)

            new_obs, _, term, info = env.step([action1, action2])
            env.render()
            obs = new_obs
            done = term
            last_info = info
        
        result = last_info['result']
        
        if result == 1:
            bot1_score += 1
        elif result == -1:
            bot2_score += 1
        print(f"{bot1_name}: {bot1_score} - {bot2_name}: {bot2_score}")
except KeyboardInterrupt:
    pass