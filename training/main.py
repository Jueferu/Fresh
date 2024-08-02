import numpy as np
import random

def build_rocketsim_env():
    import rlgym_sim
    
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils.reward_functions import CombinedReward

    from advanced_adapted_obs import AdvancedAdaptedObs
    from lookup_act import LookupAction

    from state_setters.team_size_setter import TeamSizeSetter
    from state_setters.weighted_sample_setter import WeightedSampleSetter
    from state_setters.wall_state import WallPracticeState
    from state_setters.symmetric_setter import KickoffLikeSetter
    from state_setters.goalie_state import GoaliePracticeState
    from rlgym_sim.utils.state_setters import RandomState, DefaultState

    state_setter = TeamSizeSetter(1, WeightedSampleSetter(
        WallPracticeState(),
        KickoffLikeSetter(),
        GoaliePracticeState(),
        RandomState(),
        DefaultState()
    ))
    
    from rewards.zero_sum_reward import ZeroSumReward
    from rewards.velocity_ball_to_goal_reward import VelocityBallToGoalReward
    from rewards.velocity_player_to_ball_reward import VelocityPlayerToBallReward
    from rewards.possesion_reward import PossesionReward
    from rewards.player_face_ball_reward import PlayerFaceBallReward
    from rewards.player_behind_ball_reward import PlayerBehindBallReward

    player_face_ball_reward = ZeroSumReward(PlayerFaceBallReward(), team_spirit=.5)
    player_behind_ball_reward = ZeroSumReward(PlayerBehindBallReward(), team_spirit=.5)
    player_to_ball_reward = ZeroSumReward(VelocityPlayerToBallReward(), team_spirit=.5)
    ball_to_goal_reward = ZeroSumReward(VelocityBallToGoalReward(), team_spirit=.5)

    spawn_opponents = True
    team_size = random.randint(1, 2)
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 30
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    reward_fn = CombinedReward.from_zipped(
        (player_face_ball_reward, .5),
        (player_behind_ball_reward, 2),
        (player_to_ball_reward, 1),
        (ball_to_goal_reward, 2)
        (PossesionReward(), .5)
    )
    action_parser = LookupAction()
    obs_builder = AdvancedAdaptedObs(pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
            ang_coef=1 / np.pi,
            lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
            ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL, player_padding=2, expanding=False)

    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=state_setter)

    import rocketsimvis_rlgym_sim_client as rsv
    type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)

    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner

    n_proc = 32
    min_inference_size = max(1, int(round(n_proc * 0.9)))
    ts_per_iteration = 200_000

    learner = Learner(build_rocketsim_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      ppo_batch_size=ts_per_iteration,
                      ts_per_iteration=ts_per_iteration,
                      exp_buffer_size=ts_per_iteration*2,
                      ppo_minibatch_size= 12_500,
                      ppo_ent_coef=0.01,
                      ppo_epochs=2,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=500_000,
                      policy_layer_sizes=[2048, 2048, 1024],
                      critic_layer_sizes=[2048, 2048, 1024],
                      timestep_limit=10e15,
                      policy_lr=1e-4,
                      critic_lr=1e-4,
                      render=False)
    learner.learn()