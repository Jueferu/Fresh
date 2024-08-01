import numpy as np

def build_rocketsim_env():
    import rlgym_sim
    from rlgym_sim.utils.state_setters import RandomState
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition

    from advanced_adapted_obs import AdvancedAdaptedObs
    from lookup_act import LookupAction
    from rewards.ChildReward import ChildReward

    spawn_opponents = True
    team_size = 2
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 30
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    reward_fn = ChildReward()
    action_parser = LookupAction()
    obs_builder = AdvancedAdaptedObs(pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
            ang_coef=1 / np.pi,
            lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
            ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)
    state_setter = RandomState()

    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=state_setter)

    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner

    # processes
    n_proc = 2

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    ts_per_iteration = 100_000

    learner = Learner(build_rocketsim_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      ppo_batch_size=ts_per_iteration,
                      ts_per_iteration=ts_per_iteration,
                      exp_buffer_size=ts_per_iteration*2,
                      ppo_minibatch_size=12_500,
                      ppo_ent_coef=0.01,
                      ppo_epochs=3,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=100_000,
                      policy_layer_sizes=[2048, 2048, 1024],
                      critic_layer_sizes=[2048, 2048, 1024],
                      timestep_limit=10e15,
                      policy_lr=2e-4,
                      critic_lr=2e-4)
    learner.learn()