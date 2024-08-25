import numpy as np
import random
import os
import time

# https://github.com/redd-rl/apollo-bot/blob/main/main.py
def get_most_recent_checkpoint() -> str:
    checkpoint_load_dir = "data/checkpoints/"
    checkpoint_load_dir += str(
        max(os.listdir(checkpoint_load_dir), key=lambda d: int(d.split("-")[-1])))
    checkpoint_load_dir += "/"
    checkpoint_load_dir += str(
        max(os.listdir(checkpoint_load_dir), key=lambda d: int(d)))
    return checkpoint_load_dir

def build_rocketsim_env():
    import rlgym_sim
    
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.reward_functions import CombinedReward

    from advanced_adapted_obs import AdvancedAdaptedObs
    from lookup_act import LookupAction

    from game_condition import GameCondition
    from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition, NoTouchTimeoutCondition

    from state_setters.team_size_setter import TeamSizeSetter
    from state_setters.weighted_sample_setter import WeightedSampleSetter
    from state_setters.wall_state import WallPracticeState
    from state_setters.symmetric_setter import KickoffLikeSetter
    from state_setters.goalie_state import GoaliePracticeState
    from state_setters.dribbling_state import DribblingStateSetter
    from state_setters.jump_shot_state import JumpShotState
    from state_setters.save_state import SaveState
    from state_setters.save_shot_state import SaveShot
    from state_setters.side_high_roll_state import SideHighRoll
    from state_setters.shot_state import ShotState
    from rlgym_sim.utils.state_setters import RandomState, DefaultState

    default = WeightedSampleSetter.from_zipped(
        GoaliePracticeState(),
        RandomState(True, True, False),
        DefaultState()
    )
    state_setter = TeamSizeSetter(1, default)
    
    from rewards.zero_sum_reward import ZeroSumReward
    from rewards.distribute_rewards import DistributeRewards
    from rewards.velocity_ball_to_goal_reward import VelocityBallToGoalReward
    from rewards.velocity_player_to_ball_reward import VelocityPlayerToBallReward
    from rewards.player_is_closest_ball_reward import PlayerIsClosestBallReward
    from rewards.player_face_ball_reward import PlayerFaceBallReward
    from rewards.player_behind_ball_reward import PlayerBehindBallReward
    from rewards.touch_ball_hitforce_reward import TouchBallRewardScaledByHitForce
    from rewards.speedflip_kickoff_reward import SpeedflipKickoffReward
    from rewards.allign_ball_reward import AlignBallGoal
    from rewards.air_reward import AirReward
    from rewards.possesion_reward import PossesionReward
    from rewards.player_velocity_reward import PlayerVelocityReward
    from rewards.goal_speed_and_placement_reward import GoalSpeedAndPlacementReward
    from rewards.kickoff_proximity_reward import KickoffProximityReward
    from rewards.save_boost_reward import SaveBoostReward
    from rewards.boost_pickup_reward import BoostPickupReward
    from rewards.aerial_distance_reward import AerialDistanceReward
    from rewards.dribble_reward import DribbleReward

    from rlgym_sim.utils.reward_functions.common_rewards import EventReward, LiuDistanceBallToGoalReward

    goal_reward = 1
    aggression_bias = .5
    concede_reward = -goal_reward * (1 - aggression_bias)

    rewards = CombinedReward.from_zipped(
        # begginer
        (ZeroSumReward(TouchBallRewardScaledByHitForce(), .5, 1), 2.5),
        (ZeroSumReward(VelocityPlayerToBallReward(), .5, 1), 2.5),
        (PlayerFaceBallReward(), .1),
        (AirReward(), 0.05),
        # intermediate
        (VelocityBallToGoalReward(), 20),
        (EventReward(goal=goal_reward, concede=concede_reward), 30),
        (ZeroSumReward(PlayerIsClosestBallReward(), 1, 1), 5),
        # advanced
        #(PlayerVelocityReward(), .2),
        (ZeroSumReward(BoostPickupReward(), .5, 1), 5),
        (ZeroSumReward(PlayerBehindBallReward(), .5, 1), 5),
    )

    spawn_opponents = True
    team_size = 3
    tick_skip = 8

    no_touch_seconds = 10
    no_touch_ticks = int(round(no_touch_seconds * 120 / tick_skip))

    terminal_conditions = [GoalScoredCondition(), NoTouchTimeoutCondition(no_touch_ticks)]

    reward_fn = rewards
    action_parser = LookupAction()
    obs_builder = AdvancedAdaptedObs(
            pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
            ang_coef=1 / np.pi,
            lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
            ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL, 
            player_padding=3, 
            expanding=False)

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

    n_proc = 40
    min_inference_size = max(1, int(round(n_proc * 0.9)))
    ts_per_iteration = 100_000

    try:
        checkpoint_load_dir = get_most_recent_checkpoint()
        print(f"Loading checkpoint: {checkpoint_load_dir}")
    except:
        print("checkpoint load dir not found.")
        checkpoint_load_dir = None

    learner = Learner(build_rocketsim_env,
                      n_proc=n_proc,
                      checkpoint_load_folder=checkpoint_load_dir,
                      min_inference_size=min_inference_size,
                      ppo_batch_size=ts_per_iteration,
                      ts_per_iteration=ts_per_iteration,
                      exp_buffer_size=ts_per_iteration*3,
                      ppo_minibatch_size=50_000,
                      ppo_ent_coef=0.01,
                      ppo_epochs=2,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=1_000_000,
                      policy_layer_sizes=[2048, 2048, 1024, 1024],
                      critic_layer_sizes=[2048, 2048, 1024, 1024],
                      timestep_limit=10e15,
                      policy_lr=1e-4,
                      critic_lr=1e-4,
                      render=True)
    start_time = time.time()

    learner.learn()

    end_time = time.time()
    trained_time = end_time - start_time
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(trained_time))
    
    print(f"Trained for {formatted_time}!")