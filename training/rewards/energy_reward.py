# https://github.com/Kaiyotech/Opti/blob/main/rewards.py#L723
import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import MASS, GRAVITY_Z, CEILING_Z, CAR_MAX_SPEED

GRAVITY = -GRAVITY_Z
MASS = 180

def clamp(n, smallest, largest): return max(smallest, min(n, largest))

class EnergyReward(RewardFunction):
    def __init__(self):
        super().__init__()
    
    def reset(self, initial_state: GameState):
        pass

    def get_reward(playerself, player: PlayerData, state: GameState, previous_action: np.ndarray):
        energy_reward = 0
        
        # max_energy is supersonic at ceiling, use to norm, ignore jump/dodge and boost
        max_energy = (MASS * GRAVITY * (CEILING_Z - 17)) + (0.5 * MASS * (CAR_MAX_SPEED * CAR_MAX_SPEED))
        energy_reward = 0
        # add height PE
        energy_reward += 1.1 * MASS * GRAVITY * player_position[2]
        # add KE
        velocity = np.linalg.norm(player_velocity)
        energy_reward += 0.5 * MASS * (velocity * velocity)
        # add boost
        energy_reward += 7.97e5 * player.boost_amount * 100
        if player.has_jump:
            energy_reward += 0.8 * 0.5 * MASS * (292 * 292)
        if player.has_flip:
            dodge_impulse = 500 + (velocity / 17) if velocity <= 1700 else (600 - (velocity - 1700))
            # cheat a bit to encourage the dodge usage
            dodge_impulse = max(dodge_impulse - 25, 0)
            energy_reward += 0.9 * 0.5 * MASS * (dodge_impulse * dodge_impulse)

        # this is some demo logic that I haven't figured out lol
        norm_energy = energy_reward / max_energy
        if player.is_demoed:
            norm_energy = 0
            
        enegy_reward = clamp(norm_energy, -1, 1)
        if np.isnan(energy_reward):
            energy_reward = 0

        return enegy_reward