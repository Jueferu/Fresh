from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters import StateWrapper, RandomState 
import random

class TeamSizeSetter(StateSetter):
    def __init__(self, count: 1, setter: StateSetter):
        super().__init__()
        self.default = setter or RandomState(True, True, False)
        self.count = count

    def build_wrapper(self, max_team_size: int, spawn_opponents: bool) -> StateWrapper:
        wrapper = StateWrapper(blue_count=self.count, orange_count=self.count if spawn_opponents else 0)

        self.count += 1
        if self.count > max_team_size:
            self.count = 1

        return wrapper

    def reset(self, state_wrapper: StateWrapper):
        self.default.reset(state_wrapper)