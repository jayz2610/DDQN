from src.base.GridActions import GridActions


class GridRewardParams:
    def __init__(self):
        self.boundary_penalty = 1.0
        self.empty_battery_penalty = 150.0
        self.movement_penalty = 0.2
        self.data_multiplier = 1.0  # Tune this
        self.landing_reward = 1  # Tune this
        self.failed_landing_penalty = 2.0  # Tune this (positive value for penalty)
        self.repeated_boundary_penalty_multiplier = 0.5  # 可调：重复碰壁的基础惩罚乘数
        self.max_consecutive_hits_for_penalty_scaling = 5  # 可调：惩罚增长的最大计次，超过此次数惩罚可能不再线性增加或使用最大值


class GridRewards:
    def __init__(self, stats, params: GridRewardParams = None):
        self.params = params if params is not None else GridRewardParams()
        self.cumulative_reward: float = 0.0

        if stats and hasattr(stats, 'add_log_data_callback'):
            stats.add_log_data_callback('cumulative_reward', self.get_cumulative_reward)

    def get_cumulative_reward(self):
        return self.cumulative_reward

    def calculate_motion_rewards(self, state, action: GridActions, next_state):
        reward = 0.0
        if not next_state.landed:
            # Penalize battery Consumption
            reward -= self.params.movement_penalty

        # Penalize not moving (This happens when it either tries to land or fly into a boundary or hovers or fly into
        # a cell occupied by another agent)
        if tuple(state.position) == tuple(next_state.position) and \
                not next_state.landed and action not in [GridActions.HOVER, GridActions.LAND]:
            reward -= self.params.boundary_penalty

        # Penalize battery dead
        if next_state.movement_budget == 0 and not next_state.landed:
            reward -= self.params.empty_battery_penalty

        return reward

    def reset(self):
        self.cumulative_reward = 0.0
