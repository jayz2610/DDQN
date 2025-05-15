import numpy as np

from src.State import State
from src.base.GridActions import GridActions
from src.base.GridRewards import GridRewards, GridRewardParams
# from src.Map.Map import Map


class RewardParams(GridRewardParams):
    def __init__(self):
        super().__init__()


class Rewards(GridRewards):

    def __init__(self, reward_params: RewardParams, stats):
        # Pass specific params to base class init (requires base init to accept params)
        super().__init__(stats, params=reward_params)
        self.reset()

    def calculate_reward(self, state: State, action: GridActions, next_state: State, agent_id: int) -> float:
        """
        Calculates the reward for a given agent transition (state, action -> next_state).
        Includes motion penalties, data reward, landing reward, and failed landing penalty.
        """
        step_reward = 0.0 # Default return value
        original_state_active = state.active_agent
        original_next_state_active = next_state.active_agent

        try:
            # Set active agent context for state properties/methods
            if state is None or next_state is None: return 0.0
            state.active_agent = agent_id
            next_state.active_agent = agent_id

            # --- a. Calculate base motion rewards/penalties ---
            motion_reward = super().calculate_motion_rewards(state, action, next_state)
            step_reward += motion_reward

            # --- b. Calculate Data Collection Reward ---
            if hasattr(state, 'collected') and state.collected is not None and \
                    hasattr(next_state, 'collected') and next_state.collected is not None:
                # Ensure collected arrays are valid before sum
                current_collected_sum = np.sum(state.collected) if state.collected is not None else 0
                next_collected_sum = np.sum(next_state.collected) if next_state.collected is not None else 0
                data_gain = next_collected_sum - current_collected_sum

                if data_gain > 1e-9:  # Using a small epsilon to avoid floating point issues
                    data_reward = self.params.data_multiplier * data_gain
                    step_reward += data_reward
                    print(f"[Reward Debug] Agent {agent_id}: Data gain {data_gain:.2f}, Data Reward: {data_reward:.2f}")

            # --- c. NEW: Calculate Landing Rewards/Penalties ---
            if action == GridActions.LAND:
                print(f"[Reward Debug] Agent {agent_id} Took LAND action. Next state landed: {next_state.landed}") # Debug print
                if next_state.landed:  # Successfully landed
                    step_reward += self.params.landing_reward
                    # print(f"[Reward Debug] Agent {agent_id}: Successful LAND! Reward: +{self.params.landing_reward}")
                else:  # Attempted to land but failed (e.g., not in a landing zone)
                    step_reward -= self.params.failed_landing_penalty
                    # print(f"[Reward Debug] Agent {agent_id}: Failed LAND! Penalty: -{self.params.failed_landing_penalty}")

            # --- d. Repeated Boundary Hit Penalty ---
            if hasattr(self.params, 'repeated_boundary_penalty_multiplier') and \
                    hasattr(self.params, 'max_consecutive_hits_for_penalty_scaling') and \
                    hasattr(next_state, 'consecutive_invalid_moves') and \
                    0 <= agent_id < len(next_state.consecutive_invalid_moves):

                num_consecutive_hits = next_state.consecutive_invalid_moves[agent_id]
                if num_consecutive_hits > 1:  # Penalize starting from the 2nd consecutive hit
                    # Linear penalty:
                    # 惩罚因子可以是你定义的 self.params.max_consecutive_hits_for_penalty_scaling
                    # 这里的 penalty_factor 是额外惩罚的“倍数”
                    penalty_factor = min(num_consecutive_hits - 1,
                                         getattr(self.params, 'max_consecutive_hits_for_penalty_scaling', 5))
                    penalty = self.params.repeated_boundary_penalty_multiplier * penalty_factor
                    step_reward -= penalty
                    # print(f"[Reward Debug] Agent {agent_id}: Repeated wall hit. Hits: {num_consecutive_hits}, Penalty Factor: {penalty_factor}, Additional Penalty: -{penalty:.2f}")
                    if hasattr(self, 'cumulative_reward'):  # 确保属性存在
                        self.cumulative_reward += step_reward
                    else:  # 如果基类没有这个属性，但我们期望有，则需要初始化
                        # 这个情况理论上不应该发生，因为 GridRewards 应该有 cumulative_reward
                        print(
                            "[Warning] cumulative_reward attribute not found in Rewards object during calculate_reward.")


        except Exception as e:
            print(f"!!!!!!!! ERROR inside calculate_reward for Agent {agent_id} !!!!!!!!")
            print(f"Error message: {e}")
            print(f"Action: {action.name if isinstance(action, GridActions) else action}") # Print action name safely
            # Print states safely
            try:
                 print(f"State Pos: {state.position if state else 'None'}, Budget: {state.movement_budget if state else 'None'}, Landed: {state.landed if state else 'None'}, Terminal: {state.terminal if state else 'None'}")
                 print(f"Next State Pos: {next_state.position if next_state else 'None'}, Budget: {next_state.movement_budget if next_state else 'None'}, Landed: {next_state.landed if next_state else 'None'}, Terminal: {next_state.terminal if next_state else 'None'}")
            except Exception as e_print:
                 print(f"Error printing state details: {e_print}")
            import traceback
            traceback.print_exc()
            step_reward = 0.0

        finally:
            # Restore original active agents
            if state is not None: state.active_agent = original_state_active
            if next_state is not None: next_state.active_agent = original_next_state_active

            # Return the final step reward
            if step_reward is None: return 0.0
            return float(step_reward)

    def reset(self):
        super().reset()
