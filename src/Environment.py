import copy
import distutils.util
import random
import tqdm

from src.DDQN.Agent import DDQNAgentParams, DDQNAgent
from src.DDQN.Trainer import DDQNTrainerParams, DDQNTrainer
from src.Display import DHDisplay
from src.Grid import GridParams, Grid
from src.Physics import PhysicsParams, Physics
from src.Rewards import RewardParams, Rewards
from src.State import State
from src.base.Environment import BaseEnvironment, BaseEnvironmentParams
from src.base.GridActions import GridActions


class EnvironmentParams(BaseEnvironmentParams):
    def __init__(self):
        super().__init__()
        self.grid_params = GridParams()
        self.reward_params = RewardParams()
        self.trainer_params = DDQNTrainerParams()
        self.agent_params = DDQNAgentParams()
        self.physics_params = PhysicsParams()


class Environment(BaseEnvironment):

    def __init__(self, params: EnvironmentParams):
        self.display = DHDisplay()
        super().__init__(params, self.display)
        self.params = params
        self.grid = Grid(params.grid_params, stats=self.stats)
        print("[Environment Init] Attempting to initialize Rewards...")
        try:
            # The potentially failing line:
            self.rewards = Rewards(params.reward_params, stats=self.stats)
            # If successful, print the type
            print(f"[Environment Init] Rewards initialized: Type={type(self.rewards)}")
            if self.rewards is None:
                print("[Environment Init] WARNING: Rewards() constructor returned None!")
        except Exception as e:
            # Catch ANY exception during Rewards.__init__
            print(f"[Environment Init] ***** ERROR initializing Rewards *****: {e}")
            import traceback
            traceback.print_exc()  # Print the full traceback for the internal error
            self.rewards = None  # Explicitly set to None if initialization failed
        # Get dims correctly from your grid/map object
        height, width = self.grid.shape  # Get first two dimensions
        print(f"[Environment Init] Detected Grid Dimensions: Width={width}, Height={height}")  # Optional: Debug print
        # Pass dims to Physics constructor
        self.physics = Physics(params=params.physics_params, stats=self.stats, grid_width=width, grid_height=height)
        self.agent = DDQNAgent(params.agent_params, self.grid.get_example_state(), self.physics.get_example_action(),
                               stats=self.stats)
        self.trainer = DDQNTrainer(params.trainer_params, agent=self.agent)
        self.display.set_channel(self.physics.channel)
        self.first_action = True
        self.last_actions = []
        self.last_rewards = []
        self.last_states = []

    def _select_landing_target(self):
        """Selects a landing target coordinate."""
        return self.grid.map_image.get_random_landing_position()

    def _select_nearest_landing_target(self, current_pos_tuple: tuple):
        """
        Selects the NEAREST reachable landing zone coordinate using A* distance.

        Args:
            current_pos_tuple: The agent's current position as a tuple (x, y).

        Returns:
            The coordinate tuple (x, y) of the nearest landing zone point,
            or None if no landing points are reachable.
        """
        landing_points = self.grid.map_image.get_starting_vector()  # Get all landing points [(x,y), ...]

        if not landing_points:
            print("[Target Logic Error] No landing points defined in map!")
            return None  # No landing points exist

        best_target = None
        min_dist = float('inf')

        print(
            f"[Target Logic] Finding nearest landing point from {current_pos_tuple} among {len(landing_points)} options.")
        # Iterate through all possible landing points
        for point in landing_points:
            goal_tuple = tuple(point)  # Ensure it's a tuple

            # Skip if already at this landing point
            if goal_tuple == current_pos_tuple:
                # If already at a landing point, that's the closest one
                return current_pos_tuple

            # Calculate A* path distance
            path = self.grid.map_image.a_star(current_pos_tuple, goal_tuple)
            dist = len(path) - 1 if path else float('inf')  # Path length includes start, so distance is len-1

            # Update if this landing point is closer
            if dist < min_dist:
                min_dist = dist
                best_target = goal_tuple

        if best_target:
            print(f"[Target Logic] Nearest landing point found: {best_target} (A* Dist: {min_dist})")
        else:
            print(f"[Target Logic Warning] No reachable landing point found from {current_pos_tuple}!")

        return best_target  # Return the coordinates of the closest reachable landing point

    # --- END NEW METHOD ---

    def _select_next_device_target(self, current_state: State, agent_id: int):
        """Selects the next best IoT device target coordinate, EXCLUDING current position."""
        current_pos_tuple = tuple(current_state.positions[agent_id])
        if current_state.device_list is None: return None

        # Find devices that still need data collected AND ARE NOT at the agent's current location
        available_devices = [
            dev for dev in current_state.device_list.devices
            if not dev.depleted and tuple(dev.position) != current_pos_tuple # Exclude current location
        ]

        if not available_devices:
            print(f"[Target Logic] Agent {agent_id}: No *other* available devices remaining.")
            return None

        # Strategy: Select nearest available device (excluding current location)
        nearest_device = None
        min_dist = float('inf')
        for device in available_devices:
            goal_tuple = tuple(device.position)
            # Use A* distance
            path = self.grid.map_image.a_star(current_pos_tuple, goal_tuple)
            dist = len(path) - 1 if path else float('inf')

            if dist < min_dist:
                min_dist = dist
                nearest_device = device

        if nearest_device:
            print(f"[Target Logic] Agent {agent_id}: Selected nearest OTHER device {nearest_device.position} (Dist: {min_dist}) as target.")
            return tuple(nearest_device.position)
        else:
            print(f"[Target Logic] Agent {agent_id}: Could not find nearest OTHER device target.")
            return None
        # --- End Strategy ---
        # Other strategies possible: device with most data, data/distance ratio, etc

    # 在 Environment class 中

    def _update_agent_target(self, state: State, agent_id: int):
        """
        Checks conditions and updates the target for the agent if needed.
        Updates consecutive_zero_rate_hovers counter but does NOT force target change based on it.
        """
        original_active = state.active_agent
        state.active_agent = agent_id

        current_target = state.target
        current_pos = tuple(state.position)

        # --- Return-to-base conditions ---
        BUDGET_THRESHOLD = getattr(self.params.reward_params, 'budget_threshold_for_return', 25)
        DATA_RATIO_THRESHOLD = getattr(self.params.reward_params, 'data_ratio_threshold_for_return', 0.98)

        budget_low = state.movement_budget < BUDGET_THRESHOLD

        # --- 原有的 data_nearly_collected 判断 ---
        data_nearly_collected_by_ratio = False
        if state.initial_total_data > 0:
            data_nearly_collected_by_ratio = state.get_collection_ratio() > DATA_RATIO_THRESHOLD
        # 如果没有初始数据，一种处理方式是可以认为数据收集任务"已完成" (因为无数据可收)
        # 或者保持为False，取决于您希望的行为。如果希望无数据时也应该尽快返回，则设为True。
        # 这里我假设若无初始数据，则数据收集方面已满足返回条件。
        elif state.initial_total_data == 0 and state.device_list and state.device_list.num_devices > 0:
            # 有设备但总数据为0，也算收集完了（虽然有点奇怪）
            data_nearly_collected_by_ratio = True
        elif state.device_list is None or state.device_list.num_devices == 0:
            # 没有设备，自然算收集完了
            data_nearly_collected_by_ratio = True

        # --- 新增：检查是否所有设备都已耗尽 ---
        all_devices_individually_depleted = False  # 默认不是所有都耗尽
        if state.device_list and state.device_list.devices:  # 确保设备列表存在且不为空
            all_devices_individually_depleted = all(dev.depleted for dev in state.device_list.devices)
        elif state.device_list is None or state.device_list.num_devices == 0:  # 如果没有设备
            all_devices_individually_depleted = True  # 也可以认为“所有设备（0个）都耗尽了”

        # --- 合并条件 ---
        # 现在，如果收集比例达标，或者所有设备都单独耗尽了，都算数据收集任务接近完成
        data_nearly_collected = data_nearly_collected_by_ratio or all_devices_individually_depleted

        should_return_to_base = budget_low or data_nearly_collected

        # --- Target invalid/depleted conditions ---
        target_is_none = (current_target is None)
        target_device_depleted = False
        target_device_data_str = "N/A"

        # --- Logic to update hover counter ---
        is_at_device_target = False  # Flag if currently at a non-depleted device target
        if not target_is_none and state.device_list:
            landing_target_coord = self._select_landing_target()
            if current_target != landing_target_coord:  # Target is likely a device
                target_device_object = None
                for dev in state.device_list.devices:
                    if tuple(dev.position) == current_target:
                        target_device_object = dev
                        target_device_data_str = f"{dev.collected_data:.1f}/{dev.data:.1f}"
                        if dev.depleted:
                            target_device_depleted = True
                            #print(f"[Target Logic] Agent {agent_id}: Target device {current_target} depleted.") # Keep print if needed
                        break

                # Update unproductive hover counter IF at the target device AND it's NOT depleted yet
                if current_pos == current_target and target_device_object is not None and not target_device_depleted:
                    is_at_device_target = True  # Mark that we are hovering at a valid device
                    current_rate_to_target = 0
                    if hasattr(target_device_object, 'get_data_rate'):
                        current_rate_to_target = target_device_object.get_data_rate(current_pos, self.physics.channel)

                    if current_rate_to_target <= 1e-6:  # Rate is negligible
                        state.consecutive_zero_rate_hovers[agent_id] += 1
                       #print(f"[Target Logic] Agent {agent_id}: Unproductive hover at {current_target}, count: {state.consecutive_zero_rate_hovers[agent_id]}")
                    else:  # Productive hover
                        state.consecutive_zero_rate_hovers[agent_id] = 0

        # Reset hover counter if not currently hovering at a device target
        if not is_at_device_target:
            state.consecutive_zero_rate_hovers[agent_id] = 0

        # --- Determine if a new target is needed (only if None or depleted) ---
        needs_new_target = target_is_none or target_device_depleted
        # NOTE: We REMOVED unproductive_hovering from this condition

        # --- Optional: Print status ---
        #print(f"[Target Status] Agent {agent_id} Pos:{current_pos} Target:{current_target} (Data: {target_device_data_str}) Budget:{state.movement_budget} NeedsNew:{needs_new_target} ShouldReturn:{should_return_to_base} UnprodHovers:{state.consecutive_zero_rate_hovers[agent_id]}")

        # --- Decide the Final Target Based on Priority ---
        final_target = current_target
        target_updated = False  # Flag if target changed

        if should_return_to_base:
            potential_new_target = self._select_nearest_landing_target(current_pos)
            if potential_new_target is None: potential_new_target = current_target  # Fallback
            if current_target != potential_new_target:
               #print(f"[Target Logic] Agent {agent_id}: Return condition met. Targeting NEAREST landing zone {potential_new_target}.")
                final_target = potential_new_target
                target_updated = True

        elif needs_new_target:
            #print(f"[Target Logic] Agent {agent_id}: Needs new target (IsNone:{target_is_none}, Depleted:{target_device_depleted}). Selecting next device.")
            potential_new_target = self._select_next_device_target(state, agent_id)
            if potential_new_target is None:  # No more devices
                #print(f"[Target Logic] Agent {agent_id}: No more devices available, targeting landing zone.")
                potential_new_target = self._select_nearest_landing_target(current_pos)
                if potential_new_target is None: potential_new_target = current_target  # Fallback

            if current_target != potential_new_target:
                final_target = potential_new_target
                target_updated = True

        # --- Update State Only If Target Has Actually Changed ---
        if target_updated:
            print(f"[Target Logic] Agent {agent_id}: Updating target from {current_target} to {final_target}.")
            state.set_target(final_target)
            state.consecutive_zero_rate_hovers[agent_id] = 0  # Reset counter when target changes

        # --- Restore Original Context ---
        state.active_agent = original_active

    def test_episode(self):
        """
        Runs a single episode using the learned policy (exploitation only)
        and logs the results to ModelStats for testing evaluation.
        """
        print("[Test Episode] Starting...")
        # 1. Initialize state and stats for the episode
        state = copy.deepcopy(self.init_episode())  # Get initial state for this test episode
        self.stats.on_episode_begin(self.episode_count)  # Notify stats logger

        # 2. Initialize tracking lists specific to this test episode
        num_agents = state.num_agents
        # Use separate lists for tracking last states/actions during testing
        # to correctly calculate rewards and log to stats
        test_last_states = [None] * num_agents
        test_last_actions = [None] * num_agents

        # This variable holds the state passed between time steps
        current_state_in_test = state

        # 3. Loop through steps until the episode is terminated
        while not current_state_in_test.all_terminal:  # Check overall episode termination
            # This variable holds the state passed between agent turns within a step
            state_for_next_agent = current_state_in_test

            # 4. Loop through all agents
            for agent_id in range(num_agents):
                # The state before this agent acts is the one from the previous agent's turn
                state_for_this_agent = state_for_next_agent
                # Set active agent for state properties/methods if they rely on it
                state_for_this_agent.active_agent = agent_id

                # Skip if this agent is already in a terminal state
                if state_for_this_agent.terminals[agent_id]:
                    continue

                # --- Get Action (Using exploitation policy for testing) ---
                action = self.agent.get_exploitation_action_target(state_for_this_agent)
                action_enum = GridActions(action)  # Get enum for printing, optional

                # --- Log experience from the PREVIOUS step to stats ---
                last_state = test_last_states[agent_id]
                last_action = test_last_actions[agent_id]
                # Only log if it's not the very first action for this agent
                if last_state is not None:
                    # Calculate reward for the transition: last_state -> state_for_this_agent
                    # Ensure calculate_reward uses the correct active_agent context
                    original_active = state_for_this_agent.active_agent
                    state_for_this_agent.active_agent = agent_id  # Ensure context
                    reward = self.rewards.calculate_reward(
                        last_state,
                        GridActions(last_action),  # Previous action
                        state_for_this_agent,  # State BEFORE current action
                        agent_id  # Pass agent ID
                    )
                    state_for_this_agent.active_agent = original_active  # Restore

                    # Add experience (s, a, r, s') to stats trajectory
                    # s' is the state *before* the current physics step executes
                    self.stats.add_experience(
                        (last_state, last_action, reward,
                         copy.deepcopy(state_for_this_agent))  # Log state before physics step
                    )

                # --- Store state/action for the NEXT logging iteration ---
                test_last_states[agent_id] = copy.deepcopy(state_for_this_agent)
                test_last_actions[agent_id] = action  # Store the integer action value

                # --- Execute Physics Step (In-Place Modification) ---
                # print(f"[DEBUG][TestEp] Calling physics.step for Agent {agent_id} with Action {action_enum.name}")
                self.physics.step(action_enum, state_for_this_agent)  # Pass state object to be modified
                # print(f"[DEBUG][TestEp] physics.step finished for Agent {agent_id}")

                # The modified state becomes the input state for the next agent's turn
                state_for_next_agent = state_for_this_agent

                # --- Log FINAL experience if agent became terminal THIS step ---
                # Check terminal status AFTER the physics step
                if state_for_next_agent.terminals[agent_id] and test_last_states[agent_id] is not None:
                    # Calculate reward for the action JUST taken that resulted in termination
                    original_active = state_for_next_agent.active_agent
                    state_for_next_agent.active_agent = agent_id  # Ensure context
                    final_reward = self.rewards.calculate_reward(
                        test_last_states[agent_id],  # State before action
                        GridActions(test_last_actions[agent_id]),  # Action just taken
                        state_for_next_agent,  # Resulting terminal state
                        agent_id  # Pass agent ID
                    )
                    state_for_next_agent.active_agent = original_active  # Restore

                    # Add final experience (s, a, r, s') to stats
                    self.stats.add_experience(
                        (test_last_states[agent_id], test_last_actions[agent_id], final_reward,
                         copy.deepcopy(state_for_next_agent))  # Log terminal state
                    )
                # --- End of Agent Loop ---

            # Update the state for the next step (time t+1) in the while loop
            current_state_in_test = state_for_next_agent

        # --- End of Episode ---
        self.stats.on_episode_end(self.episode_count)
        # Log accumulated data from self.stats.trajectory to TensorBoard testing logs
        print("[Test Episode] Finished. Logging test data...")
        # Make sure self.step_count provides a meaningful step value for logging
        self.stats.log_testing_data(step=self.step_count)
        print("[Test Episode] Logging complete.")

    def step(self, state: State, random=False):
        if len(self.last_states) != state.num_agents: self.last_states = [None] * state.num_agents
        if len(self.last_actions) != state.num_agents: self.last_actions = [None] * state.num_agents
        next_state = state
        for agent_id in range(state.num_agents):
            current_agent_state = next_state
            current_agent_state.active_agent = agent_id

            if current_agent_state.terminals[agent_id]:
                continue

            # Update target and unproductive hover count BEFORE action selection
            self._update_agent_target(current_agent_state, agent_id)

            # --- Action Selection ---
            action = GridActions.HOVER.value  # Default needed?

            if random:  # During replay memory fill - Use A* Guidance + Stuck Hover Fix
                target = current_agent_state.target
                current_pos_tuple = tuple(current_agent_state.positions[agent_id])
                action_chosen = False

                # Get max stuck hover steps from params (defined in EnvironmentParams)
                # Handle potential missing parameter with a default
                max_stuck_hovers = self.params.trainer_params.max_stuck_hovers_prefill

                if target is not None:
                    if target == current_pos_tuple:  # Agent is AT the target
                        action_chosen = True
                        # Check if it's landing zone or device
                        h, w = self.grid.shape
                        tx, ty = target[0], target[1]
                        is_landing_target = False
                        if 0 <= ty < h and 0 <= tx < w:
                            is_landing_target = self.grid.map_image.start_land_zone[ty, tx]

                        if is_landing_target:
                            action = GridActions.LAND.value
                            #print(
                               # f"[Fill Memory Guided] Agent {agent_id}: Reached landing target {target}, taking action LAND")
                        else:  # At a device target
                            # Check unproductive hover count
                            if current_agent_state.consecutive_zero_rate_hovers[agent_id] < max_stuck_hovers:
                                action = GridActions.HOVER.value  # Hover if not stuck too long
                               # print(
                                  #  f"[Fill Memory Guided] Agent {agent_id}: Reached device target {target}, Hovering (Unprod Streak: {current_agent_state.consecutive_zero_rate_hovers[agent_id]}).")
                            else:
                                # Stuck hovering unproductively, force random N/S/E/W
                              #  print(
                                   # f"[Fill Memory Guided] Agent {agent_id}: Stuck unproductive hover at {target} (Streak: {current_agent_state.consecutive_zero_rate_hovers[agent_id]}). Taking random move.")
                                move_actions = [GridActions.NORTH.value, GridActions.SOUTH.value,
                                                GridActions.EAST.value, GridActions.WEST.value]
                                action = random.choice(move_actions)
                                # Note: counter reset happens in _update_agent_target if agent moves or target changes

                    else:  # Not at target, navigate towards it using A* path step
                        path = self.grid.map_image.a_star(current_pos_tuple, target)
                        if path and len(path) > 1:
                            next_node = path[1]
                            tdx = next_node[0] - current_pos_tuple[0]
                            tdy = next_node[1] - current_pos_tuple[1]
                            # Basic direction logic
                            if abs(tdx) > abs(tdy):
                                action = GridActions.EAST.value if tdx > 0 else GridActions.WEST.value
                            elif abs(tdy) >= abs(tdx):
                                action = GridActions.SOUTH.value if tdy > 0 else GridActions.NORTH.value
                            else:
                                action = GridActions.HOVER.value
                            action_chosen = True
                         #   print(f"[Fill Memory Guided by A* Path] Agent {agent_id}: Target {target}, next node {next_node}, taking action {GridActions(action).name}")
                        else:  # No path
                            action = GridActions.HOVER.value
                            action_chosen = True
                        #    print(f"[Fill Memory Guided by A* Path] Agent {agent_id}: No path to target {target} or already there, taking action HOVER")

                if not action_chosen:  # No target
                    action = GridActions.HOVER.value
                #    print(f"[Fill Memory Guided] Agent {agent_id}: No target, taking action HOVER")

                # Final print for guided action
              #  print(f"[Fill Memory Guided] Agent {agent_id}: Final Action Chosen: {GridActions(action).name}")

            else:  # During training/testing - Use learned policy
                # State passed to agent now includes consecutive_zero_rate_hovers
                action = self.agent.act(current_agent_state)
                print(
                    f"[Train Step] Agent {agent_id}: Using policy action {GridActions(action).name} (Target: {current_agent_state.target}, UnprodHovers: {current_agent_state.consecutive_zero_rate_hovers[agent_id]})")

            # --- Experience Collection (Before Action) ---
            # (Keep existing logic, ensure calculate_reward uses agent_id)
            last_state_for_agent = self.last_states[agent_id]
            last_action_for_agent = self.last_actions[agent_id]
            if not self.first_action and last_state_for_agent is not None:
                reward = self.rewards.calculate_reward(
                    last_state_for_agent, GridActions(last_action_for_agent),
                    current_agent_state, agent_id
                )
                self.trainer.add_experience(
                    last_state_for_agent, last_action_for_agent,
                    reward, current_agent_state
                )
                self.stats.add_experience(
                    (last_state_for_agent, last_action_for_agent, reward,
                     copy.deepcopy(current_agent_state))
                )

            # --- Store Current State/Action ---
            self.last_states[agent_id] = copy.deepcopy(current_agent_state)
            self.last_actions[agent_id] = action

            # --- Execute Physics Step (In-Place) ---
            self.physics.step(GridActions(action), next_state)  # Modifies next_state

            # --- Terminal Experience Collection ---
            # (Keep existing logic, ensure calculate_reward uses agent_id)
            if next_state.terminals[agent_id] and self.last_states[agent_id] is not None:
                final_reward = self.rewards.calculate_reward(
                    self.last_states[agent_id], GridActions(self.last_actions[agent_id]),
                    next_state, agent_id
                )
                self.trainer.add_experience(
                    self.last_states[agent_id], self.last_actions[agent_id],
                    final_reward, next_state
                )
                self.stats.add_experience(
                    (self.last_states[agent_id], self.last_actions[agent_id], final_reward,
                     copy.deepcopy(next_state))
                )
            # --- End of loop for one agent ---

        # --- After processing all agents ---
        self.step_count += 1
        self.first_action = False
        self.grid.state = next_state
        return next_state

    # --- MODIFIED init_episode ---
    def init_episode(self, init_state=None):
        state = super().init_episode(init_state)
       # print("[Environment.init_episode] Base state initialized.")
      #  print("[Environment.init_episode] Resetting rewards and physics...")
        if self.rewards is None: raise ValueError("self.rewards is None")
        if self.physics is None: raise ValueError("self.physics is None")
        self.rewards.reset()
        self.physics.reset(state)
     #   print("[Environment.init_episode] Components reset.")
        self.last_states = [None] * state.num_agents
        self.last_actions = [None] * state.num_agents
        self.first_action = True

        # --- Initialize new state list ---
        state.consecutive_zero_rate_hovers = [0] * state.num_agents

      #  print("[Environment.init_episode] Setting initial targets...")
        for agent_id in range(state.num_agents):
            state.active_agent = agent_id
            self._update_agent_target(state, agent_id) # Set initial target
            # print(f"[Environment.init_episode] Initial target for Agent {agent_id}: {state.target}")

        state.active_agent = 0
        self.grid.state = state
        return state

    def test_scenario(self, scenario):
        """Runs a specific scenario using the exploitation policy."""
        print(f"[Test Scenario] Starting scenario...")
        # Use deepcopy from scenario object if it contains a state object
        if hasattr(scenario, 'init_state') and isinstance(scenario.init_state, State):
             state = copy.deepcopy(self.init_episode(scenario.init_state))
        elif isinstance(scenario, State): # If scenario IS the init_state
             state = copy.deepcopy(self.init_episode(scenario))
        else:
             print("[Test Scenario Error] Invalid scenario object passed.")
             return

        self.stats.on_episode_begin(self.episode_count) # Use episode_count or a scenario counter?

        num_agents = state.num_agents
        test_last_states = [None] * num_agents
        test_last_actions = [None] * num_agents
        current_state_in_test = state

        while not current_state_in_test.all_terminal:
            state_for_next_agent = current_state_in_test
            for agent_id in range(num_agents):
                state_for_this_agent = state_for_next_agent
                state_for_this_agent.active_agent = agent_id

                if state_for_this_agent.terminals[agent_id]:
                    continue

                # Update target using the same logic
                self._update_agent_target(state_for_this_agent, agent_id)

                # Get EXPLOITATION action
                action = self.agent.get_exploitation_action_target(state_for_this_agent)
                action_enum = GridActions(action)

                # --- Log experience to stats (similar to test_episode) ---
                last_state = test_last_states[agent_id]
                last_action = test_last_actions[agent_id]
                if last_state is not None:
                    original_active = state_for_this_agent.active_agent
                    state_for_this_agent.active_agent = agent_id
                    reward = self.rewards.calculate_reward(last_state, GridActions(last_action), state_for_this_agent, agent_id)
                    state_for_this_agent.active_agent = original_active
                    self.stats.add_experience((last_state, last_action, reward, copy.deepcopy(state_for_this_agent)))

                test_last_states[agent_id] = copy.deepcopy(state_for_this_agent)
                test_last_actions[agent_id] = action

                # --- Execute Physics (In-Place) ---
                self.physics.step(action_enum, state_for_this_agent) # Modify state in-place
                state_for_next_agent = state_for_this_agent # Pass modified state

                # --- Log terminal experience ---
                if state_for_next_agent.terminals[agent_id] and test_last_states[agent_id] is not None:
                     original_active = state_for_next_agent.active_agent
                     state_for_next_agent.active_agent = agent_id
                     final_reward = self.rewards.calculate_reward(test_last_states[agent_id], GridActions(test_last_actions[agent_id]), state_for_next_agent, agent_id)
                     state_for_next_agent.active_agent = original_active
                     self.stats.add_experience((test_last_states[agent_id], test_last_actions[agent_id], final_reward, copy.deepcopy(state_for_next_agent)))

            current_state_in_test = state_for_next_agent

        self.stats.on_episode_end(self.episode_count) # Use appropriate counter
        print("[Test Scenario] Finished.")
        # Optionally log scenario data if needed
        # self.stats.log_testing_data(step=some_scenario_step_counter)