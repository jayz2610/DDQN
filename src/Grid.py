import numpy as np

from src.DeviceManager import DeviceManagerParams, DeviceManager
from src.State import State
from src.base.BaseGrid import BaseGrid, BaseGridParams


class GridParams(BaseGridParams):
    def __init__(self):
        super().__init__()
        self.num_agents_range = [1, 3]
        self.device_manager = DeviceManagerParams()
        self.multi_agent = False
        self.fixed_starting_idcs = False
        self.starting_idcs = [1, 2, 3]


class Grid(BaseGrid):

    def __init__(self, params: GridParams, stats):
        super().__init__(params, stats)
        self.params = params
        if params.multi_agent:
            self.num_agents = params.num_agents_range[0]
        else:
            self.num_agents = 1

        self.device_list = None
        self.device_manager = DeviceManager(self.params.device_manager)

        free_space = np.logical_not(
            np.logical_or(self.map_image.obstacles, self.map_image.start_land_zone))
        free_idcs = np.where(free_space)
        self.device_positions = list(zip(free_idcs[1], free_idcs[0]))

    def get_comm_obstacles(self):
        return self.map_image.obstacles

    def get_data_map(self):
        return self.device_list.get_data_map(self.shape)

    def get_collected_map(self):
        return self.device_list.get_collected_map(self.shape)

    def get_device_list(self):
        return self.device_list

    def get_grid_params(self):
        return self.params

    def init_episode(self):
        # Generate devices for this episode
        self.device_list = self.device_manager.generate_device_list(self.device_positions)

        # Determine number of agents for this episode
        if self.params.multi_agent:
            # Ensure range values are valid integers
            low = int(self.params.num_agents_range[0])
            high = int(self.params.num_agents_range[1])
            self.num_agents = np.random.randint(low=low, high=high + 1)  # High is exclusive
        else:
            self.num_agents = 1

        # Create the state object
        state = State(self.map_image, self.num_agents, self.params.multi_agent)
        # Initialize device maps in the state object
        state.reset_devices(self.device_list)

        # ---Starting Position Logic ---
        if self.params.fixed_starting_idcs:
            # Check if starting_idcs parameter exists and has enough positions
            if not hasattr(self.params, 'starting_idcs'):
                raise AttributeError("fixed_starting_idcs is True, but 'starting_idcs' is missing in grid_params")
            if len(self.params.starting_idcs) < self.num_agents:
                raise ValueError(
                    f"Not enough starting positions provided in params.starting_idcs ({len(self.params.starting_idcs)}) for num_agents ({self.num_agents})")

            # Directly assign the first num_agents positions from the list
            # Ensure the positions are in the format state expects (e.g., list of lists)
            state.positions = [list(pos) for pos in self.params.starting_idcs[:self.num_agents]]
            print(f"[Grid Init] Using fixed starting positions: {state.positions}")  # Debug
        else:
            # Randomly choose distinct indices from the available starting_vector
            if len(self.starting_vector) < self.num_agents:
                raise ValueError(
                    f"Not enough available starting positions in map ({len(self.starting_vector)}) for num_agents ({self.num_agents})")
            # Use np.random.choice to get unique indices
            chosen_indices = np.random.choice(len(self.starting_vector), size=self.num_agents, replace=False)
            # Assign positions based on chosen indices
            state.positions = [list(self.starting_vector[i]) for i in chosen_indices]  # Use list() for consistency
            print(f"[Grid Init] Using random starting positions: {state.positions}")  # Debug
        # --- END CORRECTED Starting Position Logic ---

        # Initialize movement budgets
        # Ensure range values are valid integers
        low_budget = int(self.params.movement_range[0])
        high_budget = int(self.params.movement_range[1])
        state.movement_budgets = np.random.randint(low=low_budget,
                                                   high=high_budget + 1,  # High is exclusive
                                                   size=self.num_agents)
        state.initial_movement_budgets = state.movement_budgets.copy()
        print(f"[Grid Init] Initial budgets: {state.movement_budgets}")  # Debug

        # Initialize other necessary list-based states based on num_agents
        state.landeds = [False] * self.num_agents
        state.terminals = [False] * self.num_agents
        state.device_coms = [-1] * self.num_agents
        state.current_targets = [None] * self.num_agents  # Initialize targets if using Strategy 2

        return state

    def init_scenario(self, scenario):
        self.device_list = scenario.device_list
        self.num_agents = scenario.init_state.num_agents

        return scenario.init_state

    def get_example_state(self):
        if self.params.multi_agent:
            num_agents = self.params.num_agents_range[0]
        else:
            num_agents = 1
        state = State(self.map_image, num_agents, self.params.multi_agent)
        state.device_map = np.zeros(self.shape, dtype=float)
        state.collected = np.zeros(self.shape, dtype=float)
        return state

    def get_optimal_path(self, start, goal):
        # Returns the optimal path using A* algorithm
        return self.map_image.a_star(start, goal)

