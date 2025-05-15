import numpy as np
from src.Map.Map import Map
from src.StateUtils import pad_centered
from src.base.BaseState import BaseState


class State(BaseState):
    def __init__(self, map_init: Map, num_agents: int, multi_agent: bool):
        super().__init__(map_init)
        #self.global_paths = {}  # Initialize global_paths here as a dictionary

        self.device_list = None
        self.device_map = None
        self.active_agent = 0
        self.num_agents = num_agents
        self.multi_agent = multi_agent

        # Multi-agent is creating lists
        self.positions = [[0, 0] for _ in range(num_agents)]
        self.movement_budgets = [0] * num_agents
        self.landeds = [False] * num_agents
        self.terminals = [False] * num_agents
        self.device_coms = [-1] * num_agents
        # Target is a tuple (x, y) or None if no target
        self.current_targets = [None] * num_agents
        self.last_failed_land_pos = [None] * num_agents
        self.consecutive_zero_rate_hovers = [0] * num_agents
        self.initial_movement_budgets = [0] * num_agents
        self.consecutive_invalid_moves = [0] * num_agents
        self.initial_total_data = 0
        self.collected = None

    @property
    def position(self):
        if hasattr(self, 'positions') and 0 <= self.active_agent < len(self.positions):
            return self.positions[self.active_agent]
        else:
            # 如果位置信息不可用，返回一个默认值或引发错误
            # print(f"[State Warning] Active agent {self.active_agent} position not found.")
            return [0, 0]  # 或者更合适的默认/错误处理

    @property
    def movement_budget(self):
        return self.movement_budgets[self.active_agent]

    @property
    def initial_movement_budget(self):
        return self.initial_movement_budgets[self.active_agent]

    @property
    def landed(self):
        return self.landeds[self.active_agent]

    @property
    def terminal(self):
        return self.terminals[self.active_agent]

    @property
    def all_landed(self):
        return all(self.landeds)

    @property
    def all_terminal(self):
        return all(self.terminals)

    def is_terminal(self):
        return self.all_terminal

    def set_landed(self, landed):
        self.landeds[self.active_agent] = landed

    def set_position(self, position):
        self.positions[self.active_agent] = position

    def decrement_movement_budget(self):
        self.movement_budgets[self.active_agent] -= 1

    def set_terminal(self, terminal):
        self.terminals[self.active_agent] = terminal

    def set_device_com(self, device_com):
        self.device_coms[self.active_agent] = device_com


    @property
    def target(self):
        """Gets the current target tuple (x, y) or None for the active agent."""
        if 0 <= self.active_agent < len(self.current_targets):
            return self.current_targets[self.active_agent]
        return None

    def set_target(self, target_location):
        """Sets the target for the active agent. Target should be tuple (x, y) or None."""
        if 0 <= self.active_agent < len(self.current_targets):
            self.current_targets[self.active_agent] = target_location

    def get_active_agent(self):
        return self.active_agent

    def get_remaining_data(self):
        return np.sum(self.device_map)

    def get_total_data(self):
        return self.initial_total_data

    def get_scalars(self, give_position=False,max_hovers_for_norm=10.0,
                    max_invalid_moves_for_norm=5.0):

        # 首先，获取地图维度和当前活动智能体的位置
        if hasattr(self, 'no_fly_zone') and self.no_fly_zone is not None:
            map_height, map_width = self.no_fly_zone.shape[:2]
        elif hasattr(self, 'shape') and self.shape is not None:  # 后备方案
            map_height, map_width = self.shape[:2]
        else:
            print("[State Error] Map dimensions unknown in get_scalars. Cannot proceed.")
            # 如果无法获取地图维度，返回一个空数组或包含错误指示的数组
            # 这个错误需要解决，否则所有依赖维度的计算都会失败
            return np.array([], dtype=np.float32)

        current_agent_position = self.position  # 使用 @property position 获取当前活动智能体的位置

        # 确保 current_agent_position 是有效的列表或元组 [x, y]
        if not (isinstance(current_agent_position, (list, tuple)) and len(current_agent_position) == 2):
            print(
                f"[State Error] Invalid current_agent_position format: {current_agent_position} for agent {self.active_agent}. Cannot proceed.")
            # 返回空数组或错误指示
            return np.array([], dtype=np.float32)

        current_pos_x, current_pos_y = current_agent_position[0], current_agent_position[1]

        scalars = []
        # --- 预算归一化部分 (你已有的逻辑) ---
        norm_budget = 0.0
        current_initial_budget = 0
        if hasattr(self, 'initial_movement_budgets') and \
                0 <= self.active_agent < len(self.initial_movement_budgets):
            current_initial_budget = self.initial_movement_budgets[self.active_agent]
        elif hasattr(self, 'initial_movement_budget'):  # 兼容单个预算值的情况
            current_initial_budget = self.initial_movement_budget

        if current_initial_budget > 0:
            current_budget = 0
            if hasattr(self, 'movement_budgets') and \
                    0 <= self.active_agent < len(self.movement_budgets):
                current_budget = self.movement_budgets[self.active_agent]
            elif hasattr(self, 'movement_budget'):  # 兼容单个预算值
                current_budget = self.movement_budget
            norm_budget = current_budget / current_initial_budget
        else:
            current_budget = 0  # Default if no budget info
            if hasattr(self, 'movement_budgets') and \
                    0 <= self.active_agent < len(self.movement_budgets):
                current_budget = self.movement_budgets[self.active_agent]
            elif hasattr(self, 'movement_budget'):
                current_budget = self.movement_budget
            norm_budget = 0.0 if current_budget <= 0 else 1.0
        # --- 结束预算归一化修改 ---

        scalars = [norm_budget]

        if give_position:
            pos = self.position  # self.position 应该返回当前活动智能体的位置
            norm_pos_x = 0.0
            norm_pos_y = 0.0
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                # Normalize position (0 to 1)
                norm_pos_x = pos[0] / map_width if map_width > 0 else 0.0
                norm_pos_y = pos[1] / map_height if map_height > 0 else 0.0
            # 使用 np.clip确保值在[0,1]范围内
            scalars.extend([np.clip(norm_pos_x, 0.0, 1.0), np.clip(norm_pos_y, 0.0, 1.0)])

        # --- 目标相对距离 tdx, tdy 部分 ---
        target = self.target  # self.target 应该返回当前活动智能体的目标
        norm_tdx, norm_tdy = 0.0, 0.0
        current_pos = self.position  # self.position 同上

        if target is not None and isinstance(current_pos, (list, tuple)) and len(current_pos) >= 2:
            try:
                tdx = target[0] - current_pos[0]
                tdy = target[1] - current_pos[1]
                # Normalize tdx, tdy (approx -1 to 1)
                norm_tdx = tdx / map_width if map_width > 0 else 0.0
                norm_tdy = tdy / map_height if map_height > 0 else 0.0
            except (TypeError, IndexError) as e:
                norm_tdx, norm_tdy = 0.0, 0.0
        # 使用 np.clip 确保值在[-1,1]范围内
        scalars.extend([np.clip(norm_tdx, -1.0, 1.0), np.clip(norm_tdy, -1.0, 1.0)])

        # --- 未产出悬停次数 unproductive_hovers 部分 ---
        unproductive_hovers_raw = 0  # 先获取原始值
        if hasattr(self, 'consecutive_zero_rate_hovers') and \
                0 <= self.active_agent < len(self.consecutive_zero_rate_hovers):
            unproductive_hovers_raw = self.consecutive_zero_rate_hovers[self.active_agent]

     # max_expected_hovers = 10.0  # 这个值最好是可配置的，或者从 reward_params/agent_params 获取
        norm_unproductive_hovers = 0.0
        if max_hovers_for_norm > 0:
            norm_unproductive_hovers = min(unproductive_hovers_raw / max_hovers_for_norm,1.0)
        elif unproductive_hovers_raw > 0:  # 如果 max_expected_hovers 为0但实际有hovers，则设为1
            norm_unproductive_hovers = 1.0

        scalars.append(norm_unproductive_hovers)

        # --- 5. 连续无效移动次数 consecutive_hits 部分 ---
        raw_consecutive_hits = 0
        if hasattr(self, 'consecutive_invalid_moves') and \
                0 <= self.active_agent < len(self.consecutive_invalid_moves):
            raw_consecutive_hits = self.consecutive_invalid_moves[self.active_agent]
        # max_hits_for_state_norm 这个值最好是可配置的
        max_hits_for_state_norm = getattr(self.params if hasattr(self, 'params') else object(),
                                          'max_invalid_moves_for_norm', 5.0)
        norm_consecutive_hits = 0.0
        if max_hits_for_state_norm > 0:
            norm_consecutive_hits = min(raw_consecutive_hits / max_hits_for_state_norm, 1.0)
        elif raw_consecutive_hits > 0:
            norm_consecutive_hits = 1.0
        scalars.append(norm_consecutive_hits)

        # ========== 新增：明确的边界指示标量 ==========
        # 确保 current_pos_for_td (即 self.position) 是有效的 [x,y] 列表或元组
        at_north_boundary = 1.0 if current_pos_y == 0 else 0.0
        at_south_boundary = 1.0 if current_pos_y >= map_height - 1 else 0.0
        at_west_boundary = 1.0 if current_pos_x == 0 else 0.0
        at_east_boundary = 1.0 if current_pos_x >= map_width - 1 else 0.0

        scalars.extend([at_north_boundary, at_south_boundary, at_west_boundary, at_east_boundary])

        return np.array(scalars, dtype=np.float32)

    def get_num_scalars(self, give_position=False):
        """Calculates the total number of scalars returned."""
        count = 1 + 2 + 1 + 1 + 4  # budget + target_dx_dy + unprod_hovers + consecutive_hits
        if give_position:
            count += 2
        return count

    def get_boolean_map(self):
        padded_red = pad_centered(self, np.concatenate([np.expand_dims(self.no_fly_zone, -1),
                                                        np.expand_dims(self.obstacles, -1)], axis=-1), 1)
        if self.multi_agent:
            padded_rest = pad_centered(self,np.concatenate([np.expand_dims(self.landing_zone, -1),
                                                            self.get_agent_bool_maps()],axis=-1), 0)
        else:
            padded_rest = pad_centered(self, np.expand_dims(self.landing_zone, -1), 0)
        return np.concatenate([padded_red, padded_rest], axis=-1)

    def get_boolean_map_shape(self):
        return self.get_boolean_map().shape

    def get_float_map(self):
        if self.multi_agent:
            return pad_centered(self, np.concatenate([np.expand_dims(self.device_map, -1),
                                                      self.get_agent_float_maps()], axis=-1), 0)
        else:
            return pad_centered(self, np.expand_dims(self.device_map, -1), 0)

    def get_float_map_shape(self):
        return self.get_float_map().shape

    def is_in_landing_zone(self):
        # Check if the landing_zone map exists and is valid
        if hasattr(self, 'landing_zone') and self.landing_zone is not None \
                and isinstance(self.landing_zone, np.ndarray):
            try:
                h, w = self.landing_zone.shape[:2]  # Get map dimensions
                # Ensure position is valid list/tuple before accessing elements
                current_pos = self.position  # Get active agent's position
                if not isinstance(current_pos, (list, tuple)) or len(current_pos) < 2:
                    print(f"[LANDING CHECK Error] Invalid position format: {current_pos}")
                    return False

                y, x = int(round(current_pos[1])), int(round(current_pos[0]))  # Use rounded int indices

                # Check bounds before indexing
                if 0 <= y < h and 0 <= x < w:
                    zone_value = self.landing_zone[y, x]
                    print(f"[LANDING CHECK] Pos:({x},{y}), Bounds:(W:{w},H:{h}), ZoneValue:{zone_value}")  # DEBUG PRINT
                    return zone_value  # Return the boolean value from the map
                else:
                    print(f"[LANDING CHECK] Pos:({x},{y}) is OUT OF BOUNDS (W:{w}, H:{h})")  # DEBUG PRINT
                    return False  # Out of bounds means not in landing zone
            except Exception as e:
                print(f"[LANDING CHECK Error] Error during check for pos {self.position}: {e}")
                return False  # Error during check
        else:
            print("[LANDING CHECK Error] landing_zone attribute missing or not a numpy array.")
            return False  # Not in zone if map missing

    def is_in_no_fly_zone(self):
        if not hasattr(self, 'no_fly_zone') or self.no_fly_zone is None:
            return True  # Treat missing NFZ map as entirely no-fly (safer)

        h, w = self.no_fly_zone.shape[:2]
        y, x = self.position[1], self.position[0]

        # Out of bounds check
        if not (0 <= y < h and 0 <= x < w):
            return True  # Out of bounds is implicitly NFZ

        # Check NFZ map and occupied status
        return self.no_fly_zone[y, x] or self.is_occupied()

    def is_occupied(self):
        if not self.multi_agent:
            return False
        for i, pos in enumerate(self.positions):
            if self.terminals[i]:
                continue
            if i == self.active_agent:
                continue
            if pos == self.position:
                return True
        return False

    def get_collection_ratio(self):
        return np.sum(self.collected) / self.initial_total_data

    def get_collected_data(self):
        return np.sum(self.collected)

    def reset_devices(self, device_list):
        # Ensure shape comes from a valid map layer
        map_shape = self.no_fly_zone.shape if hasattr(self, 'no_fly_zone') and self.no_fly_zone is not None else None
        if map_shape is None:
             print("[State Error] Cannot reset devices, map shape unknown.")
             return

        self.device_map = device_list.get_data_map(map_shape)
        self.collected = np.zeros(map_shape, dtype=float)
        self.initial_total_data = device_list.get_total_data()
        self.device_list = device_list # Store the whole DeviceList object

    def get_agent_bool_maps(self):
        agent_map = np.zeros(self.no_fly_zone.shape + (1,), dtype=bool)
        for agent in range(self.num_agents):
            # agent_map[self.positions[agent][1], self.positions[agent][0]][0] = self.landeds[agent]
            agent_map[self.positions[agent][1], self.positions[agent][0]][0] = not self.terminals[agent]
        return agent_map

    def get_agent_float_maps(self):
        agent_map = np.zeros(self.no_fly_zone.shape + (1,), dtype=float)
        for agent in range(self.num_agents):
            agent_map[self.positions[agent][1], self.positions[agent][0]][0] = self.movement_budgets[agent]
        return agent_map

    def get_device_scalars(self, max_num_devices, relative):
        devices = np.zeros(3 * max_num_devices, dtype=np.float32)
        if relative:
            for k, dev in enumerate(self.device_list.devices):
                devices[k * 3] = dev.position[0] - self.position[0]
                devices[k * 3 + 1] = dev.position[1] - self.position[1]
                devices[k * 3 + 2] = dev.data - dev.collected_data
        else:
            for k, dev in enumerate(self.device_list.devices):
                devices[k * 3] = dev.position[0]
                devices[k * 3 + 1] = dev.position[1]
                devices[k * 3 + 2] = dev.data - dev.collected_data
        return devices

    def get_uav_scalars(self, max_num_uavs, relative):
        uavs = np.zeros(4 * max_num_uavs, dtype=np.float32)
        if relative:
            for k in range(max_num_uavs):
                if k >= self.num_agents:
                    break
                uavs[k * 4] = self.positions[k][0] - self.position[0]
                uavs[k * 4 + 1] = self.positions[k][1] - self.position[1]
                uavs[k * 4 + 2] = self.movement_budgets[k]
                uavs[k * 4 + 3] = not self.terminals[k]
        else:
            for k in range(max_num_uavs):
                if k >= self.num_agents:
                    break
                uavs[k * 4] = self.positions[k][0]
                uavs[k * 4 + 1] = self.positions[k][1]
                uavs[k * 4 + 2] = self.movement_budgets[k]
                uavs[k * 4 + 3] = not self.terminals[k]
        return uavs
