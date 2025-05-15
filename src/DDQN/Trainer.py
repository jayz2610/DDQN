from src.DDQN.Agent import DDQNAgent
from src.DDQN.ReplayMemory import ReplayMemory
import tqdm


class DDQNTrainerParams:
    def __init__(self):
        self.batch_size = 128
        self.num_steps = 1e6
        self.rm_pre_fill_ratio = 0.5
        self.rm_pre_fill_random = True
        self.eval_period = 5
        self.rm_size = 50000
        self.load_model = ""
        self.max_stuck_hovers_prefill = 3  # <--- 新增参数及其默认值


class DDQNTrainer:
    def __init__(self, params: DDQNTrainerParams, agent: DDQNAgent):
        self.params = params
        self.replay_memory = ReplayMemory(size=params.rm_size)
        self.agent = agent
        self.use_scalar_input = self.agent.params.use_scalar_input

        if self.params.load_model != "":
            print("Loading model", self.params.load_model, "for agent")
            self.agent.load_weights(self.params.load_model)

        self.prefill_bar = None

    def add_experience(self, state, action, reward, next_state):
        experience_tuple = None  # Initialize
        components_ok = True  # Flag to track if all components are valid

        try:
            # --- Extract components ---
            term = next_state.terminals[next_state.active_agent]  # Get terminal flag first

            if self.use_scalar_input:
                s_dev = state.get_device_scalars(self.agent.params.max_devices, self.agent.params.relative_scalars)
                s_uav = state.get_uav_scalars(self.agent.params.max_uavs, self.agent.params.relative_scalars)
                s_scal = state.get_scalars(give_position=True)
                ns_dev = next_state.get_device_scalars(self.agent.params.max_devices,
                                                       self.agent.params.relative_scalars)
                ns_uav = next_state.get_uav_scalars(self.agent.params.max_uavs, self.agent.params.relative_scalars)
                ns_scal = next_state.get_scalars(give_position=True)
                components = [s_dev, s_uav, s_scal, action, reward, ns_dev, ns_uav, ns_scal, term]
                names = ["s_dev", "s_uav", "s_scal", "action", "reward", "ns_dev", "ns_uav", "ns_scal", "term"]

            else:  # Map based input
                s_bool_map = state.get_boolean_map()
                s_float_map = state.get_float_map()
                s_scalars = state.get_scalars()
                ns_bool_map = next_state.get_boolean_map()
                ns_float_map = next_state.get_float_map()
                ns_scalars = next_state.get_scalars()
                components = [s_bool_map, s_float_map, s_scalars, action, reward,
                              ns_bool_map, ns_float_map, ns_scalars, term]
                names = ["s_bool_map", "s_float_map", "s_scalars", "action", "reward",
                         "ns_bool_map", "ns_float_map", "ns_scalars", "term"]

            # --- Check each component for None ---
            for i, comp in enumerate(components):
                if comp is None:
                    print(f"[Trainer Error] Component '{names[i]}' is None before storing! Skipping experience.")
                    components_ok = False
                    break  # Stop checking if one is None

            # --- Store experience only if all components are valid ---
            if components_ok:
                experience_tuple = tuple(components)
                self.replay_memory.store(experience_tuple)
            # else: The error message was already printed above

        except Exception as e:
            print(f"[Trainer Error] Exception during get/store experience: {e}")
            import traceback
            traceback.print_exc()
            # Optionally re-raise or handle differently

    def train_agent(self):
        if self.params.batch_size > self.replay_memory.get_size():
            return
        mini_batch = self.replay_memory.sample(self.params.batch_size)

        self.agent.train(mini_batch)

    def should_fill_replay_memory(self):
        target_size = self.replay_memory.get_max_size() * self.params.rm_pre_fill_ratio
        if self.replay_memory.get_size() >= target_size or self.replay_memory.full:
            if self.prefill_bar:
                self.prefill_bar.close()
            return False

        if self.prefill_bar is None:
            print("Filling replay memory")
            self.prefill_bar = tqdm.tqdm(total=target_size)

        self.prefill_bar.update(self.replay_memory.get_size() - self.prefill_bar.n)

        return True
