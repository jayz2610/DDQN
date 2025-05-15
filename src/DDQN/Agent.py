import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D

import numpy as np

from src.base.GridActions import GridActions


def print_node(x):
    print(x)
    return x


class DDQNAgentParams:
    def __init__(self):
        # Convolutional part config
        self.conv_layers = 2
        self.conv_kernel_size = 5
        self.conv_kernels = 16

        # Fully Connected config
        self.hidden_layer_size = 256
        self.hidden_layer_num = 3

        # Training Params
        self.learning_rate = 3e-5
        self.alpha = 0.005
        self.gamma = 0.95

        # Exploration strategy
        self.soft_max_scaling = 0.1

        # Global-Local Map
        self.global_map_scaling = 3
        self.local_map_size = 17

        # Scalar inputs instead of map
        self.use_scalar_input = False
        self.relative_scalars = False
        self.blind_agent = False
        self.max_uavs = 3
        self.max_devices = 10

        self.scalar_norm_max_hovers = 5.0  # 对应之前的 max_expected_hovers
        self.scalar_norm_max_invalid_moves = 5.0  # 对应之前的 max_hits_for_state_norm
        # Printing
        self.print_summary = False


class DDQNAgent(object):

    def __init__(self, params: DDQNAgentParams, example_state, example_action, stats=None):

        self.params = params
        gamma = tf.constant(self.params.gamma, dtype=float)
        self.align_counter = 0

        self.boolean_map_shape = example_state.get_boolean_map_shape()
        self.float_map_shape = example_state.get_float_map_shape()
        self.scalars = example_state.get_num_scalars(give_position=self.params.use_scalar_input)
        self.num_actions = len(type(example_action))
        self.num_map_channels = self.boolean_map_shape[2] + self.float_map_shape[2]

        # Create shared inputs
        action_input = Input(shape=(), name='action_input', dtype=tf.int64)
        reward_input = Input(shape=(), name='reward_input', dtype=tf.float32)
        termination_input = Input(shape=(), name='termination_input', dtype=tf.bool)
        q_star_input = Input(shape=(), name='q_star_input', dtype=tf.float32)

        if self.params.blind_agent:
            scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
            states = [scalars_input]
            self.q_network = self.build_blind_model(scalars_input)
            self.target_network = self.build_blind_model(scalars_input, 'target_')
            self.hard_update()

        elif self.params.use_scalar_input:
            devices_input = Input(shape=(3 * self.params.max_devices,), name='devices_input', dtype=tf.float32)
            uavs_input = Input(shape=(4 * self.params.max_uavs,), name='uavs_input', dtype=tf.float32)
            scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
            states = [devices_input,
                      uavs_input,
                      scalars_input]

            self.q_network = self.build_scalars_model(states)
            self.target_network = self.build_scalars_model(states, 'target_')
            self.hard_update()

        else:
            boolean_map_input = Input(shape=self.boolean_map_shape, name='boolean_map_input', dtype=tf.bool)
            float_map_input = Input(shape=self.float_map_shape, name='float_map_input', dtype=tf.float32)
            scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
            states = [boolean_map_input,
                      float_map_input,
                      scalars_input]

            map_cast = tf.cast(boolean_map_input, dtype=tf.float32)
            padded_map = tf.concat([map_cast, float_map_input], axis=3)

            self.q_network = self.build_model(padded_map, scalars_input, states)
            self.target_network = self.build_model(padded_map, scalars_input, states, 'target_')
            self.hard_update()

            self.global_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                          outputs=self.global_map)
            self.local_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                         outputs=self.local_map)
            self.total_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                         outputs=self.total_map)

        q_values = self.q_network.output
        q_target_values = self.target_network.output

        # Define Q* in min(Q - (r + gamma_terminated * Q*))^2
        max_action = tf.argmax(q_values, axis=1, name='max_action', output_type=tf.int64)
        max_action_target = tf.argmax(q_target_values, axis=1, name='max_action', output_type=tf.int64)
        one_hot_max_action = tf.one_hot(max_action, depth=self.num_actions, dtype=float)
        q_star = tf.reduce_sum(tf.multiply(one_hot_max_action, q_target_values, name='mul_hot_target'), axis=1,
                               name='q_star')
        self.q_star_model = Model(inputs=states, outputs=q_star)

        # Define Bellman loss
        one_hot_rm_action = tf.one_hot(action_input, depth=self.num_actions, on_value=1.0, off_value=0.0, dtype=float)
        one_cold_rm_action = tf.one_hot(action_input, depth=self.num_actions, on_value=0.0, off_value=1.0, dtype=float)
        q_old = tf.stop_gradient(tf.multiply(q_values, one_cold_rm_action))
        gamma_terminated = tf.multiply(tf.cast(tf.math.logical_not(termination_input), tf.float32), gamma)
        q_update = tf.expand_dims(tf.add(reward_input, tf.multiply(q_star_input, gamma_terminated)), 1)
        q_update_hot = tf.multiply(q_update, one_hot_rm_action)
        q_new = tf.add(q_update_hot, q_old)
        q_loss = tf.losses.MeanSquaredError()(q_new, q_values)
        self.q_loss_model = Model(
            inputs=states + [action_input, reward_input, termination_input, q_star_input],
            outputs=q_loss)

        # Exploit act model
        self.exploit_model = Model(inputs=states, outputs=max_action)
        self.exploit_model_target = Model(inputs=states, outputs=max_action_target)

        # Softmax explore model
        softmax_scaling = tf.divide(q_values, tf.constant(self.params.soft_max_scaling, dtype=float))
        softmax_action = tf.math.softmax(softmax_scaling, name='softmax_action')
        self.soft_explore_model = Model(inputs=states, outputs=softmax_action)

        self.q_optimizer = tf.optimizers.Adam(learning_rate=params.learning_rate, amsgrad=True)

        if self.params.print_summary:
            self.q_loss_model.summary()

        if stats:
            stats.set_model(self.target_network)

    def build_model(self, map_proc, states_proc, inputs, name=''):

        flatten_map = self.create_map_proc(map_proc, name)

        layer = Concatenate(name=name + 'concat')([flatten_map, states_proc])
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(
                layer)
        output = Dense(self.num_actions, activation='linear', name=name + 'output_layer')(layer)

        model = Model(inputs=inputs, outputs=output)

        return model

    def build_scalars_model(self, inputs, name=''):
        layer = Concatenate(name=name + 'concat')(inputs)
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu')(layer)
        output = Dense(self.num_actions, activation='linear')(layer)
        return Model(inputs=inputs, outputs=output)

    def build_blind_model(self, inputs, name=''):
        layer = inputs
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(
                layer)
        output = Dense(self.num_actions, activation='linear', name=name + 'output_layer')(layer)

        model = Model(inputs=inputs, outputs=output)

        return model

    def create_map_proc(self, conv_in, name):

        # Forking for global and local map
        # Global Map
        global_map = tf.stop_gradient(
            AvgPool2D((self.params.global_map_scaling, self.params.global_map_scaling))(conv_in))

        self.global_map = global_map
        self.total_map = conv_in

        for k in range(self.params.conv_layers):
            global_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                                strides=(1, 1),
                                name=name + 'global_conv_' + str(k + 1))(global_map)

        flatten_global = Flatten(name=name + 'global_flatten')(global_map)

        # Local Map
        crop_frac = float(self.params.local_map_size) / float(self.boolean_map_shape[0])
        local_map = tf.stop_gradient(tf.image.central_crop(conv_in, crop_frac))
        self.local_map = local_map

        for k in range(self.params.conv_layers):
            local_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                               strides=(1, 1),
                               name=name + 'local_conv_' + str(k + 1))(local_map)

        flatten_local = Flatten(name=name + 'local_flatten')(local_map)

        return Concatenate(name=name + 'concat_flatten')([flatten_global, flatten_local])

    def act(self, state):
        return self.get_soft_max_exploration(state)

    def get_random_action(self):
        return np.random.randint(0, self.num_actions)

    def get_exploitation_action(self, state):

        if self.params.blind_agent:
            scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]
            return self.exploit_model(scalars).numpy()[0]

        if self.params.use_scalar_input:
            devices_in = state.get_device_scalars(self.params.max_devices, relative=self.params.relative_scalars)[tf.newaxis, ...]
            uavs_in = state.get_uav_scalars(self.params.max_uavs, relative=self.params.relative_scalars)[tf.newaxis, ...]
            scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]
            return self.exploit_model([devices_in, uavs_in, scalars]).numpy()[0]

        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]

        return self.exploit_model([boolean_map_in, float_map_in, scalars]).numpy()[0]

    def get_soft_max_exploration(self, state):
        # 根据你的 Agent 配置 (blind_agent, use_scalar_input) 准备正确的网络输入
        # 下面是一个针对 use_scalar_input=False (即使用地图+标量) 的示例
        # 你需要根据你的 Agent 配置调整这部分输入准备逻辑

        active_agent_id_for_debug = state.active_agent  # 获取当前活动agent的ID，用于打印
        if self.params.blind_agent:
            # === BLIND AGENT INPUT PREPARATION ===
            scalars_np_for_debug = np.array(state.get_scalars(give_position=True), dtype=np.single)  # 用于打印
            scalars_tf_input = scalars_np_for_debug[tf.newaxis, ...]

            # 使用 training=False 来获取Q值，以确保行为一致性（例如关闭Dropout）
            raw_q_values = self.q_network(scalars_tf_input, training=False).numpy()[0]
            p = self.soft_explore_model(scalars_tf_input, training=False).numpy()[0]
            is_at_lz_target = False

            if hasattr(state, 'position') and hasattr(state, 'target') and \
                    state.target is not None and tuple(state.position) == tuple(state.target):
                # 这个 self.env._is_position_landing_zone 需要 agent 能访问到这个方法
                # 或者 state 对象自身能判断当前位置是否是着陆区
                # 假设 state 对象有 is_in_landing_zone 方法判断 state.position
                if hasattr(state, 'is_in_landing_zone') and state.is_in_landing_zone():
                    is_at_lz_target = True

            if is_at_lz_target:
                active_agent_id_for_debug = state.active_agent  # 假设有 active_agent
                print(
                    f"[Q-DEBUG LANDING] Agent {active_agent_id_for_debug} AT LANDING TARGET {state.position}. Q-values:")
                q_value_str = ", ".join(
                    [f"{GridActions(i).name}:{raw_q_values[i]:.4f}" for i in range(self.num_actions)])
                print(f"[Q-DEBUG LANDING] Raw Q-Values: {q_value_str}")
                prob_str = ", ".join([f"{GridActions(i).name}:{p[i]:.4f}" for i in range(self.num_actions)])
                print(f"[Q-DEBUG LANDING] Softmax Probs: {prob_str}")

            # ========== Q 值和状态打印 ==========
            # 你可以添加条件只在特定情况下打印，例如 if tuple(state.position) == (0,0):
            if tuple(state.position) == (0, 0) and active_agent_id_for_debug == 0:  # 例如只为agent 0在(0,0)时打印
                print(
                    f"[Q-DEBUG] Agent {active_agent_id_for_debug} Pos: {state.position}, Target: {state.target}, Budget: {state.movement_budget}")
                print(f"[Q-DEBUG] Scalars (Blind): {scalars_np_for_debug}")
                q_value_str = ", ".join(
                    [f"{GridActions(i).name}:{raw_q_values[i]:.3f}" for i in range(self.num_actions)])
                print(f"[Q-DEBUG] Raw Q-Values: {q_value_str}")
                prob_str = ", ".join([f"{GridActions(i).name}:{p[i]:.3f}" for i in range(self.num_actions)])
                print(f"[Q-DEBUG] Softmax Probs: {prob_str}")
            # ========== 结束打印 ==========

        elif self.params.use_scalar_input:
            # === SCALAR INPUT AGENT PREPARATION ===
            devices_in_np = state.get_device_scalars(self.params.max_devices, relative=self.params.relative_scalars)
            uavs_in_np = state.get_uav_scalars(self.params.max_uavs, relative=self.params.relative_scalars)
            scalars_np_for_debug = np.array(state.get_scalars(give_position=True), dtype=np.single)  # 用于打印

            devices_tf_input = devices_in_np[tf.newaxis, ...]
            uavs_tf_input = uavs_in_np[tf.newaxis, ...]
            scalars_tf_input = scalars_np_for_debug[tf.newaxis, ...]

            model_inputs = [devices_tf_input, uavs_tf_input, scalars_tf_input]
            raw_q_values = self.q_network(model_inputs, training=False).numpy()[0]
            p = self.soft_explore_model(model_inputs, training=False).numpy()[0]

            # ========== Q 值和状态打印 ==========
            if tuple(state.position) == (0, 0) and active_agent_id_for_debug == 0:
                print(
                    f"[Q-DEBUG] Agent {active_agent_id_for_debug} Pos: {state.position}, Target: {state.target}, Budget: {state.movement_budget}, ConsecutiveHits: {state.consecutive_invalid_moves[active_agent_id_for_debug] if hasattr(state, 'consecutive_invalid_moves') else 'N/A'}")
                # print(f"[Q-DEBUG] Scalars (ScalarInput): {scalars_np_for_debug}") # 可能会很长
                q_value_str = ", ".join(
                    [f"{GridActions(i).name}:{raw_q_values[i]:.3f}" for i in range(self.num_actions)])
                print(f"[Q-DEBUG] Raw Q-Values: {q_value_str}")
                prob_str = ", ".join([f"{GridActions(i).name}:{p[i]:.3f}" for i in range(self.num_actions)])
                print(f"[Q-DEBUG] Softmax Probs: {prob_str}")
            # ========== 结束打印 ==========

        else:  # 使用地图 + 标量输入
            # === MAP + SCALAR INPUT AGENT PREPARATION ===
            boolean_map_in_tf = tf.convert_to_tensor(state.get_boolean_map()[tf.newaxis, ...])
            float_map_in_tf = tf.convert_to_tensor(state.get_float_map()[tf.newaxis, ...])
            # scalars_np_for_debug = np.array(state.get_scalars(give_position=self.params.use_scalar_input), dtype=np.single) # 根据配置获取give_position
            # 在这里，因为不是 use_scalar_input=True，所以 get_scalars 应该不带 give_position=True (除非你的模型设计如此)
            # 查阅 DDQNAgent __init__ 中 scalars 的获取方式: self.scalars = example_state.get_num_scalars(give_position=self.params.use_scalar_input)
            # 这表明 get_scalars 的 give_position 参数应与 params.use_scalar_input 一致。
            # 但对于地图模型，通常只传递不含显式位置的标量，位置信息由地图提供。
            # 假设对于地图模型，我们使用不含显式位置的标量：
            scalars_np_for_debug = np.array(state.get_scalars(give_position=False), dtype=np.single)
            scalars_tf_input = scalars_np_for_debug[tf.newaxis, ...]

            model_inputs = [boolean_map_in_tf, float_map_in_tf, scalars_tf_input]
            raw_q_values = self.q_network(model_inputs, training=False).numpy()[0]
            p = self.soft_explore_model(model_inputs, training=False).numpy()[0]

            # ========== Q 值和状态打印 ==========
            if tuple(state.position) == (0, 0) and active_agent_id_for_debug == 0:
                print(
                    f"[Q-DEBUG] Agent {active_agent_id_for_debug} Pos: {state.position}, Target: {state.target}, Budget: {state.movement_budget}, ConsecutiveHits: {state.consecutive_invalid_moves[active_agent_id_for_debug] if hasattr(state, 'consecutive_invalid_moves') else 'N/A'}")
                # print(f"[Q-DEBUG] Scalars (MapInput): {scalars_np_for_debug}") # 可能会很长
                q_value_str = ", ".join([f"{GridActions(i).name}:{raw_q_values[i]:.3f}" for i in
                                         range(self.num_actions)])  # 使用 GridActions 枚举值作为索引
                print(f"[Q-DEBUG] Raw Q-Values: {q_value_str}")
                prob_str = ", ".join([f"{GridActions(i).name}:{p[i]:.3f}" for i in range(self.num_actions)])
                print(f"[Q-DEBUG] Softmax Probs: {prob_str}")
            # ========== 结束打印 ==========

            # 检查并修正概率分布 p
        if np.any(np.isnan(p)) or np.abs(np.sum(p) - 1.0) > 1e-5 or np.any(p < 0):
            # print(f"[Q-DEBUG WARNING] Invalid probability distribution p: {p}. Sum: {np.sum(p)}. Forcing uniform.")
            p = np.full(self.num_actions, 1.0 / self.num_actions, dtype=np.float32)  # 确保类型匹配

        return np.random.choice(self.num_actions, size=1, p=p)[0]  # 返回选择的动作的整数值

    def get_exploitation_action_target(self, state):

        if self.params.blind_agent:
            scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]
            return self.exploit_model_target(scalars).numpy()[0]

        if self.params.use_scalar_input:
            devices_in = state.get_device_scalars(self.params.max_devices, relative=self.params.relative_scalars)[tf.newaxis, ...]
            uavs_in = state.get_uav_scalars(self.params.max_uavs, relative=self.params.relative_scalars)[tf.newaxis, ...]
            scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]

            return self.exploit_model_target([devices_in, uavs_in, scalars]).numpy()[0]

        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]

        return self.exploit_model_target([boolean_map_in, float_map_in, scalars]).numpy()[0]

    def hard_update(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def soft_update(self, alpha):
        weights = self.q_network.get_weights()
        target_weights = self.target_network.get_weights()
        self.target_network.set_weights(
            [w_new * alpha + w_old * (1. - alpha) for w_new, w_old in zip(weights, target_weights)])

    def train(self, experiences):
        boolean_map = experiences[0]
        float_map = experiences[1]
        scalars = tf.convert_to_tensor(experiences[2], dtype=tf.float32)
        action = tf.convert_to_tensor(experiences[3], dtype=tf.int64)
        reward = experiences[4]
        next_boolean_map = experiences[5]
        next_float_map = experiences[6]
        next_scalars = tf.convert_to_tensor(experiences[7], dtype=tf.float32)
        terminated = experiences[8]

        if self.params.blind_agent:
            q_star = self.q_star_model(
                [next_scalars])
        else:
            q_star = self.q_star_model(
                [next_boolean_map, next_float_map, next_scalars])

        # Train Value network
        with tf.GradientTape() as tape:

            if self.params.blind_agent:
                q_loss = self.q_loss_model(
                    [scalars, action, reward,
                     terminated, q_star])
            else:
                q_loss = self.q_loss_model(
                    [boolean_map, float_map, scalars, action, reward,
                     terminated, q_star])
        q_grads = tape.gradient(q_loss, self.q_network.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grads, self.q_network.trainable_variables))

        self.soft_update(self.params.alpha)

    def save_weights(self, path_to_weights):
        self.target_network.save_weights(path_to_weights)

    def save_model(self, path_to_model):
        self.target_network.save(path_to_model)

    def load_weights(self, path_to_weights):
        self.q_network.load_weights(path_to_weights)
        self.hard_update()

    def get_global_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.global_map_model([boolean_map_in, float_map_in]).numpy()

    def get_local_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.local_map_model([boolean_map_in, float_map_in]).numpy()

    def get_total_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.total_map_model([boolean_map_in, float_map_in]).numpy()
