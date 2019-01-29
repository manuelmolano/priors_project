import tensorflow as tf
import tensorflow.contrib.slim as slim
import networks as nts
import utils as ut
import numpy as np


class AC_Network():
    def __init__(self, a_size, state_size, scope, trainer, num_units, network):
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.st = tf.placeholder(shape=[None, 1, state_size, 1],
                                     dtype=tf.float32)
            self.prev_rewards = tf.placeholder(shape=[None, 1],
                                               dtype=tf.float32)
            self.prev_actions = tf.placeholder(shape=[None],
                                               dtype=tf.int32)

            self.prev_actions_onehot = tf.one_hot(self.prev_actions, a_size,
                                                  dtype=tf.float32)

            hidden = tf.concat([slim.flatten(self.st), self.prev_rewards,
                                self.prev_actions_onehot], 1)

            # call RNN network
            if network == 'relu':
                net = nts.RNN_ReLU
            elif network == 'lstm':
                net = nts.RNN
            elif network == 'gru':
                net = nts.RNN_GRU
            elif network == 'ugru':
                net = nts.RNN_UGRU
            else:
                raise ValueError('Unknown network')

            self.st_init, self.st_in, self.st_out, self.actions,\
                self.actions_onehot, self.policy, self.value =\
                net(hidden, self.prev_rewards, a_size, num_units)

            # Only the worker network needs ops for loss functions
            # and gradient updating.
            if scope != 'global':
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],
                                                 dtype=tf.float32)

                self.resp_outputs = \
                    tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(
                    tf.square(self.target_v -
                              tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(
                    self.policy * tf.log(self.policy + 1e-7))
                self.policy_loss = -tf.reduce_sum(
                    tf.log(self.resp_outputs + 1e-7)*self.advantages)
                self.loss = 0.5 * self.value_loss +\
                    self.policy_loss -\
                    self.entropy * 0.05

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms =\
                    tf.clip_by_global_norm(self.gradients, 999.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(
                    zip(grads, global_vars))


class Worker():
    def __init__(self, game, name, a_size, state_size, trainer,
                 model_path, global_epss, data_path, num_units, network):
        self.name = "worker_" + str(name)
        self.number = name
        self.folder = data_path + '/trains/train_' + str(self.number)
        self.model_path = model_path
        self.trainer = trainer
        self.global_epss = global_epss
        self.increment = self.global_epss.assign_add(1)
        self.network = network
        self.eps_rewards = []
        self.eps_mean_values = []

        self.summary_writer = tf.summary.FileWriter(self.folder)

        # Create the local copy of the network and the tensorflow op
        # to copy global parameters to local network
        self.local_AC = AC_Network(a_size, state_size, self.name, trainer,
                                   num_units, network)
        self.update_local_ops = ut.update_target_graph('global', self.name)
        self.env = game

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        states = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]

        prev_rewards = [0] + rewards[:-1].tolist()
        prev_actions = [0] + actions[:-1].tolist()
        values = rollout[:, 3]

        self.pr = prev_rewards
        self.pa = prev_actions
        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = ut.discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards +\
            gamma * self.value_plus[1:] -\
            self.value_plus[:-1]
        advantages = ut.discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.st_init
        if self.network == 'lstm':
            feed_dict = {self.local_AC.target_v: discounted_rewards,
                         self.local_AC.state: np.stack(states, axis=0),
                         self.local_AC.prev_rewards: np.vstack(prev_rewards),
                         self.local_AC.prev_actions: prev_actions,
                         self.local_AC.actions: actions,
                         self.local_AC.advantages: advantages,
                         self.local_AC.state_in[0]: rnn_state[0],
                         self.local_AC.state_in[1]: rnn_state[1]}
        elif (self.network == 'relu') or\
             (self.network == 'gru') or\
             (self.network == 'ugru'):
            feed_dict = {self.local_AC.target_v: discounted_rewards,
                         self.local_AC.st: np.stack(states, axis=0),
                         self.local_AC.prev_rewards: np.vstack(prev_rewards),
                         self.local_AC.prev_actions: prev_actions,
                         self.local_AC.actions: actions,
                         self.local_AC.advantages: advantages,
                         self.local_AC.st_in: rnn_state}

        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        aux = len(rollout)
        return v_l / aux, p_l / aux, e_l / aux, g_n, v_n

    def work(self, gamma, sess, coord, saver, train, exp_dur):
        eps_count = sess.run(self.global_epss)
        num_eps_tr_stats = int(1000/self.env.upd_net)
        num_epss_end = int(exp_dur/self.env.upd_net)
        num_epss_save_model = int(5000/self.env.upd_net)
        total_steps = 0
        print("Starting worker " + str(self.number))
        # get first state
        s = self.env.new_trial()
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                eps_buffer = []
                eps_values = []
                eps_reward = 0
                eps_step_count = 0
                d = False
                r = 0
                a = 0
                rnn_state = self.local_AC.st_init
                while not d:
                    if self.network == 'lstm':
                        feed_dict = {
                                    self.local_AC.state: [s],
                                    self.local_AC.prev_rewards: [[r]],
                                    self.local_AC.prev_actions: [a],
                                    self.local_AC.state_in[0]: rnn_state[0],
                                    self.local_AC.state_in[1]: rnn_state[1]}
                    elif (self.network == 'relu') or\
                         (self.network == 'gru') or\
                         (self.network == 'ugru'):
                        feed_dict = {
                                    self.local_AC.st: [s],
                                    self.local_AC.prev_rewards: [[r]],
                                    self.local_AC.prev_actions: [a],
                                    self.local_AC.st_in: rnn_state}

                    # Take an action using probs from policy network output
                    a_dist, v, rnn_state_new = sess.run(
                                                        [self.local_AC.policy,
                                                         self.local_AC.value,
                                                         self.local_AC.st_out],
                                                        feed_dict=feed_dict)

                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)
                    rnn_state = rnn_state_new
                    # new_state, reward, update_net, new_trial
                    s1, r, d, nt = self.env.step(a)
                    # save samples for training the network later
                    eps_buffer.append([s, a, r, v[0, 0]])
                    eps_values.append(v[0, 0])
                    eps_reward += r
                    total_steps += 1
                    eps_step_count += 1
                    s = s1

                self.eps_rewards.append(eps_reward)
                self.eps_mean_values.append(np.mean(eps_values))

                # Update the network using the experience buffer
                # at the end of the episode
                if len(eps_buffer) != 0 and train:
                    v_l, p_l, e_l, g_n, v_n = \
                        self.train(eps_buffer, sess, gamma, 0.0)

                # Periodically save model parameters and summary statistics.
                if eps_count % num_eps_tr_stats == 0 and eps_count != 0:
                    if eps_count % num_epss_save_model == 0 and\
                       self.name == 'worker_0' and\
                       train and\
                       len(self.eps_rewards) != 0:
                        saver.save(sess, self.model_path +
                                   '/model-' + str(eps_count) + '.cptk')
                    mean_reward = np.mean(self.eps_rewards[-10:])
                    mean_value = np.mean(self.eps_mean_values[-10:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward',
                                      simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Value',
                                      simple_value=float(mean_value))

                    performance_aux = np.vstack(np.array(self.env.perf_mat))

                    for ind_crr in range(performance_aux.shape[1]):
                        mean_performance = np.mean(performance_aux[:, ind_crr])
                        summary.value.add(tag='Perf/Perf_' + str(ind_crr),
                                          simple_value=float(mean_performance))

                    if train:
                        summary.value.add(tag='Losses/Value Loss',
                                          simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss',
                                          simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy',
                                          simple_value=float(e_l))
                        summary.value.add(tag='Losses/Grad Norm',
                                          simple_value=float(g_n))
                        summary.value.add(tag='Losses/Var Norm',
                                          simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, eps_count)

                    self.summary_writer.flush()

                if self.name == 'worker_0':
                    sess.run(self.increment)

                eps_count += 1
                if eps_count > num_epss_end:
                    break