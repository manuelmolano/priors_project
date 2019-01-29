import threading
import multiprocessing
import os
import utils
import numpy as np
import tensorflow as tf
import A3C_agent as ag
import task


def main_priors(load_model=False, train=True, gamma=.8, up_net=5,
                trial_dur=10, rep_prob=(0.2, 0.8), exp_dur=1000,
                rewards=(-0.1, 0.0, 1.0, -1.0), block_dur=200,
                num_units=32, stim_ev=.3, network='ugru',
                learning_rate=1e-3, instance=0, main_folder=''):
    a_size = 3  # number of actions
    state_size = a_size  # number of inputs

    data_path = utils.folder_name(gamma=gamma, up_net=up_net,
                                  trial_dur=trial_dur, rep_prob=rep_prob,
                                  exp_dur=exp_dur, rewards=rewards,
                                  block_dur=block_dur, num_units=num_units,
                                  stim_ev=stim_ev, network=network,
                                  learning_rate=learning_rate,
                                  instance=instance, main_folder=main_folder)

    data = {'trial_dur': trial_dur, 'rep_prob': rep_prob,
            'rewards': rewards, 'stim_ev': stim_ev,
            'block_dur': block_dur, 'gamma': gamma, 'num_units': num_units,
            'up_net': up_net, 'network': network}

    model_path = data_path + '/model_meta_context'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not train:
        data_path = data_path + '/test/'

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    np.savez(data_path + '/experiment_setup.npz', **data)

    tf.reset_default_graph()
    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32,
                                      name='global_episodes',
                                      trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        ag.AC_Network(a_size, state_size, 'global',
                      None, num_units, network)  # Generate global net
        # Set workers to number of available CPU threads
        num_workers = multiprocessing.cpu_count()
        workers = []
        # Create worker classes
        for i in range(num_workers):
            saving_path = data_path + '/trains/train_' + str(i)
            if not os.path.exists(saving_path):
                os.makedirs(saving_path)
            workers.append(ag.Worker(task.PriorsEnv(upd_net=up_net,
                                                    trial_dur=trial_dur,
                                                    rep_prob=rep_prob,
                                                    rewards=rewards,
                                                    block_dur=block_dur,
                                                    stim_ev=stim_ev,
                                                    folder=saving_path,
                                                    plot=(i == 0)),
                                     i, a_size, state_size,
                                     trainer, model_path, global_episodes,
                                     data_path, num_units, network))
        saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if load_model:
            print('Loading Model...')
            print(model_path)
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(gamma, sess, coord,
                                              saver, train, exp_dur)
            thread = threading.Thread(target=(worker_work))
            thread.start()
            worker_threads.append(thread)
        coord.join(worker_threads)


if __name__ == '__main__':
    # experiment duration
    exp_dur = 10**6
    # num steps per trial
    trial_dur = 10
    # rewards given for: stop fixating, keep fixating, correct, wrong
    rewards = (-0.1, 0.0, 1.0, -1.0)
    # number of trials per blocks
    block_dur = 200
    # stimulus evidence
    stim_ev = 0.5
    # prob. of repeating the stimuli in the positions of previous trial
    rep_prob = [0.2, 0.8]
    # discount factor
    gamma = 0.8
    # learning rate
    lr = 1e-3
    # num units in the network
    num_units = 32
    # trials to updated the network weights
    up_net = 5
    # network units
    net = 'ugru'
    # instance
    inst = 1
    # folder where data will be saved
    main_folder = '/home/molano/priors_project/priors/'
    main_priors(load_model=False, train=True, gamma=gamma, up_net=up_net,
                trial_dur=trial_dur, rep_prob=rep_prob, exp_dur=exp_dur,
                rewards=rewards, block_dur=block_dur,
                num_units=num_units, stim_ev=stim_ev, network=net,
                learning_rate=lr, instance=inst, main_folder=main_folder)