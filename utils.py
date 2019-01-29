from scipy.signal import lfilter
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import difflib


def update_target_graph(from_scope, to_scope):
    """
    Copies one set of variables to another.
    Used to set worker network parameters to those of global network.
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def discount(x, gamma):
    """
    Discounting function used to calculate discounted returns.
    """
    return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def norm_col_init(std=1.0):
    """
    Used to initialize weights for policy and value output layers
    """
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def look_for_folder(main_folder='priors/', exp=''):
    """
    looks for a given folder and returns it.
    If it cannot find it, returns possible candidates
    """
    data_path = ''
    possibilities = []
    for root, dirs, files in os.walk(main_folder):
        ind = root.rfind('/')
        possibilities.append(root[ind+1:])
        if root[ind+1:] == exp:
            data_path = root
            break

    if data_path == '':
        candidates = difflib.get_close_matches(exp, possibilities,
                                               n=1, cutoff=0.)
        print(exp + ' NOT FOUND IN ' + main_folder)
        if len(candidates) > 0:
            print('possible candidates:')
            print(candidates)

    return data_path


def list_str(l):
    """
    list to str
    """
    nice_string = str(l[0])
    for ind_el in range(1, len(l)):
        nice_string += '_'+str(l[ind_el])
    return nice_string


def num2str(num):
    """
    pass big number to thousands
    """
    return str(int(num/1000))+'K'


def rm_lines():
    ax = plt.gca()
    ax.clear()


def plot_trials_start(trials, minimo, maximo, num_steps, color='k'):
    trials = np.nonzero(trials)[0] - 0.5
    cond = np.logical_and(trials >= 0, trials <= num_steps)
    trials = trials[np.where(cond)]
    for ind_tr in range(len(trials)):
        plt.plot([trials[ind_tr], trials[ind_tr]], [minimo, maximo],
                 '--'+color, lw=1)
    plt.xlim(0-0.5, num_steps-0.5)


def folder_name(gamma=0.8, up_net=5, trial_dur=10,
                rep_prob=(.2, .8), exp_dur=10**6,
                rewards=(-0.1, 0.0, 1.0, -1.0),
                block_dur=200, num_units=32,
                stim_ev=0.5, network='ugru', learning_rate=10e-3,
                instance=0, main_folder=''):
    return main_folder + '/td_' + str(trial_dur) + '_rp_' +\
        str(list_str(rep_prob)) + '_r_' +\
        str(list_str(rewards)) + '_bd_' + str(block_dur) +\
        '_ev_' + str(stim_ev) + '_g_' + str(gamma) + '_lr_' +\
        str(learning_rate) + '_nu_' + str(num_units) + '_un_' + str(up_net) +\
        '_net_' + str(network) + '_ed_' + str(num2str(exp_dur)) +\
        '_' + str(instance) + '/'