import numpy as np


class data():
    def __init__(self, folder=''):
        # point by point parameter mats saved for some trials
        self.states_point = []
        self.net_state_point = []
        self.rewards_point = []
        self.done_point = []
        self.actions_point = []
        self.corrects_point = []
        self.new_trial_point = []
        self.trials_point = []
        self.stims_conf_point = []
        # where to save the trials data
        self.folder = folder

    def reset(self):
        """
        reset all mats
        """
        # reset parameters mat
        self.states_point = []
        self.net_state_point = []
        self.rewards_point = []
        self.done_point = []
        self.actions_point = []
        self.corrects_point = []
        self.new_trial_point = []
        self.stims_conf_point = []
        self.trials_point = []

    def update(self, new_state=[], net_state=[], reward=None, update_net=None,
               action=None, correct=[], new_trial=None, num_trials=None,
               stim_conf=[]):
        """
        append available info
        """
        if len(new_state) != 0:
            self.states_point.append(new_state)
        if len(net_state) != 0:
            self.net_state_point.append(net_state)
        if reward is not None:
            self.rewards_point.append(reward)
        if update_net is not None:
            self.done_point.append(update_net)  # 0 by construction
        if action is not None:
            self.actions_point.append(action)
        if len(correct) != 0:
            self.corrects_point.append(correct)
        if new_trial is not None:
            self.new_trial_point.append(new_trial)  # 0 by construction
        if num_trials is not None:
            self.trials_point.append(num_trials)
        if len(stim_conf) != 0:
            self.stims_conf_point.append(stim_conf)

    def save(self, num_trials):
        """
        save data
        """
        data = {'states': self.states_point, 'net_state': self.net_state_point,
                'rewards': self.rewards_point, 'done_flags': self.done_point,
                'actions': self.actions_point, 'corrects': self.corrects_point,
                'new_trial_flags': self.new_trial_point,
                'trials_saved': self.trials_point,
                'stims_conf': self.stims_conf_point}
        np.savez(self.folder + '/all_points_' + str(num_trials) + '.npz',
                 **data)