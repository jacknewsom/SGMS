from module.searchspace.hyperparameters.mnist_hyperparameter_space import MNISTHyperparameterSpace
from module.controller.base_controller import BaseController
from fe_supermodel import FeatureExtractorSupermodel
from module.utils.torch_modules import Policy
from torch.distributions import Categorical
from collections import OrderedDict
import numpy as np
import torch
import os

class FeatureExtractorController(BaseController):
    def __init__(
        self, 
        evaluator,
        N,
        latent_space_size,
        latent_space_channels,
        input_space_size=torch.LongTensor([128,128,128]),
        archspace_weight_directory='fesupermodel_weights/',
        archspace_epochs=15,
        use_baseline=True,
        reward_map_fn=None,
        device=None
        ):
        super(FeatureExtractorController, self).__init__()

        self.archspace = FeatureExtractorSupermodel(
            evaluator,
            N,
            latent_space_size,
            latent_space_channels,
            input_space_size,
            archspace_weight_directory,
            archspace_epochs,
            device)

        optimizers = [torch.optim.Adam, torch.optim.AdamW]
        lrs = [0.001, 0.005, 0.0001]
        self.hpspace = MNISTHyperparameterSpace(optimizers, lrs)

        self.converged = False

        self.device = device
        self.use_baseline = use_baseline
        self.reward_map_fn = reward_map_fn

        self.policies = OrderedDict()
        self.policies['archspace'] = OrderedDict()
        self.policies['hpspace'] = OrderedDict()
        self.policies['hpspace']['optimizers'] = OrderedDict()
        self.policies['hpspace']['lrs'] = OrderedDict()

        parameters = []
        for key in self.archspace.search_dimensions:
            policy = Policy(len(self.archspace.search_dimensions[key]), device)
            self.policies['archspace'][key] = policy
            parameters += [policy.parameters()]

        self.policies['hpspace']['optimizers'] = Policy(len(optimizers), device)
        self.policies['hpspace']['lrs'] = Policy(len(lrs), device)

        parameters += [self.policies['hpspace']['optimizers'].parameters()]
        parameters += [self.policies['hpspace']['lrs'].parameters()]
        parameters = [{'params': p} for p in parameters]

        self.optimizer = torch.optim.AdamW(parameters)

    def has_converged(self):
        return self.converged

    def sample(self):
        archspace_actions = []
        for key in self.policies['archspace']:
            policy = self.policies['archspace'][key]
            action = Categorical(policy()).sample()
            archspace_actions.append(action)

        optim = Categorical(self.policies['hpspace']['optimizers']()).sample()
        lr = Categorical(self.policies['hpspace']['lrs']()).sample()
        hp_actions = [optim, lr]

        return archspace_actions, hp_actions

    def argmax(self):
        archspace_actions = []
        for key in self.policies['archspace']:
            policy = self.policies['archspace'][key]
            action = torch.argmax(policy.params)
            archspace_actions.append(action)

        optim = torch.argmax(self.policies['hpspace']['optimizers'].params)
        lr = torch.argmax(self.policies['hpspace']['lrs'].params)
        hp_actions = [optim, lr]

        return archspace_actions, hp_actions

    def update(self, rollouts):
        rewards = [r[2] for r in rollouts]

        # exponentiate rewards 
        if self.reward_map_fn:
            rewards = [self.reward_map_fn(r) for r in rewards]

        # calculate rewards using average reward as baseline
        if self.use_baseline and len(rollouts) > 1:
            avg_reward = np.mean(rewards)
            rewards = [r-avg_reward for r in rewards]

        # calculate log probabilities for each time step
        log_prob = []
        for t in rollouts:
            _log_prob = []
            layerwise_actions, hp_actions = t[:2]
            for layer_action, key in zip(layerwise_actions, self.policies['archspace']):
                layer_policy = self.policies['archspace'][key]
                _log_prob.append(Categorical(layer_policy()).log_prob(layer_action))
            for action, key in zip(hp_actions, self.policies['hpspace']):
                policy = self.policies['hpspace'][key]
                _log_prob.append(Categorical(policy()).log_prob(action))
            log_prob.append(torch.stack(_log_prob).sum())

        self.optimizer.zero_grad()
        loss = [-r * lp for r, lp in zip(rewards, log_prob)]
        loss = torch.stack(loss).sum()
        loss.backward()
        self.optimizer.step()

    def save_policies(self, directory='fecontroller_weights/'):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        if directory[-1] != '/':
            directory += '/'
        for k in self.policies['archspace']:
            torch.save(self.policies['archspace'][k].state_dict(), directory + 'archspace_' + str(k))
        for k in self.policies['hpspace']:
            torch.save(self.policies['hpspace'][k].state_dict(), directory + 'hpspace_' + k)

    def load_policies(self, directory='fecontroller_weights/'):
        if not os.path.isdir(directory):
            raise ValueError('Directory %s does not exist' % directory)

        if directory[-1] != '/':
            directory += '/'
        for k in self.policies['archspace']:
            _ = torch.load(directory + 'archspace_' + str(k))
            self.policies['archspace'][k].load_state_dict(_)
        for k in self.policies['hpspace']:
            _ = torch.load(directory + 'hpspace_' + k)
            self.policies['hpspace'][k].load_state_dict(_)