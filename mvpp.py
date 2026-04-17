import itertools
import math
import os.path
import re
import sys
import time
import torch

from clingo.control import Control
import numpy as np


class MVPP(object):
    def __init__(self, program, k=1, eps=0.000001):
        self.k = k
        self.eps = eps

        # each element in self.parameters is a list of probabilities
        self.parameters = []
        # each element in self.learnable is a list of Boolean values
        self.learnable = []
        # self.asp is the ASP part of the LPMLN program
        self.asp = ""
        # self.pi_prime is the ASP program \Pi' defined for the semantics
        self.pi_prime = ""
        # self.remain_probs is a list of probs, each denotes a remaining prob given those non-learnable probs
        self.remain_probs = []

        # self.pc: dictionary of concepts
        self.pc, self.parameters, self.learnable, self.asp, self.pi_prime, self.remain_probs = self.parse(program)
        self.normalize_probs()

    def parse(self, program):
        pc = {}
        parameters = []
        learnable = []
        asp = ""
        pi_prime = ""
        remain_probs = []

        # if program is a file
        if os.path.isfile(program):
            with open(program, 'r') as program:
                lines = program.readlines()
        # if program is a string containing all rules of an LPMLN program
        elif type(program) is str and re.sub(r'\n%[^\n]*', '\n', program).strip().endswith(('.', ']')):
            lines = program.split('\n')
        else:
            print("Error! The MVPP program {} is not valid.".format(program))
            sys.exit()

        for line in lines:
            if re.match(r"@?[0-9]\.?[0-9]*(?:e-[0-9]+)?\s.*;.*", line):
                list_of_atoms = []
                list_of_probs = []
                list_of_bools = []
                choices = line.strip()[:-1].split(";")
                for choice in choices:
                    prob, atom = choice.strip().split(" ", maxsplit=1)
                    # Note that we remove all spaces in atom since clingo output does not contain space in atom
                    list_of_atoms.append(atom.replace(" ", ""))
                    if prob.startswith("@"):
                        list_of_probs.append(float(prob[1:]))
                        list_of_bools.append(True)
                    else:
                        list_of_probs.append(float(prob))
                        list_of_bools.append(False)

                    # Fill in pc so that each concept entry contains a list of all the values it can take
                    split_list = atom.split('(')
                    concept_name = split_list[0]
                    concept_args = split_list[1].replace(" ", "")
                    # Create a unique id for each concept
                    concept_id = f"{concept_name}/{len(concept_args.split(','))}:{concept_args.split(',')[0]},{concept_args.split(',')[1]}"
                    concept_value = concept_args.split(',')[-1].split(')')[0].strip()
                    if concept_id not in pc:
                        pc[concept_id] = [concept_value]
                    else:
                        pc[concept_id].append(concept_value)

                parameters.append(list_of_probs)
                learnable.append(list_of_bools)
                pi_prime += "1{" + "; ".join(list_of_atoms) + "}1.\n"
            else:
                asp += (line.strip() + "\n")

        pi_prime += asp

        for ruleIdx, list_of_bools in enumerate(learnable):
            remain_prob = 1
            for atomIdx, b in enumerate(list_of_bools):
                if b == False:
                    remain_prob -= parameters[ruleIdx][atomIdx]
            remain_probs.append(remain_prob)
        return pc, parameters, learnable, asp, pi_prime, remain_probs

    def normalize_probs(self):
        for ruleIdx, list_of_bools in enumerate(self.learnable):
            summation = 0
            # 1st, we turn each probability into [0+eps,1-eps]
            for atomIdx, b in enumerate(list_of_bools):
                if b == True:
                    if self.parameters[ruleIdx][atomIdx] >= 1:
                        self.parameters[ruleIdx][atomIdx] = 1 - self.eps
                    elif self.parameters[ruleIdx][atomIdx] <= 0:
                        self.parameters[ruleIdx][atomIdx] = self.eps

            # 2nd, we normalize the probabilities
            for atomIdx, b in enumerate(list_of_bools):
                if b == True:
                    summation += self.parameters[ruleIdx][atomIdx]
            for atomIdx, b in enumerate(list_of_bools):
                if b == True:
                    self.parameters[ruleIdx][atomIdx] = self.parameters[ruleIdx][atomIdx] / summation * \
                                                        self.remain_probs[ruleIdx]
        return True

    def prob_of_interpretation(self, models):
        net_confs = torch.stack(self.parameters)
        concept_indices = torch.arange(len(net_confs)).repeat(len(models), 1)
        probs = net_confs[concept_indices, models].prod(1)
        return probs

    # We assume obs is a string containing a valid Clingo program,
    # and each obs is written in constraint form
    def find_one_SM_under_obs(self, obs):
        program = self.pi_prime + obs
        clingo_control = Control(["--warn=none"])
        models = []
        clingo_control.add("base", [], program)
        clingo_control.ground([("base", [])])
        clingo_control.solve(on_model=lambda model: models.append(model.symbols(atoms=True)))
        models = [[str(atom) for atom in model] for model in models]
        return models

    # We assume obs is a string containing a valid Clingo program,
    # and each obs is written in constraint form
    def find_all_SM_under_obs(self, obs):
        program = self.pi_prime + obs
        clingo_control = Control(["0", "--warn=none"])
        models = []
        try:
            clingo_control.add("base", [], program)
        except:
            print("\nPi': \n{}".format(program))
        clingo_control.ground([("base", [])])
        clingo_control.solve(on_model = lambda model: models.append(model.symbols(atoms=True)))
        models = [[str(atom) for atom in model] for model in models]
        return models

    def model_to_network_preds(self, m):
        # Extract network predictions from stable model
        m = str(m).split(' ')
        network_preds = [0 for _ in m]
        for atom in m:
            split_list = atom.split('(')
            concept_name = split_list[0]
            concept_args = split_list[1].replace(" ", "")
            concept_id = f"{concept_name}/{len(concept_args.split(','))}:{concept_args.split(',')[0]},{concept_args.split(',')[1]}"
            concept_value = concept_args.split(',')[-1].split(')')[0].strip()
            # Maintain order of concepts from pc
            network_preds[list(self.pc).index(concept_id)] = self.pc[concept_id].index(concept_value)
        return network_preds

    def find_k_SM_under_obs(self, obs, k=3, opt=False):
        # Create show statements so clingo only outputs neural concepts
        show_list = []
        for pc in self.pc:
            show_item = f"#show {pc.split(':')[0]}."
            if show_item not in show_list:
                show_list.append(show_item)
        show_string = ' '.join(show_list)

        program = self.pi_prime + obs + show_string
        if opt:
            clingo_control = Control(["--warn=none", '--opt-mode=optN', str(k)])
            def model_fun(model):
                if model.optimality_proven:
                    models.append(self.model_to_network_preds(model))
        else:
            clingo_control = Control(["--warn=none", str(k)])
            def model_fun(model):
                models.append(self.model_to_network_preds(model))

        models = []
        try:
            clingo_control.add("base", [], program)
        except:
            print("\nPi': \n{}".format(program))
        clingo_control.ground([("base", [])])
        clingo_control.solve(on_model=model_fun)
        return torch.IntTensor(models)

    # there might be some duplications in SMs when optimization option is used
    # and the duplications are removed by this method
    def remove_duplicate_SM(self, models):
        models.sort()
        return list(models for models, _ in itertools.groupby(models))

    # Note that the MVPP program cannot contain weak constraints
    def find_all_most_probable_SM_under_obs_noWC(self, obs):
        """Return a list of stable models, each is a list of strings
        @param obs: a string of a set of constraints/facts
        """
        program = self.pi_prime + obs + '\n'
        # for each probabilistic rule with n atoms, add n weak constraints
        for ruleIdx, atom in enumerate(self.pc):
            for valueIdx, value in enumerate(self.pc[atom]):
                if self.parameters[ruleIdx][valueIdx] < 0.00674:
                    penalty = -1000 * -5
                else:
                    penalty = int(-1000 * math.log(self.parameters[ruleIdx][valueIdx]))
                atom_name = f"{atom.split('/')[0]}({atom.split(':')[1]},{value})"
                program += ':~ {}. [{}, {}, {}]\n'.format(atom_name, penalty, ruleIdx, valueIdx)

        clingo_control = Control(['--warn=none', '--opt-mode=optN', '0', '-t', '8'])
        models = []
        clingo_control.add("base", [], program)
        clingo_control.ground([("base", [])])
        clingo_control.solve(on_model = lambda model: models.append(model.symbols(atoms=True)) if model.optimality_proven else None)
        models = [[str(atom) for atom in model] for model in models]
        return self.remove_duplicate_SM(models)

    def find_one_most_probable_SM_under_obs_noWC(self, obs=''):
        """Return a list of a single stable model, which is a list of strings
        @param obs: a string of a set of constraints/facts
        """
        program = self.pi_prime + obs + '\n'
        # for each probabilistic rule with n atoms, add n weak constraints
        for ruleIdx, atom in enumerate(self.pc):
            for valueIdx, value in enumerate(self.pc[atom]):
                if self.parameters[ruleIdx][valueIdx] < 0.00674:
                    penalty = -1000 * -5
                else:
                    penalty = int(-1000 * math.log(self.parameters[ruleIdx][valueIdx]))
                atom_name = f"{atom.split('/')[0]}({atom.split(':')[1]},{value})"
                program += ':~ {}. [{}, {}, {}]\n'.format(atom_name, penalty, ruleIdx, valueIdx)

        clingo_control = Control(['--warn=none', '-t', '8'])
        models = []
        clingo_control.add("base", [], program)
        clingo_control.ground([("base", [])])
        clingo_control.solve(on_model = lambda model: models.append(model.symbols(atoms=True)))
        models = [[str(atom) for atom in model] for model in models]
        return [models[-1]]

    def find_all_opt_SM_under_obs_WC(self, obs):
        """ Return a list of stable models, each is a list of strings
        @param obs: a string of a set of constraints/facts

        """
        program = self.pi_prime + obs
        clingo_control = Control(['--warn=none', '--opt-mode=optN', '0'])
        models = []
        try:
            clingo_control.add("base", [], program)
        except:
            print('\nSyntax Error in Program: Pi\': \n{}'.format(program))
            sys.exit()
        clingo_control.ground([("base", [])])
        clingo_control.solve(on_model = lambda model: models.append(model.symbols(atoms=True)) if model.optimality_proven else None)
        models = [[str(atom) for atom in model] for model in models]
        return self.remove_duplicate_SM(models)

    # compute P(O)
    def inference_obs_exact(self, obs):
        prob = 0
        models = self.find_all_SM_under_obs(obs)
        for I in models:
            prob += self.prob_of_interpretation(I)
        return prob

    def gradient(self, ruleIdx, atomIdx, obs):
        net_confs = torch.stack(self.parameters)
        models = self.find_k_SM_under_obs(obs, k=3)
        probs = self.prob_of_interpretation(models)
        in_model = models[:,ruleIdx] == net_confs[ruleIdx][atomIdx]
        p_obs_i = sum(probs / net_confs[ruleIdx][atomIdx] * in_model)
        p_obs_j = sum(probs / net_confs[ruleIdx, models[:,ruleIdx]])
        p_obs = sum(probs)
        gradient = (p_obs_i - p_obs_j) / p_obs
        return gradient.item()

    def mvppLearnRule(self, models, probs, num_out):
        net_confs = torch.stack(self.parameters)
        denominator = sum(probs)
        if denominator == 0:
            return torch.zeros(len(self.learnable), num_out)

        concept_indices = torch.arange(len(net_confs)).repeat(len(models), 1)
        concept_probs = net_confs[concept_indices, models]

        weighted_probs = probs.view(len(probs),1) / concept_probs
        nums_out = torch.arange(num_out)
        multiplication_factor = (models.unsqueeze(-1) == nums_out).float() * 2 - 1
        gradients = (weighted_probs.unsqueeze(-1) * multiplication_factor).sum(0) / denominator

        return gradients

    def mvppLearn(self, models):
        probs = self.prob_of_interpretation(models)
        if len(models) != 0:
            return self.mvppLearnRule(models, probs, len(self.learnable[0]))
        else:
            return [[0.0 for item in l] for l in self.parameters]

    # gradients are stored in numpy array instead of list
    # obs is a string
    def gradients_one_obs(self, obs, opt=False):
        """Return an np-array denoting the gradients
        @param obs: a string for observation
        @param opt: a Boolean denoting whether we use optimal stable models instead of stable models
        """
        models = self.find_k_SM_under_obs(obs, k=0, opt=opt)

        return self.mvppLearn(models)

    # gradients are stored in numpy array instead of list
    def gradients_multi_obs(self, list_of_obs):
        gradients = [[0.0 for item in l] for l in self.parameters]
        for obs in list_of_obs:
            gradients = [[c+d for c,d in zip(i,j)] for i,j in zip(gradients,self.gradients_one_obs(obs))]
        return gradients

    # list_of_obs is either a list of strings or a file containing observations separated by "#evidence"
    def learn_exact(self, list_of_obs, lr=0.01, thres=0.0001, max_iter=None):
        # if list_of_obs is an evidence file, we need to first turn it into a list of strings
        if type(list_of_obs) is str and os.path.isfile(list_of_obs):
            with open(list_of_obs, 'r') as f:
                list_of_obs = f.read().strip().strip("#evidence").split("#evidence")
        print("Start learning by exact computation with {} observations...\n\nInitial parameters: {}".format(len(list_of_obs), self.parameters))
        time_init = time.time()
        check_continue = True
        iteration = 1
        while check_continue:
            old_parameters = self.parameters
            print("\n#### Iteration {} ####\n".format(iteration))
            check_continue = False
            dif = [[lr*grad for grad in l] for l in self.gradients_multi_obs(list_of_obs)]

            for ruleIdx, list_of_bools in enumerate(self.learnable):
            # 1st, we turn each gradient into [-0.2, 0.2]
                for atomIdx, b in enumerate(list_of_bools):
                    if b == True:
                        if dif[ruleIdx][atomIdx] > 0.2 :
                            dif[ruleIdx][atomIdx] = 0.2
                        elif dif[ruleIdx][atomIdx] < -0.2:
                            dif[ruleIdx][atomIdx] = -0.2

            self.parameters = [[c+d for c,d in zip(i,j)] for i,j in zip(dif,self.parameters)]
            self.normalize_probs()

            # we termintate if the change of the parameters is lower than thres
            dif = [[abs(c-d) for c,d in zip(i,j)] for i,j in zip(old_parameters,self.parameters)]
            print("After {} seconds of training (in total)".format(time.time()-time_init))
            print("Current parameters: {}".format(self.parameters))
            maxdif = max([max(l) for l in dif])
            print("Max change on probabilities: {}".format(maxdif))
            iteration += 1
            if maxdif > thres:
                check_continue = True
            if max_iter is not None:
                if iteration > max_iter:
                    check_continue = False
        print("\nFinal parameters: {}".format(self.parameters))

    ##############################
    ####### Sampling Method ######
    ##############################

    # it will generate k sample stable models for a k-coherent program under a specific total choice
    def k_sample(self):
        asp_with_facts = self.asp
        clingo_control = Control(["0", "--warn=none"])
        models = []
        for ruleIdx,list_of_atoms in enumerate(self.pc):
            tmp = np.random.choice(list_of_atoms, 1, p=self.parameters[ruleIdx])
            asp_with_facts += tmp[0]+".\n"
        clingo_control.add("base", [], asp_with_facts)
        clingo_control.ground([("base", [])])
        result = clingo_control.solve(on_model = lambda model: models.append(model.symbols(shown=True)))
        models = [[str(atom) for atom in model] for model in models]
        return models

    # it will generate k*num sample stable models
    def sample(self, num=1):
        models = []
        for i in range(num):
            models = models + self.k_sample()
        return models

    # it will generate at least num of samples that satisfy obs
    def sample_obs(self, obs, num=50):
        count = 0
        models = []
        while count < num:
            asp_with_facts = self.asp
            asp_with_facts += obs
            clingo_control = Control(["0", "--warn=none"])
            models_tmp = []
            for ruleIdx,list_of_atoms in enumerate(self.pc):
                tmp = np.random.choice(list_of_atoms, 1, p=self.parameters[ruleIdx])
                asp_with_facts += tmp[0]+".\n"
            clingo_control.add("base", [], asp_with_facts)
            clingo_control.ground([("base", [])])
            result = clingo_control.solve(on_model = lambda model: models_tmp.append(model.symbols(shown=True)))
            if str(result) == "SAT":
                models_tmp = [[str(atom) for atom in model] for model in models_tmp]
                count += len(models_tmp)
                models = models + models_tmp
            elif str(result) == "UNSAT":
                pass
            else:
                print("Error! The result of a clingo call is not SAT nor UNSAT!")
        return models

    # it will generate at least num of samples that satisfy obs
    def sample_obs2(self, obs, num=50):
        count = 0
        models = []
        candidate_sm = []
        # we first find out all stable models that satisfy obs
        program = self.pi_prime + obs
        clingo_control = Control(['0', '--warn=none'])
        clingo_control.add('base', [], program)
        clingo_control.ground([('base', [])])
        clingo_control.solve(None, lambda model: candidate_sm.append(model.symbols(shown=True)))
        candidate_sm = [[str(atom) for atom in model] for model in candidate_sm]
        probs = [self.prob_of_interpretation(model) for model in candidate_sm]

        while count < num:
            asp_with_facts = self.pi_prime
            asp_with_facts += obs
            clingo_control = Control(["0", "--warn=none"])
            models_tmp = []
            for ruleIdx,list_of_atoms in enumerate(self.pc):
                tmp = np.random.choice(list_of_atoms, 1, p=self.parameters[ruleIdx])
                asp_with_facts += tmp[0]+".\n"
            clingo_control.add("base", [], asp_with_facts)
            clingo_control.ground([("base", [])])
            result = clingo_control.solve(on_model = lambda model: models_tmp.append(model.symbols(shown=True)))
            if str(result) == "SAT":
                models_tmp = [[str(atom) for atom in model] for model in models_tmp]
                count += len(models_tmp)
                models = models + models_tmp
            elif str(result) == "UNSAT":
                pass
            else:
                print("Error! The result of a clingo call is not SAT nor UNSAT!")
        return models

    # we compute the gradients (numpy array) w.r.t. all probs in the ruleIdx-th rule
    # given models that satisfy obs
    def gradient_given_models(self, ruleIdx, models):
        arity = len(self.parameters[ruleIdx])
        n_i = torch.bincount(models[:, ruleIdx], minlength=arity) / self.parameters[ruleIdx]
        summation = torch.sum(n_i * -1)
        gradients = (summation.repeat_interleave(arity) + 2 * n_i) / len(models)
        return gradients


    # gradients are stored in numpy array instead of list
    # obs is a string
    def gradients_one_obs_by_sampling(self, obs, num=50):
        gradients = np.array([[0.0 for item in l] for l in self.parameters])
        # 1st, we generate at least num of stable models that satisfy obs
        models = self.sample_obs(obs=obs, num=num)

        # 2nd, we compute the gradients w.r.t. the probs in each rule
        for ruleIdx,list_of_bools in enumerate(self.learnable):
            gradients[ruleIdx] = self.gradient_given_models(ruleIdx, models)
            for atomIdx, b in enumerate(list_of_bools):
                if b == False:
                    gradients[ruleIdx][atomIdx] = 0
        return gradients

    # we compute the gradients (numpy array) w.r.t. all probs given list_of_obs
    def gradients_multi_obs_by_sampling(self, list_of_obs, num=50):
        gradients = np.array([[0.0 for item in l] for l in self.parameters])

        # we itereate over all obs
        for obs in list_of_obs:
            # 1st, we generate at least num of stable models that satisfy obs
            models = self.sample_obs(obs=obs, num=num)

            # 2nd, we accumulate the gradients w.r.t. the probs in each rule
            for ruleIdx,list_of_bools in enumerate(self.learnable):
                gradients[ruleIdx] += self.gradient_given_models(ruleIdx, models)
                for atomIdx, b in enumerate(list_of_bools):
                    if b == False:
                        gradients[ruleIdx][atomIdx] = 0
        return gradients

    # we compute the gradients (numpy array) w.r.t. all probs given list_of_obs
    # while we generate at least one sample without considering probability distribution
    def gradients_multi_obs_by_one_sample(self, list_of_obs):
        gradients = np.array([[0.0 for item in l] for l in self.parameters])

        # we itereate over all obs
        for obs in list_of_obs:
            # 1st, we generate one stable model that satisfy obs
            models = self.find_one_SM_under_obs(obs=obs)

            # 2nd, we accumulate the gradients w.r.t. the probs in each rule
            for ruleIdx,list_of_bools in enumerate(self.learnable):
                gradients[ruleIdx] += self.gradient_given_models(ruleIdx, models)
                for atomIdx, b in enumerate(list_of_bools):
                    if b == False:
                        gradients[ruleIdx][atomIdx] = 0
        return gradients

    # list_of_obs is either a list of strings or a file containing observations separated by "#evidence"
    def learn_by_sampling(self, list_of_obs, num_of_samples=50, lr=0.01, thres=0.0001, max_iter=None, num_pretrain=1):
        # Step 0: Evidence Preprocessing: if list_of_obs is an evidence file, 
        # we need to first turn it into a list of strings
        if type(list_of_obs) is str and os.path.isfile(list_of_obs):
            with open(list_of_obs, 'r') as f:
                list_of_obs = f.read().strip().strip("#evidence").split("#evidence")

        print("Start learning by sampling with {} observations...\n\nInitial parameters: {}".format(len(list_of_obs), self.parameters))
        time_init = time.time()

        # Step 1: Parameter Pre-training: we pretrain the parameters 
        # so that it's easier to generate sample stable models
        assert type(num_pretrain) is int
        if num_pretrain >= 1:
            print("\n#######################################################\nParameter Pre-training for {} iterations...\n#######################################################".format(num_pretrain))
            for iteration in range(num_pretrain):
                print("\n#### Iteration {} for Pre-Training ####\nGenerating 1 stable model for each observation...\n".format(iteration+1))
                dif = lr * self.gradients_multi_obs_by_one_sample(list_of_obs)
                self.parameters = (np.array(self.parameters) + dif).tolist()
                self.normalize_probs()

                print("After {} seconds of training (in total)".format(time.time()-time_init))
                print("Current parameters: {}".format(self.parameters))

        # Step 2: Parameter Training: we train the parameters using "list_of_obs until"
        # (i) the max change on probabilities is lower than "thres", or
        # (ii) the number of iterations is more than "max_iter"
        print("\n#######################################################\nParameter Training for {} iterations or until converge...\n#######################################################".format(max_iter))
        check_continue = True
        iteration = 1
        while check_continue:
            print("\n#### Iteration {} ####".format(iteration))
            old_parameters = np.array(self.parameters)            
            check_continue = False

            print("Generating {} stable model(s) for each observation...\n".format(num_of_samples))
            dif = lr * self.gradients_multi_obs_by_sampling(list_of_obs, num=num_of_samples)

            self.parameters = (np.array(self.parameters) + dif).tolist()
            self.normalize_probs()
            
            print("After {} seconds of training (in total)".format(time.time()-time_init))
            print("Current parameters: {}".format(self.parameters))

            # we termintate if the change of the parameters is lower than thres
            dif = np.array(self.parameters) - old_parameters
            dif = abs(max(dif.min(), dif.max(), key=abs))
            print("Max change on probabilities: {}".format(dif))

            iteration += 1
            if dif > thres:
                check_continue = True
            if max_iter is not None:
                if iteration > max_iter:
                    check_continue = False

        print("\nFinal parameters: {}".format(self.parameters))
