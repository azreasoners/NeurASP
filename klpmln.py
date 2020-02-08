import os.path
import re
import numpy as np
import clingo
import math
import itertools
import time
import sys

class MVPP(object):
    def __init__(self, program, k=1, eps=0.0001):
        self.k = k
        self.eps = eps

        # each element in self.pc is a list of atoms (one list for one prob choice rule)
        self.pc = []
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

        self.pc, self.parameters, self.learnable, self.asp, self.pi_prime, self.remain_probs = self.parse(program)
        self.normalize_probs()

    def parse(self, program):
        pc = []
        parameters = []
        learnable = []
        asp = ""
        pi_prime = ""
        remain_probs = []

        lines = []
        # if program is a file
        if os.path.isfile(program):
            with open(program, 'r') as program:
                lines = program.readlines()
                # print("lines1: {}".format(lines))
        # if program is a string containing all rules of an LPMLN program
        elif type(program) is str and re.sub(r'\n%[^\n]*', '\n', program).strip().endswith(('.', ']')):
            lines = program.split('\n')
            # print("lines2: {}".format(lines))
        else:
            print("Error! The MVPP program {} is not valid.".format(program))
            sys.exit()

        for line in lines:
            if re.match(r".*[0-1]\.?[0-9]*\s.*;.*", line):
                list_of_atoms = []
                list_of_probs = []
                list_of_bools = []
                choices = line.strip()[:-1].split(";")
                for choice in choices:
                    # print(choice)
                    prob, atom = choice.strip().split(" ", maxsplit=1)
                    # Note that we remove all spaces in atom since clingo output does not contain space in atom
                    list_of_atoms.append(atom.replace(" ", ""))
                    if prob.startswith("@"):
                        list_of_probs.append(float(prob[1:]))
                        list_of_bools.append(True)
                    else:
                        list_of_probs.append(float(prob))
                        list_of_bools.append(False)
                pc.append(list_of_atoms)
                parameters.append(list_of_probs)
                learnable.append(list_of_bools)
                pi_prime += "1{"+"; ".join(list_of_atoms)+"}1.\n"
            else:
                asp += (line.strip()+"\n")

        pi_prime += asp

        for ruleIdx, list_of_bools in enumerate(learnable):
            remain_prob = 1
            for atomIdx, b in enumerate(list_of_bools):
                if b == False:
                    remain_prob -= parameters[ruleIdx][atomIdx]
            remain_probs.append(remain_prob)

        # parameters = np.array([np.array(l) for l in parameters])
        return pc, parameters, learnable, asp, pi_prime, remain_probs

    def normalize_probs(self):
        for ruleIdx, list_of_bools in enumerate(self.learnable):
            summation = 0
            # 1st, we turn each probability into [0+eps,1-eps]
            for atomIdx, b in enumerate(list_of_bools):
                if b == True:
                    if self.parameters[ruleIdx][atomIdx] >=1 :
                        self.parameters[ruleIdx][atomIdx] = 1- self.eps
                    elif self.parameters[ruleIdx][atomIdx] <=0:
                        self.parameters[ruleIdx][atomIdx] = self.eps

            # 2nd, we normalize the probabilities
            for atomIdx, b in enumerate(list_of_bools):
                if b == True:
                    summation += self.parameters[ruleIdx][atomIdx]
            for atomIdx, b in enumerate(list_of_bools):
                if b == True:
                    self.parameters[ruleIdx][atomIdx] = self.parameters[ruleIdx][atomIdx] / summation * self.remain_probs[ruleIdx]

        return True

    def prob_of_interpretation(self, I):
        prob = 1.0
        # I must be a list of atoms, where each atom is a string
        while not isinstance(I[0], str):
            I = I[0]
        for ruleIdx,list_of_atoms in enumerate(self.pc):
            for atomIdx, atom in enumerate(list_of_atoms):
                if atom in I:
                    prob = prob * self.parameters[ruleIdx][atomIdx]
        return prob

    # we assume obs is a string containing a valid Clingo program, 
    # and each obs is written in constraint form
    def find_one_SM_under_obs(self, obs):
        program = self.pi_prime + obs
        # print("program:\n{}\n".format(program))
        clingo_control = clingo.Control(["--warn=none"])
        models = []
        # print("\nPi': \n{}".format(program))
        clingo_control.add("base", [], program)
        # print("point 3")
        clingo_control.ground([("base", [])])
        # print("point 4")
        clingo_control.solve(None, lambda model: models.append(model.symbols(atoms=True)))
        # print("point 5")
        models = [[str(atom) for atom in model] for model in models]
        # print("point 6")
        # print("All stable models of Pi' under obs \"{}\" :\n{}\n".format(obs,models))
        return models

    # we assume obs is a string containing a valid Clingo program, 
    # and each obs is written in constraint form
    def find_all_SM_under_obs(self, obs):
        program = self.pi_prime + obs
        # print("program:\n{}\n".format(program))
        clingo_control = clingo.Control(["0", "--warn=none"])
        models = []
        # print("\nPi': \n{}".format(program))
        try:
            clingo_control.add("base", [], program)
        except:
            print("\nPi': \n{}".format(program))
        # print("point 3")
        clingo_control.ground([("base", [])])
        # print("point 4")
        clingo_control.solve(None, lambda model: models.append(model.symbols(atoms=True)))
        # print("point 5")
        models = [[str(atom) for atom in model] for model in models]
        # print("point 6")
        # print("All stable models of Pi' under obs \"{}\" :\n{}\n".format(obs,models))
        return models

    # k = 0 means to find all stable models
    def find_k_SM_under_obs(self, obs, k=3):
        # breakpoint()
        program = self.pi_prime + obs
        # print("program:\n{}\n".format(program))
        clingo_control = clingo.Control(["--warn=none", str(int(k))])
        models = []
        # print("\nPi': \n{}".format(program))
        try:
            clingo_control.add("base", [], program)
        except:
            print("\nPi': \n{}".format(program))
        # print("point 3")
        clingo_control.ground([("base", [])])
        # print("point 4")
        clingo_control.solve(None, lambda model: models.append(model.symbols(atoms=True)))
        # print("point 5")
        models = [[str(atom) for atom in model] for model in models]
        # print("point 6")
        # print("All stable models of Pi' under obs \"{}\" :\n{}\n".format(obs,models))
        return models

    # there might be some duplications in SMs when optimization option is used
    # and the duplications are removed by this method
    def remove_duplicate_SM(self, models):
        models.sort()
        return list(models for models,_ in itertools.groupby(models))

    # Note that the MVPP program cannot contain weak constraints
    def find_all_most_probable_SM_under_obs_noWC(self, obs):
        """Return a list of stable models, each is a list of strings
        @param obs: a string of a set of constraints/facts
        """
        program = self.pi_prime + obs + '\n'
        # for each probabilistic rule with n atoms, add n weak constraints
        for ruleIdx, atoms in enumerate(self.pc):
            for atomIdx, atom in enumerate(atoms):
                if self.parameters[ruleIdx][atomIdx] < 0.00674:
                    penalty = -1000 * -5
                else:
                    penalty = int(-1000 * math.log(self.parameters[ruleIdx][atomIdx]))
                program += ':~ {}. [{}, {}, {}]\n'.format(atom, penalty, ruleIdx, atomIdx)

        # print("program:\n{}\n".format(program))
        clingo_control = clingo.Control(['--warn=none', '--opt-mode=optN', '0', '-t', '8'])
        models = []
        clingo_control.add("base", [], program)
        clingo_control.ground([("base", [])])
        clingo_control.solve(None, lambda model: models.append(model.symbols(atoms=True)) if model.optimality_proven else None)
        models = [[str(atom) for atom in model] for model in models]
        return self.remove_duplicate_SM(models)

    def find_one_most_probable_SM_under_obs_noWC(self, obs=''):
        """Return a list of a single stable model, which is a list of strings
        @param obs: a string of a set of constraints/facts
        """
        program = self.pi_prime + obs + '\n'
        # for each probabilistic rule with n atoms, add n weak constraints
        for ruleIdx, atoms in enumerate(self.pc):
            for atomIdx, atom in enumerate(atoms):
                if self.parameters[ruleIdx][atomIdx] < 0.00674:
                    penalty = -1000 * -5
                else:
                    penalty = int(-1000 * math.log(self.parameters[ruleIdx][atomIdx]))
                program += ':~ {}. [{}, {}, {}]\n'.format(atom, penalty, ruleIdx, atomIdx)

        # print("program:\n{}\n".format(program))
        clingo_control = clingo.Control(['--warn=none', '-t', '8'])
        models = []
        clingo_control.add("base", [], program)
        clingo_control.ground([("base", [])])
        clingo_control.solve(None, lambda model: models.append(model.symbols(atoms=True)))
        models = [[str(atom) for atom in model] for model in models]
        return [models[-1]]

    def find_all_opt_SM_under_obs_WC(self, obs):
        """ Return a list of stable models, each is a list of strings
        @param obs: a string of a set of constraints/facts

        TODO: not all SM
        """
        program = self.pi_prime + obs
        clingo_control = clingo.Control(['--warn=none', '--opt-mode=optN', '0'])
        models = []
        try:
            clingo_control.add("base", [], program)
        except:
            print('\nSyntax Error in Program: Pi\': \n{}'.format(program))
            sys.exit()
        clingo_control.ground([("base", [])])
        clingo_control.solve(None, lambda model: models.append(model.symbols(atoms=True)) if model.optimality_proven else None)
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
        # we will compute P(I)/p_i where I satisfies obs and c=v_i
        p_obs_i = 0
        # we will compute P(I)/p_j where I satisfies obs and c=v_j for i!=j
        p_obs_j = 0
        # we will compute P(I) where I satisfies obs
        p_obs = 0

        # 1st, we generate all I that satisfies obs
        models = self.find_k_SM_under_obs(obs, k=3)
        # print("models are: {}".format(models))
        # 2nd, we iterate over each model I, and check if I satisfies c=v_i
        c_equal_vi = self.pc[ruleIdx][atomIdx]
        # print("c_equal_vi is: {}".format(c_equal_vi))
        p_i = self.parameters[ruleIdx][atomIdx]
        # print("p_i is: {}".format(p_i))
        for I in models:
            p_I = self.prob_of_interpretation(I)
            # print("p_I is: {}".format(p_I))
            # print("I: {}\t p_I: {}\t p_i: {}".format(I,p_I,p_i))
            p_obs += p_I
            if c_equal_vi in I:
                # if p_i == 0:
                #     p_i = self.eps
                p_obs_i += p_I/p_i
            else:
                for atomIdx2, p_j in enumerate(self.parameters[ruleIdx]):
                    c_equal_vj = self.pc[ruleIdx][atomIdx2]
                    if c_equal_vj in I:
                        # if p_j == 0:
                        #     p_j = self.eps
                        p_obs_j += p_I/p_j

        # 3rd, we compute gradient
        # print("p_obs_i: {}\t p_obs_j: {}\t p_obs: {}".format(p_obs_i,p_obs_j,p_obs))
        gradient = (p_obs_i-p_obs_j)/p_obs
        # print("gradient is: {}\n".format(gradient))

        return gradient

    # gradients are stored in numpy array instead of list
    # obs is a string
    # def gradients_one_obs(self, obs):
    #     gradients = [[0.0 for item in l] for l in self.parameters]
    #     models = self.find_k_SM_under_obs(obs, k=3)
    #     for ruleIdx,list_of_bools in enumerate(self.learnable):
    #         for atomIdx, b in enumerate(list_of_bools):
    #             if b == True:
    #                 # print("ruleIdx: {}\t atomIdx: {}\t obs: {}".format(ruleIdx, atomIdx, obs))
    #                 gradients[ruleIdx][atomIdx] = self.gradient(ruleIdx, atomIdx, obs)
    #     return gradients

    def mvppLearnRule(self, ruleIdx, models, probs):
        """Return a np array denoting the gradients for the probabilities in rule ruleIdx

        @param ruleIdx: an integer denoting a rule index
        @param models: the list of models that satisfy an underlined observation O, each model is a list of string
        @param probs: a list of probabilities, one for each model
        """
        
        gradients = []
        # if there is only 1 stable model, we learn from complete interpretation
        if len(models) == 1:
            model = models[0]
            # we compute the gradient for each p_i in the ruleIdx-th rule
            p = 0
            for i, cEqualsVi in enumerate(self.pc[ruleIdx]):
                if cEqualsVi in model:
                    gradients.append(1.0)
                    p = self.parameters[ruleIdx][i]
                else:
                    gradients.append(-1.0)
            for i, cEqualsVi in enumerate(self.pc[ruleIdx]):
                gradients[i] = gradients[i]/p

        # if there are more than 1 stable models, we use the equation in the proposition in the NeurASP paper
        else:
            denominator = sum(probs)
            # we compute the gradient for each p_i in the ruleIdx-th rule
            for i, cEqualsVi in enumerate(self.pc[ruleIdx]):
                numerator = 0
                # we accumulate the numerator by looking at each model I that satisfies O
                for modelIdx, model in enumerate(models):
                    # if I satisfies cEqualsVi
                    if cEqualsVi in model:
                        if self.parameters[ruleIdx][i] != 0:
                            numerator += probs[modelIdx] / self.parameters[ruleIdx][i]
                        else:
                            numerator += probs[modelIdx] / (self.parameters[ruleIdx][i] + self.eps)


                    # if I does not satisfy cEqualsVi
                    else:
                        for atomIdx, atom in enumerate(self.pc[ruleIdx]):
                            if atom in model:
                                if self.parameters[ruleIdx][atomIdx]!=0:
                                    numerator -= probs[modelIdx] / self.parameters[ruleIdx][atomIdx]
                                else:
                                    numerator -= probs[modelIdx] / (self.parameters[ruleIdx][atomIdx]+self.eps)

                # gradients.append(numerator / denominator)
                if denominator == 0 :
                    gradients.append(0)
                    # return np.array([0.0 for i in self.pc[ruleIdx]])
                else:
                    gradients.append(numerator / denominator)
        # print(gradients)
        return np.array(gradients)

    def mvppLearn(self, models):
        probs = [self.prob_of_interpretation(model) for model in models]
        gradients = np.array([[0.0 for item in l] for l in self.parameters])
        if len(models) != 0:
            # we compute the gradients w.r.t. the probs in each rule
            for ruleIdx,list_of_bools in enumerate(self.learnable):
                gradients[ruleIdx] = self.mvppLearnRule(ruleIdx, models, probs)
                for atomIdx, b in enumerate(list_of_bools):
                    if b == False:
                        gradients[ruleIdx][atomIdx] = 0
        # print(gradients)
        return gradients

    # gradients are stored in numpy array instead of list
    # obs is a string
    def gradients_one_obs(self, obs, opt=False):
        """Return an np-array denoting the gradients
        @param obs: a string for observation
        @param opt: a Boolean denoting whether we use optimal stable models instead of stable models
        """
        if opt:
            models = self.find_all_opt_SM_under_obs_WC(obs)
        else:
            models = self.find_k_SM_under_obs(obs, k=0)
            # program = self.pi_prime + obs
            # print(len(models))
            # breakpoint()
        # print('obs:\n{}'.format(obs))
        # print('models:\n{}'.format(models))
        # print('program:\n{}'.format(self.pi_prime))
        return self.mvppLearn(models)

    # gradients are stored in numpy array instead of list
    def gradients_multi_obs(self, list_of_obs):
        # gradients = np.zeros(self.parameters.shape)
        gradients = [[0.0 for item in l] for l in self.parameters]
        for obs in list_of_obs:
            # print("1")
            # print(gradients)
            # print("2")
            # print(self.gradients_one_obs(obs))
            gradients = [[c+d for c,d in zip(i,j)] for i,j in zip(gradients,self.gradients_one_obs(obs))]
            # gradients += self.gradients_one_obs(obs)
        #     print("3")
        #     print(gradients)
        # sys.exit()
        return gradients

    # list_of_obs is either a list of strings or a file containing observations separated by "#evidence"
    def learn_exact(self, list_of_obs, lr=0.01, thres=0.0001, max_iter=None):
        # if list_of_obs is an evidence file, we need to first turn it into a list of strings
        if type(list_of_obs) is str and os.path.isfile(list_of_obs):
            with open(list_of_obs, 'r') as f:
                list_of_obs = f.read().strip().strip("#evidence").split("#evidence")
        print("Start learning by exact computation with {} observations...\n\nInitial parameters: {}".format(len(list_of_obs), self.parameters))
        # print(list_of_obs)
        # sys.exit()
        time_init = time.time()
        check_continue = True
        iteration = 1
        while check_continue:
            old_parameters = self.parameters
            # print("===1===")
            # print(self.parameters)
            print("\n#### Iteration {} ####\n".format(iteration))
            check_continue = False
            # gradients_np = self.gradients_multi_obs(list_of_obs)
            # print(self.gradients_multi_obs(list_of_obs))
            dif = [[lr*grad for grad in l] for l in self.gradients_multi_obs(list_of_obs)]
            # dif = lr * self.gradients_multi_obs(list_of_obs)
            # print("dif :{}".format(dif))


            for ruleIdx, list_of_bools in enumerate(self.learnable):
            # 1st, we turn each gradient into [-0.2, 0.2]
                for atomIdx, b in enumerate(list_of_bools):
                    if b == True:
                        if dif[ruleIdx][atomIdx] > 0.2 :
                            dif[ruleIdx][atomIdx] = 0.2
                        elif dif[ruleIdx][atomIdx] < -0.2:
                            dif[ruleIdx][atomIdx] = -0.2


            # self.parameters = self.parameters + dif
            self.parameters = [[c+d for c,d in zip(i,j)] for i,j in zip(dif,self.parameters)]
            self.normalize_probs()

            # we termintate if the change of the parameters is lower than thres
            # dif = np.array(self.parameters) - old_parameters
            # print("1")
            # print(old_parameters)
            # print("2")
            # print(self.parameters)
            dif = [[abs(c-d) for c,d in zip(i,j)] for i,j in zip(old_parameters,self.parameters)]
            # print("3")
            # print(dif)
            # sys.exit()
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
        clingo_control = clingo.Control(["0", "--warn=none"])
        models = []
        for ruleIdx,list_of_atoms in enumerate(self.pc):
            tmp = np.random.choice(list_of_atoms, 1, p=self.parameters[ruleIdx])
            # print(tmp)
            asp_with_facts += tmp[0]+".\n"
        clingo_control.add("base", [], asp_with_facts)
        clingo_control.ground([("base", [])])
        result = clingo_control.solve(None, lambda model: models.append(model.symbols(shown=True)))
        models = [[str(atom) for atom in model] for model in models]
        # print("k")
        # print(models)
        return models

    # it will generate k*num sample stable models
    def sample(self, num=1):
        models = []
        for i in range(num):
            models = models + self.k_sample()
        # print("test")
        # print(models)
        return models

    # it will generate at least num of samples that satisfy obs
    def sample_obs(self, obs, num=50):
        count = 0
        models = []
        while count < num:
            # breakpoint()
            asp_with_facts = self.asp
            asp_with_facts += obs
            clingo_control = clingo.Control(["0", "--warn=none"])
            models_tmp = []
            for ruleIdx,list_of_atoms in enumerate(self.pc):
                # print("parameters before: {}".format(self.parameters[ruleIdx]))
                # self.normalize_probs()
                # print("parameters after: {}\n".format(self.parameters[ruleIdx]))
                tmp = np.random.choice(list_of_atoms, 1, p=self.parameters[ruleIdx])
                # print(tmp)
                asp_with_facts += tmp[0]+".\n"
            # breakpoint()
            clingo_control.add("base", [], asp_with_facts)
            clingo_control.ground([("base", [])])
            result = clingo_control.solve(None, lambda model: models_tmp.append(model.symbols(shown=True)))
            if str(result) == "SAT":
                models_tmp = [[str(atom) for atom in model] for model in models_tmp]
                # print("models_tmp:")
                # print(models_tmp)
                count += len(models_tmp)
                models = models + models_tmp
                # print("count: {}".format(count))
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
        clingo_control = clingo.Control(['0', '--warn=none'])
        clingo_control.add('base', [], program)
        clingo_control.ground([('base', [])])
        clingo_control.solve(None, lambda model: candidate_sm.append(model.symbols(shown=True)))
        candidate_sm = [[str(atom) for atom in model] for model in candidate_sm]
        probs = [self.prob_of_interpretation(model) for model in candidate_sm]
        breakpoint()

        while count < num:
            breakpoint()
            asp_with_facts = self.pi_prime
            asp_with_facts += obs
            clingo_control = clingo.Control(["0", "--warn=none"])
            models_tmp = []
            for ruleIdx,list_of_atoms in enumerate(self.pc):
                # print("parameters before: {}".format(self.parameters[ruleIdx]))
                # self.normalize_probs()
                # print("parameters after: {}\n".format(self.parameters[ruleIdx]))
                tmp = np.random.choice(list_of_atoms, 1, p=self.parameters[ruleIdx])
                # print(tmp)
                asp_with_facts += tmp[0]+".\n"
            # breakpoint()
            clingo_control.add("base", [], asp_with_facts)
            clingo_control.ground([("base", [])])
            result = clingo_control.solve(None, lambda model: models_tmp.append(model.symbols(shown=True)))
            if str(result) == "SAT":
                models_tmp = [[str(atom) for atom in model] for model in models_tmp]
                # print("models_tmp:")
                # print(models_tmp)
                count += len(models_tmp)
                models = models + models_tmp
                # print("count: {}".format(count))
            elif str(result) == "UNSAT":
                pass
            else:
                print("Error! The result of a clingo call is not SAT nor UNSAT!")
        return models

    # we compute the gradients (numpy array) w.r.t. all probs in the ruleIdx-th rule
    # given models that satisfy obs
    def gradient_given_models(self, ruleIdx, models):
        arity = len(self.parameters[ruleIdx])

        # we will compute N(O) and N(O,c=v_i)/p_i for each i
        n_O = 0
        n_i = [0]*arity

        # 1st, we compute N(O)
        n_O = len(models)

        # 2nd, we compute N(O,c=v_i)/p_i for each i
        for model in models:
            for atomIdx, atom in enumerate(self.pc[ruleIdx]):
                if atom in model:
                    n_i[atomIdx] += 1
        for atomIdx, p_i in enumerate(self.parameters[ruleIdx]):
            # if p_i == 0:
            #     p_i = self.eps
            n_i[atomIdx] = n_i[atomIdx]/p_i
        
        # 3rd, we compute the derivative of L'(O) w.r.t. p_i for each i
        tmp = np.array(n_i) * (-1)
        summation = np.sum(tmp)
        gradients = np.array([summation]*arity)
        # print(summation)
        # gradients = np.array([[summation for item in l] for l in self.parameters])
        # print("init gradients: {}".format(gradients))
        for atomIdx, p_i in enumerate(self.parameters[ruleIdx]):
            gradients[atomIdx] = gradients[atomIdx] + 2* n_i[atomIdx]
        gradients = gradients / n_O
        # print("n_O: {}".format(n_O))
        # print("n_i: {}\t n_O: {}\t gradients: {}".format(n_i, n_O, gradients))
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
        # print(gradients)
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
        # print(gradients)
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
        # print(gradients)
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