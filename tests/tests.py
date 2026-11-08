import unittest
import unittest.mock as mock
import numpy as np
import torch

from mvpp import MVPP
from neurasp import NeurASP


class TestNeurASP(unittest.TestCase):

    def test_prob_of_interpretation(self):
        """Test that probabilities are calculated correctly"""

        # 6 models, 9 concepts
        models = [[3, 0, 3, 1, 3, 3, 3, 0, 3], [0, 3, 1, 1, 2, 2, 0, 1, 3], [0, 2, 1, 1, 2, 1, 2, 0, 3],
                  [0, 3, 1, 0, 2, 1, 1, 0, 1], [0, 3, 3, 0, 0, 2, 1, 1, 2], [1, 3, 3, 0, 0, 2, 2, 1, 0]]

        # 4 choices per concept
        pc = [
            ['test(i1,0)', 'test(i1,1)', 'test(i1,2)', 'test(i1,3)'],
            ['test(i2,0)', 'test(i2,1)', 'test(i2,2)', 'test(i2,3)'],
            ['test(i3,0)', 'test(i3,1)', 'test(i3,2)', 'test(i3,3)'],
            ['test(i4,0)', 'test(i4,1)', 'test(i4,2)', 'test(i4,3)'],
            ['test(i5,0)', 'test(i5,1)', 'test(i5,2)', 'test(i5,3)'],
            ['test(i6,0)', 'test(i6,1)', 'test(i6,2)', 'test(i6,3)'],
            ['test(i7,0)', 'test(i7,1)', 'test(i7,2)', 'test(i7,3)'],
            ['test(i8,0)', 'test(i8,1)', 'test(i8,2)', 'test(i8,3)'],
            ['test(i9,0)', 'test(i9,1)', 'test(i9,2)', 'test(i9,3)']
        ]

        parameters = torch.tensor([[0.2, 0.1, 0.8, 0.7], [0.9, 0.9, 0.4, 0.3], [0.6, 0.3, 0.1, 0.1],
                                   [0.7, 0.1, 0.2, 0.4], [0.1, 0.7, 0.5, 0.3], [0, 0.5, 0.9, 0.1], [0.5, 0.7, 0, 0.7],
                                   [0.6, 0.3, 0.1, 0.7], [0.9, 0, 0.2, 0.5]])

        mock_return = (pc, parameters, False, "mock_asp", "mock_pi", "mock_remain_probs")

        probs = [0.00003969, 0.00006075, 0, 0, 0.000015876, 0]

        with (mock.patch.object(MVPP, 'parse', return_value=mock_return),
              mock.patch.object(MVPP, 'normalize_probs')):
            mvpp = MVPP('')
            neurasp_probs = mvpp.prob_of_interpretation(models)

        np.testing.assert_almost_equal(neurasp_probs, probs)

    def test_mvppLearnRule(self):
        """Test that gradients are calculated correctly"""

        # 8 models, 9 concepts
        models = torch.IntTensor(
                 [[1, 0, 0, 5, 7, 7, 5, 3, 2], [6, 2, 0, 1, 0, 6, 4, 5, 8], [5, 1, 8, 6, 1, 2, 4, 6, 4],
                  [8, 8, 3, 0, 7, 0, 3, 1, 3], [2, 4, 7, 1, 3, 3, 5, 1, 2], [3, 3, 5, 3, 8, 8, 3, 0, 2],
                  [3, 4, 8, 5, 8, 5, 2, 0, 4], [7, 4, 3, 1, 0, 2, 5, 7, 6]])

        # 9 choices per concept
        pc = [
            ['test(i1,0)', 'test(i1,1)', 'test(i1,2)', 'test(i1,3)', 'test(i1,4)', 'test(i1,5)', 'test(i1,6)',
             'test(i1,7)', 'test(i1,8)'],
            ['test(i2,0)', 'test(i2,1)', 'test(i2,2)', 'test(i2,3)', 'test(i2,4)', 'test(i2,5)', 'test(i2,6)',
             'test(i2,7)', 'test(i2,8)'],
            ['test(i3,0)', 'test(i3,1)', 'test(i3,2)', 'test(i3,3)', 'test(i3,4)', 'test(i3,5)', 'test(i3,6)',
             'test(i3,7)', 'test(i3,8)'],
            ['test(i4,0)', 'test(i4,1)', 'test(i4,2)', 'test(i4,3)', 'test(i4,4)', 'test(i4,5)', 'test(i4,6)',
             'test(i4,7)', 'test(i4,8)'],
            ['test(i5,0)', 'test(i5,1)', 'test(i5,2)', 'test(i5,3)', 'test(i5,4)', 'test(i5,5)', 'test(i5,6)',
             'test(i5,7)', 'test(i5,8)'],
            ['test(i6,0)', 'test(i6,1)', 'test(i6,2)', 'test(i6,3)', 'test(i6,4)', 'test(i6,5)', 'test(i6,6)',
             'test(i6,7)', 'test(i6,8)'],
            ['test(i7,0)', 'test(i7,1)', 'test(i7,2)', 'test(i7,3)', 'test(i7,4)', 'test(i7,5)', 'test(i7,6)',
             'test(i7,7)', 'test(i7,8)'],
            ['test(i8,0)', 'test(i8,1)', 'test(i8,2)', 'test(i8,3)', 'test(i8,4)', 'test(i8,5)', 'test(i8,6)',
             'test(i8,7)', 'test(i8,8)'],
            ['test(i9,0)', 'test(i9,1)', 'test(i9,2)', 'test(i9,3)', 'test(i9,4)', 'test(i9,5)', 'test(i9,6)',
             'test(i9,7)', 'test(i9,8)']
        ]

        probs = np.array([0, 0.1, 0, 1, 0.1, 0.6, 0, 0.2])

        parameters = torch.tensor([[0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1],
            [0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1], [0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1],
            [0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.3], [0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1], [0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0, 0.1],
            [0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3], [0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2]])

        mock_return = (pc, parameters, False, "mock_asp", "mock_pi", "mock_remain_probs")

        grads = [[-9, -9, -8.5, -3, -9, -9, -8.5, -8, 1], [-10, -10, -9, -4, -7, -10, -10, -10, 0],
                 [-7.25, -8.25, -8.25, 3.75, -8.25, -5.25, -8.25, -7.75, -8.25], [1, -7, -9, -3, -9, -9, -9, -9, -9],
                 [-7.75, -9.25, -9.25, -8.25, -9.25, -9.25, -9.25, 0.75, -3.25],
                 [0.5, -9.5, -7.5, -9, -9.5, -9.5, -9, -9.5, -3.5],
                 [-9, -9, -9, 7, -8.5, -7.5, -9, -9, -9], [-3.5, 1.5, -9.5, -9.5, -9.5, -8.5, -9.5, -8.5, -9.5],
                 [-9.75, -9.75, -2.75, 0.25, -9.75, -9.75, -7.75, -9.75, -9.25]]

        with (mock.patch.object(MVPP, 'parse', return_value=mock_return),
              mock.patch.object(MVPP, 'normalize_probs')):
            mvpp = MVPP('')
            neurasp_grads = mvpp.mvppLearnRule(models, torch.Tensor(probs), 9)

        np.testing.assert_almost_equal(neurasp_grads, grads)

    def test_find_k_SM_under_obs(self):
        """Test that models are found correctly"""

        # 5 concepts, 5 choices
        pi_prime = ("1{test(1,i1,1); test(1,i1,2); test(1,i1,3); test(1,i1,4); test(1,i1,5)}1.\n"
                    "1{test(1,i2,1); test(1,i2,2); test(1,i2,3); test(1,i2,4); test(1,i2,5)}1.\n"
                    "1{test(1,i3,1); test(1,i3,2); test(1,i3,3); test(1,i3,4); test(1,i3,5)}1.\n"
                    "1{test(1,i4,1); test(1,i4,2); test(1,i4,3); test(1,i4,4); test(1,i4,5)}1.\n"
                    "1{test(1,i5,1); test(1,i5,2); test(1,i5,3); test(1,i5,4); test(1,i5,5)}1.\n"
                    "result(N) :- test(1,i1,N1), test(1,i2,N2), test(1,i3,N3), test(1,i4,N4), test(1,i5,N5), "
                    "N=(N1+N3)*10+N2+N4-N5.")
        obs = ":- not result(109)."
        pc = {'test/3:1,i1': ['1', '2', '3', '4', '5'], 'test/3:1,i2': ['1', '2', '3', '4', '5'],
              'test/3:1,i3': ['1', '2', '3', '4', '5'], 'test/3:1,i4': ['1', '2', '3', '4', '5'],
              'test/3:1,i5': ['1', '2', '3', '4', '5']}

        # There is one stable model that satisfies the observation
        models = np.array([[4, 4, 4, 4, 0]])

        mock_return = (pc, [], False, "mock_asp", pi_prime, "mock_remain_probs")

        with (mock.patch.object(MVPP, 'parse', return_value=mock_return),
              mock.patch.object(MVPP, 'normalize_probs')):
            mvpp = MVPP('')
            neurasp_models = mvpp.find_k_SM_under_obs(obs)
            assert (neurasp_models == models).all()

    def test_find_all_opt_SM_under_obs_WC(self):
        """Test that optimal models are found correctly"""

        # 2 concepts, 2 choices
        pi_prime = ("1{in(1,i1,true); in(1,i1,false)}1. 1{in(1,i2,true); in(1,i2,false)}1."
                    ":- #sum{1, I : in(1,I,true)} > 1.")
        obs = ":~ in(1,i1,true). [-3,r1] :~ in(1,i2,true). [-6,r2]"
        pc = {'in/3:1,i1': ['true', 'false'], 'in/3:1,i2': ['true', 'false']}

        # There is one optimal stable model that minimises the weak constraints
        models = np.array([[1, 0]])

        mock_return = (pc, [], False, "mock_asp", pi_prime, "mock_remain_probs")

        with (mock.patch.object(MVPP, 'parse', return_value=mock_return),
              mock.patch.object(MVPP, 'normalize_probs')):
            mvpp = MVPP('')
            neurasp_models = mvpp.find_k_SM_under_obs(obs, opt=True)
            assert (neurasp_models == models).all()

    def test_stable_model_caching(self):
        """Test that stable models are stored correctly"""

        # Create a simple neural network
        m = torch.nn.Linear(9, 8)
        nnMapping = {'test': m}
        optimizer = {'test': torch.optim.Adam(m.parameters())}

        # 9 data points, each with 8 choices
        dataList = [{'i': torch.rand(9)} for i in range(9)]
        obsList = [':- not obs(8).', ':- not obs(4).', ':-not obs(8).', ':- not obs(1).', ':- not obs(3)',
                   ':- not obs(2)', ':- nots obs(2)', ':- nots obs(3).', ':- not obs(6).']
        dataset = list(zip(dataList, obsList))
        mock_return = ('program', 'program_pr', 'program_asp')
        stable_models = ['8', '4', '8', '1', '3', '2', '2', '3', '6']

        with (mock.patch.object(NeurASP, 'parse', return_value=mock_return),
              mock.patch('neurasp.MVPP') as mock_mvpp):
            mock_mvpp.return_value.find_k_SM_under_obs = lambda obs, k, opt: obs.split('(')[1].split(')')[0]
            NeurASPobj = NeurASP('dprogram', nnMapping, optimizer)
            NeurASPobj.learn(dataset, 2, storeSM=True)
            assert NeurASPobj.stableModels == {obs: sm for obs, sm in zip(obsList, stable_models)}

    def test_calculate_accuracies(self):
        """Test that accuracies calculated correctly"""
        # Set seed so that nn output is the same every time
        # Predictions for m will be [1,1,1,1,2,1,1,1] and [1,1,1,2,1,0,1,1]
        torch.manual_seed(8)

        # 2 networks processing the same input
        # 2 data points, each with 8 concepts and 3 choices
        m = torch.nn.Linear(3, 3)
        n = torch.nn.Linear(3, 3)
        # Only the j concept has latent labels
        dataset = [({'i': torch.rand(8, 3), 'j': (torch.rand(8, 3),
                     {'test_latent': torch.tensor([1, 2, 0, 0, 1, 1, 0, 1])})}, ':- not obs(1).'),
                   ({'i': torch.rand(8, 3), 'j': (torch.rand(8, 3),
                     {'test_latent': torch.tensor([1, 1, 1, 1, 0, 2, 1, 1])})}, ':- not obs(3).')]

        nnMapping = {'test': m, 'test_latent': n}
        optimizer = {'test': torch.optim.Adam(m.parameters()), 'test_latent': torch.optim.Adam(m.parameters())}
        mock_return = ('program', 'program_pr', 'program_asp')
        nn_prob = [[('test', 0, 'i', 0)], [('test', 1, 'i', 0)], [('test', 2, 'i', 0)], [('test', 3, 'i', 0)],
                   [('test', 4, 'i', 0)], [('test', 5, 'i', 0)], [('test', 6, 'i', 0)], [('test', 7, 'i', 0)],
                   [('test_latent', 0, 'j', 0)], [('test_latent', 1, 'j', 0)], [('test_latent', 2, 'j', 0)],
                   [('test_latent', 3, 'j', 0)], [('test_latent', 4, 'j', 0)], [('test_latent', 5, 'j', 0)],
                   [('test_latent', 6, 'j', 0)], [('test_latent', 7, 'j', 0)]]

        # Only obs(1) has cached stable models
        # No entry matches the nn prediction
        stable_models = {':- not obs(1).': torch.IntTensor([[2, 2, 2, 2, 2, 2, 0, 2, 0, 1, 2, 0, 1, 1, 1, 1],
                                                            [2, 0, 2, 1, 1, 1, 1, 1, 0, 2, 0, 0, 1, 1, 2, 1],
                                                            [1, 1, 1, 1, 0, 0, 2, 1, 0, 2, 2, 2, 2, 0, 1, 2],
                                                            [2, 0, 1, 2, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]])}

        # The stable model for obs(3) has to be fetched from the dmvpp
        # The sixth entry matches the nn prediction
        found_stable_models = torch.IntTensor(
            [[0, 1, 1, 2, 2, 2, 0, 0, 2, 0, 0, 2, 2, 0, 2, 2], [2, 2, 2, 0, 1, 0, 1, 2, 0, 1, 2, 2, 2, 0, 1, 0],
             [0, 1, 2, 0, 2, 2, 2, 2, 0, 2, 1, 2, 2, 2, 2, 0], [2, 0, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2],
             [2, 2, 0, 0, 0, 0, 2, 2, 1, 2, 2, 0, 2, 2, 2, 1], [2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1],
             [0, 0, 2, 0, 0, 1, 2, 0, 2, 1, 2, 1, 1, 1, 2, 1], [0, 1, 1, 1, 1, 2, 0, 2, 2, 0, 2, 1, 1, 0, 0, 0],
             [1, 2, 1, 0, 2, 0, 1, 0, 1, 2, 0, 0, 0, 0, 2, 1]])

        def mvpp_side_effect(obs, k, opt):
            if obs == ':- not obs(3).':
                return found_stable_models
            return None

        with (mock.patch.object(NeurASP, 'parse', return_value=mock_return),
              mock.patch('neurasp.MVPP') as mock_mvpp):
            mock_mvpp.parameters = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            mock_mvpp.find_k_SM_under_obs.side_effect = mvpp_side_effect
            NeurASPobj = NeurASP('dprogram', nnMapping, optimizer)
            NeurASPobj.nnOutputs = {'test': ['i'], 'test_latent': ['j']}
            NeurASPobj.stableModels = stable_models
            NeurASPobj.mvpp['nnPrRuleNum'] = 16
            NeurASPobj.mvpp['nnProb'] = nn_prob
            NeurASPobj.e = {'test': 8, 'test_latent': 8}
            down_acc, latent_acc = NeurASPobj.calculate_accuracies(dataset, mock_mvpp, True, False)

            # Only the second out of the two inputs yields the correct prediction
            assert down_acc == 0.5
            # The test network does not have latent labels
            assert latent_acc['test'] == 'unknown'
            # The test_latent network gets half of the latent labels correct
            assert latent_acc['test_latent'] == 8/16

    def test_gradient(self):
        """Test that gradient is calculated correctly"""
        # 5 concepts, 5 choices
        pi_prime = ("1{test(1,i1,1); test(1,i1,2); test(1,i1,3); test(1,i1,4); test(1,i1,5)}1.\n"
                    "1{test(1,i2,1); test(1,i2,2); test(1,i2,3); test(1,i2,4); test(1,i2,5)}1.\n"
                    "1{test(1,i3,1); test(1,i3,2); test(1,i3,3); test(1,i3,4); test(1,i3,5)}1.\n"
                    "1{test(1,i4,1); test(1,i4,2); test(1,i4,3); test(1,i4,4); test(1,i4,5)}1.\n"
                    "1{test(1,i5,1); test(1,i5,2); test(1,i5,3); test(1,i5,4); test(1,i5,5)}1.\n"
                    "result(N) :- test(1,i1,N1), test(1,i2,N2), test(1,i3,N3), test(1,i4,N4), test(1,i5,N5), "
                    "N=N1*N2+N3+N4+N5.")
        obs = ":- not result(39)."
        pc = {'test/3:1,i1': ['1', '2', '3', '4', '5'], 'test/3:1,i2': ['1', '2', '3', '4', '5'],
              'test/3:1,i3': ['1', '2', '3', '4', '5'], 'test/3:1,i4': ['1', '2', '3', '4', '5'],
              'test/3:1,i5': ['1', '2', '3', '4', '5']}
        parameters = torch.tensor([[0.2, 0.4, 0.1, 0.1, 0.2], [0.1, 0.2, 0.1, 0.4, 0.2], [0.1, 0.2, 0.1, 0.4, 0.2],
                                   [0.2, 0.0, 0.0, 0.4, 0.4], [0.5, 0.3, 0.0, 0.2, 0.0]])

        mock_return = (pc, parameters, False, "mock_asp", pi_prime, "mock_remain_probs")

        with (mock.patch.object(MVPP, 'parse', return_value=mock_return),
              mock.patch.object(MVPP, 'normalize_probs')):
            mvpp = MVPP('')
            gradient = mvpp.gradient(0, 3, obs)
        np.testing.assert_almost_equal(-5, gradient)

    def test_gradient_given_models(self):
        """Test that gradient is calculated correctly given models"""
        # 4 concepts, 7 choices, 7 models
        pc = [['test(1,i1,1)', 'test(1,i1,2)', 'test(1,i1,3)', 'test(1,i1,4)', 'test(1,i1,5)', 'test(1,i1,6)',
               'test(1,i1,7)'],
              ['test(1,i2,1)', 'test(1,i2,2)', 'test(1,i2,3)', 'test(1,i2,4)', 'test(1,i2,5)', 'test(1,i2,6)',
               'test(1,i2,7)'],
              ['test(1,i3,1)', 'test(1,i3,2)', 'test(1,i3,3)', 'test(1,i3,4)', 'test(1,i3,5)', 'test(1,i3,6)',
               'test(1,i3,7)'],
              ['test(1,i4,1)', 'test(1,i4,2)', 'test(1,i4,3)', 'test(1,i4,4)', 'test(1,i4,5)', 'test(1,i4,6)',
               'test(1,i5,7)']]
        parameters = torch.tensor([[0.3, 0.0, 0.2, 0.1, 0.1, 0.1, 0.2], [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2],
                                   [0.0, 0.2, 0.1, 0.1, 0.2, 0.2, 0.2], [0.1, 0.1, 0.3, 0.2, 0.1, 0.2, 0.0]])

        models = torch.IntTensor([[6, 5, 0, 4], [1, 3, 4, 5], [6, 1, 1, 3], [4, 6, 0, 3], [5, 6, 4, 5],
                                  [3, 3, 5, 6], [2, 1, 2, 4]])
        mock_return = (pc, parameters, False, "mock_asp", "mock_pi_prime", "mock_remain_probs")

        with (mock.patch.object(MVPP, 'parse', return_value=mock_return),
              mock.patch.object(MVPP, 'normalize_probs')):
            mvpp = MVPP('')
            neurasp_gradients = mvpp.gradient_given_models(1,  models)
        gradients = [-7.142857142857143, -4.285714285714286, -7.142857142857143, -1.4285714285714286,
                     -7.142857142857143, -4.285714285714286, -4.285714285714286]
        np.testing.assert_almost_equal(neurasp_gradients, gradients)

    def test_normalize_probs(self):
        """Test that probabilities are normalized correctly"""
        # 1 concept, 5 choices
        learnable = torch.tensor([[True, False, False, False, True]])
        parameters = torch.tensor([[0.7, 0.2, 0.3, 0.1, 0.3]])
        # Epsilon forces values to be between 0.4 and 0.6
        eps = 0.4
        # Non-learnable choices make up 60% of the probability mass, so 40% are remaining
        remain_probs = torch.tensor([0.4])
        mock_return = ("mock_pc", parameters, learnable, "mock_asp", "mock_pi_prime", remain_probs)

        with (mock.patch.object(MVPP, 'parse', return_value=mock_return)):
            mvpp = MVPP('', eps=eps)
            # Instantiating MVVP calls normalize_probs, so we get the changed parameters
            neurasp_probs = mvpp.parameters
        normalized_probs = [[0.24, 0.2, 0.3, 0.1, 0.16]]
        np.testing.assert_almost_equal(neurasp_probs,normalized_probs)
