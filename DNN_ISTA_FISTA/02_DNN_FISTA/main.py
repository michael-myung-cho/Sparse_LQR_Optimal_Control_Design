#!/usr/bin/python
"""
This file is a main python file to run for sparse LQR optimal control design problem
By Myung (Michael) Cho

minimize J(K) + gamma*G(K)
s.t. K in F

J(K): LQR Cost
G(K): Sparsity-promoting regularization term, e.g., L1 norm or block L1 norm, etc.
F = { K | max( eig( Re( A-BK ) ) ) < 0 }: set for a stable feedback system
"""

import sys
sys.modules[__name__].__dict__.clear()


import problem_setting, LFISTA, train, test


# Create the basic problem, generate dataset with fixed gamma {K_LQR, K*_SP}
"""
Need to implement or import the data set in an proper format
Datasat = {K_LQR, K*_SP}, where K_LQR: stable dense matrix K (input), K*_SP: optimal sparse matrix K (output)
"""

"""
Parameter settings
M: # of rows of feedback matrix K  
N: # of columns of feedback matrix K
r: # of data points in dataset
T: # of layers in DNN
initial_gamma: weight parameter for sparsity regularization term
initial_rho: ISTA parameter related to step-size and threshold level
trinit: step size in training
maxit: Max. iteration in training
"""
prob = problem_setting.control_dataset()
print("Problem setting finished")


# build a Learning ISTA (DNN-ISTA) network to solve the optimal control design problem
# and get the intermediate results so we can greedily extend and then refine(fine-tune)
"""
prob: dataset and other predetermined constants, e.g., epsilon
gamma: sparsity level
T: ???
"""
layers = LFISTA.build_LFISTA(prob,T=30,initial_gamma=1.0,initial_rho=100,untied=False)
print("Network organization finished")

# plan the learning
training_stages = train.setup_training(layers,prob,trinit=0.0005)
print("Training setup finished")

# do the learning (takes a while)
sess = train.do_training(training_stages,layers,prob,maxit=300,r=1000)
print("Training finished")

# test
test.do_test(layers,sess,prob)
print("Test finished")
