# ################################################################### #
# Copyright © 2017 Federico Errica                                    #
#                                                                     #
# Input-Output Bottom-Up Hidden Tree Markov Model (IOBHTMM).          #
# Bacciu, D., Micheli, A. and Sperduti, A., 2013.                     #
# An input–output hidden Markov model for tree transductions.         #
# Neurocomputing, 112, pp.34-46.                                      #
#                                                                     #
# This file is part of the IOBHTMM.                                   #
#                                                                     #
# IOBHTMM is free software: you can redistribute it and/or modify     #
# it under the terms of the GNU General Public License as published by#
# the Free Software Foundation, either version 3 of the License, or   #
# (at your option) any later version.                                 #
#                                                                     #
# IOBHTMM is distributed in the hope that it will be useful,          #
# but WITHOUT ANY WARRANTY; without even the implied warranty of      #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the        #
# GNU General Public License for more details.                        #
#                                                                     #
# You should have received a copy of the GNU General Public License   #
# along with IOBHTMM. If not, see <http://www.gnu.org/licenses/>.     #
#                                                                     #
# ################################################################### #

from IOBHTMM import IOBHTMM
from Tree import Node
import TrainingUtilities

# define the alphabet dimensions and the maximum degree of a tree
L = 10  # Maximum degree
M = 15  # Input alphabet's dimension
K = 20  # Output alphabet's dimension

# #######################################################################
#                                                                       #
#        Step 1: PREPROCESSING -> create a tree using the Node class.   #
#                                                                       #
#########################################################################

# You just create a root (a Node object with null parent and position) and add children.
# You can get the child in l-th position with an appropriate method, as shown below
# PLEASE NOTE: the position parameter goes from 0 to L-1
root_1 = Node(x=4, y=11, l=L, parent=None, position=None)
root_1.add_child(x=3, y=5, pos=0)
root_1.add_child(x=1, y=2, pos=6)
child_0 = root_1.get_lth_child(0)
child_0.add_child(x=0, y=7, pos=8)  # and so on...

# Create another sample for the dataset
root_2 = Node(x=4, y=11, l=L, parent=None, position=None)
root_2.add_child(x=3, y=5, pos=0)
root_2.add_child(x=1, y=2, pos=6)
child_0 = root_2.get_lth_child(0)
child_0.add_child(x=0, y=7, pos=8)

# A dataset (training, validation, test) is simply a list of root nodes
dataset = [root_1, root_2]
test_set = dataset  # You should also have your test set. Here, for brevity, it is equal to the training set.


# #######################################################################
#                                                                       #
#                           Step 2: TRAINING                            #
#                                                                       #
#########################################################################

# You can now define lists of hyperparameters for a holdout or k-fold cross validation
C_values = [10, 20]  # The number of possible hidden states of a node
runs = 1  # Each training is repeated "runs" times, and the results averaged
max_epochs = 30
threshold = 0.0  # A threshold for the likelihood computed at each epoch

# For classification tasks, a node can be chosen as the label of the root,
# or as the most common state in the tree (by "vote")
choose_by_vote = False

folds = 5  # The number of folds (in case of a K-FOLD cross validation)

#                       JUST FOR HOLDOUT                            #
# we split the training set between actual training and validation  #
lim = len(dataset)                                                  #
trainLimit = int(lim * 75 / 100)                                    #
trainSet = dataset[0:trainLimit]                                    #
valSet = dataset[trainLimit:]                                       #
#####################################################################


# HOLDOUT example
'''
best_C, final_likelihood, best_validation_accuracy, likelihood_history = \
    TrainingUtilities.holdout(C_values, L, M, K, IOBHTMM.FULL, trainSet, valSet, max_epochs, threshold,
                              TrainingUtilities.classification_score, runs, choose_by_vote,
                              parallel=2, store=False, name='my_holdout')

model = IOBHTMM(L, best_C, M, K, IOBHTMM.FULL)  # or IOBHTMM.FULL for full stationarity
'''

# K-FOLD example
best_C, final_likelihood, best_validation_accuracy = \
    TrainingUtilities.kfold_cv(C_values, L, M, K, IOBHTMM.POSITIONAL, dataset, folds, max_epochs, threshold,
                               TrainingUtilities.classification_score, runs, choose_by_vote,
                               parallel=2, store=False, name='my_kfold')

model = IOBHTMM(L, best_C, M, K, IOBHTMM.POSITIONAL)  # or IOBHTMM.FULL for full stationarity

# an IOBHTMM has a function predict(self, single_sample) with which you can build your score function
# which takes a set and returns the measure you want. You can use lambdas to obfuscate the model in this way
score_fun = lambda set: TrainingUtilities.classification_score(model, set)

# STEP 3: final training on the entire training set. If you want to see how training affects prediction on test phase,
# pass the test set as argument, as done below. In that case, you also need to provide the scoring function
model.train(dataset, threshold=threshold, max_epochs=max_epochs, print_score=False, pred_set=test_set, score_fun=score_fun)

# You can save/load models by calling, respectively
model.save_model('debugging_model')
model.load_model('debugging_model')

# #######################################################################
#                                                                       #
#                           Step 3: PREDICTION                          #
#                                                                       #
#########################################################################
print("Accuracy is ", TrainingUtilities.classification_score(model, test_set, choose_by_vote=choose_by_vote))
