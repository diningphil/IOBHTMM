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

import random
from IOBHTMM import IOBHTMM
import TrainingUtilities
from Inex.InexParser import Parser
import time
def current_milli_time():
    return int(round(time.time() * 1000))

inexParser = Parser()

L = 32  # Max number of positions
M = 366  # Input alphabet size
K = 11  # Output alphabet size
dataset = inexParser.parse("./Inex/inex05.train.elastic.tree", maxOutdegree=L)
TS2005 = inexParser.parse("./Inex/inex05.test.elastic.tree", maxOutdegree=L)
random.shuffle(dataset)

'''
# INEX 2006
L = 66
M = 65  # Input alphabet size
K = 57  # Output alphabet size
dataset2006 = inexParser.parse("./Inex/inex06.train.elastic.tree", maxOutdegree=L)
TS2006 = inexParser.parse("./Inex/inex06.test.elastic.tree", maxOutdegree=L)
random.shuffle(dataset2006)
'''

C_values = [30]
runs = 1
max_epochs = 30
threshold = 1000.0
choose_by_vote = False
folds = 5

# for holdout
l = len(dataset)
trainLimit = int(l * 75 / 100)

trainSet = dataset[0:trainLimit]
valSet = dataset[trainLimit:]

best_C, tr_expected_log_likelihood, val_accuracy = TrainingUtilities.kfold_cv(C_values, L, M, K, dataset, folds,
                                max_epochs, threshold, TrainingUtilities.classification_score, runs, choose_by_vote,
                                        parallel=2, store=False, name='5fold2005')

model = IOBHTMM(L, 20, M, K, IOBHTMM.POSITIONAL)  # IOBHTMM.FULL for full stationarity

# an IOBHTMM has a function predict(self, single_sample) with which you can build your score function
# which takes a set and returns the measure you want.
score_fun = lambda set: TrainingUtilities.classification_score(model, set)

model.train(dataset, threshold=1000, max_epochs=max_epochs, print_score=False, pred_set=TS2005, score_fun=score_fun)

t0 = current_milli_time()
print("Accuracy is ", TrainingUtilities.classification_score(model, TS2005))
print("Avg prediction time is ", (current_milli_time()-t0)/len(TS2005), " ms")
