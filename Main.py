import random
from IOBHTMM import IOBHTMM
import TrainingUtilities
from Inex.InexParser import Parser
import time
def current_milli_time():
    return int(round(time.time() * 1000))

inexParser = Parser()

L = 32
M = 366  # Input alphabet size
K = 11  # Output alphabet size
dataset = inexParser.parse("./Inex/inex05.train.elastic.tree", maxOutdegree=L)
TS2005 = inexParser.parse("./Inex/inex05.test.elastic.tree", maxOutdegree=L)
random.shuffle(dataset)
random.shuffle(TS2005)  # for prediction func debugging

'''
L_2006 = 66
M_2006 = 65  # Input alphabet size
K_2006 = 57  # Output alphabet size
dataset_2006 = inexParser.parse("./Inex/inex06.train.elastic.tree", maxOutdegree=L_2006)
TS2006 = inexParser.parse("./Inex/inex06.test.elastic.tree", maxOutdegree=L_2006)
random.shuffle(dataset_2006)
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

'''
best_C, tr_expected_log_likelihood, val_accuracy = TrainingUtilities.kfold_cv(C_values, L, M, K, dataset, folds,
                                max_epochs, threshold, TrainingUtilities.classification_score, runs, choose_by_vote,
                                        parallel=2, store=False, name='5fold2005')
'''
model = IOBHTMM(L, 20, M, K, IOBHTMM.POSITIONAL)

score_fun = lambda set: TrainingUtilities.classification_score(model, set)
model.train(dataset, threshold=1000, max_epochs=max_epochs, print_score=False, pred_set=TS2005, score_fun=score_fun)
#model.load_model('debugging_model')

t0 = current_milli_time()
print("Accuracy is ", TrainingUtilities.classification_score(model, TS2005))
print("Avg prediction time is ", (current_milli_time()-t0)/len(TS2005), " ms")
