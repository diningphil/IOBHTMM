from IOBHTMM import IOBHTMM
import concurrent.futures
from threading import Lock
from datetime import datetime
import numpy as np
import logging
import pickle
import math


def __parallel_holdout_computation(C, L, M, K, train_set, val_set,
                                   max_epochs, threshold, runs, choose_by_vote,
                                   best_C, score_function, tr_expected_complete_log_likelihood, val_accuracy,
                                   training_histories, lock, store, name):
    avg_ell = 0.
    avg_class_accuracy = 0.

    best_acc_over_runs = 0.
    best_tr_history = None

    accuracies = np.zeros(runs)

    for t in range(0, runs):
        print("Run %d for C = %d" % (t+1, C))

        model = IOBHTMM(L, C, M, K)
        training_history = model.train(train_set, threshold=threshold, max_epochs=max_epochs)

        class_accuracy = score_function(model, val_set, choose_by_vote=choose_by_vote)

        accuracies[t] = class_accuracy

        # Just a way to get the best training history over runs (on the VL set)
        if class_accuracy > best_acc_over_runs:

            if store:
                model.store_model("Model_C_" + str(C) + "_" + name + '_' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                # Store only models that improve results on VL set
            best_acc_over_runs = class_accuracy
            best_tr_history = training_history

        avg_ell = avg_ell + training_history[-1]
        avg_class_accuracy = avg_class_accuracy + class_accuracy

    avg_ell = avg_ell / runs
    avg_class_accuracy = avg_class_accuracy / runs

    num = accuracies - avg_class_accuracy
    std = np.sqrt(np.dot(num, num)/runs)

    lock.acquire()

    logging.info('Completed runs for C=' + str(C) + ' avg expected complete log likelihood is '
                 + str(avg_ell) + ' avg class accuracy on VL set is ' + str(avg_class_accuracy) + ' std is ' + str(std))

    if avg_class_accuracy > val_accuracy[0]:
        tr_expected_complete_log_likelihood[0] = avg_ell
        val_accuracy[0] = avg_class_accuracy
        best_C[0] = C

    training_histories.append((C, best_tr_history))

    lock.release()


def holdout(c_values, L, M, K, train_set, val_set, max_epochs, threshold, score_function, runs=1, choose_by_vote=False,
            parallel=-1, store=False, name=''):
    # PRECONDITION: score_function must return a value that must be maximised (e.g. accuracy ok, error not ok)

    logging.basicConfig(filename='./logging/CV_info_' + name + '_' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                                 '.log', level=logging.DEBUG)
    logging.info('Starting CV with parameters ' + str(c_values) + " " + str(l) + " " + str(m) + " " + str(k) + " " +
                 str(max_epochs) + " " + str(threshold) + " " + str(runs) + " " + str(choose_by_vote)
                 + " " + str(parallel) + " " + str(store))

    # Simple hold-out validation

    # An ugly trick to pass integers by reference
    best_C = [-1]
    tr_expected_complete_log_likelihood = [0.]
    val_accuracy = [0.]
    training_histories = []

    lock = Lock()

    if parallel <= 1:
        for C in c_values:
            print("Training for C value ", C)
            __parallel_holdout_computation(C, L, M, K, train_set, val_set,
                                           max_epochs, threshold, runs, choose_by_vote,
                                           best_C, score_function, tr_expected_complete_log_likelihood, val_accuracy,
                                           training_histories, lock, store, name)

    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            # Start the load operations and mark each future with its URL
            future_to_C = {executor.submit(__parallel_holdout_computation, C, L, M, K, train_set, val_set,
                                           max_epochs, threshold, runs, choose_by_vote,
                                           best_C, score_function, tr_expected_complete_log_likelihood, val_accuracy,
                                           training_histories, lock, store, name): C for C in c_values}
            for future in concurrent.futures.as_completed(future_to_C):
                C = future_to_C[future]
                try:
                    future.result()
                except Exception as exc:
                    print('%r execution generated an exception: %s' % (C, exc))
                else:
                    print('Execution for %d is completed' % C)

    logging.info("Best C was " + str(best_C) + " val accuracy on VL was " + str(val_accuracy) + " log likelihood was " +
                 str(tr_expected_complete_log_likelihood))
    if store:
        with open('Training_histories_CV_' + name + '_' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'wb') as f:
            pickle.dump(training_histories, f)
    return best_C[0], tr_expected_complete_log_likelihood[0], val_accuracy[0], training_histories


def __parallel_kfold_computation(C, L, M, K, train_set, folds,
                                   max_epochs, threshold, runs, choose_by_vote,
                                   best_C, score_function, tr_expected_complete_log_likelihood, val_accuracy,
                                   lock):
    
    set_dim = len(train_set)
    fold_dim = math.floor(set_dim / folds)

    # Average expected risk over folds
    avg_fold_accuracy = 0.
    avg_fold_ell = 0.
    folds_avg_accuracies = np.zeros(folds)

    for k in range(0, folds):
        print("Fold %d out of %d for C = %d" % (k+1, folds, C))

        # Build the folders
        val_k = train_set[k * fold_dim + 1:(k + 1) * fold_dim]

        if k == 0:
            train_k = train_set[(k+1) * fold_dim + 1:]
        elif k == folds-1:
            train_k = train_set[1:k * fold_dim]
        else:
            train_k = train_set[1:k * fold_dim] + train_set[(k+1) * fold_dim + 1:]

        # Estimate the expected risk over this val_k fold after training

        avg_runs_ell = 0.
        avg_runs_accuracy = 0.

        for t in range(0, runs):
            print("Run %d for C = %d" % (t+1, C))

            model = IOBHTMM(L, C, M, K)
            training_history = model.train(train_k, threshold=threshold, max_epochs=max_epochs)

            class_accuracy = score_function(model, val_k, choose_by_vote=choose_by_vote)

            avg_runs_ell = avg_runs_ell + training_history[-1]
            avg_runs_accuracy = avg_runs_accuracy + class_accuracy

        avg_runs_accuracy = avg_runs_accuracy / runs
        avg_runs_ell = avg_runs_ell / runs

        print("Avg class accuracy on fold ", (k+1), ": ", avg_runs_accuracy)

        avg_fold_ell = avg_fold_ell + avg_runs_ell
        avg_fold_accuracy = avg_fold_accuracy + avg_runs_accuracy
        folds_avg_accuracies[k] = avg_runs_accuracy

    avg_fold_accuracy = avg_fold_accuracy / folds
    avg_fold_ell = avg_fold_ell / folds

    num = folds_avg_accuracies - avg_fold_accuracy
    std_fold_accuracy = np.sqrt(np.dot(num, num)/folds)

    lock.acquire()

    logging.info('Completed k-fold for C=' + str(C) + ' avg expected complete log likelihood is '
                 + str(avg_fold_ell) + ' avg class accuracy on folds set is ' + str(avg_fold_accuracy) + ' std is ' +
                 str(std_fold_accuracy))

    if avg_fold_accuracy > val_accuracy[0]:
        tr_expected_complete_log_likelihood[0] = avg_fold_ell
        val_accuracy[0] = avg_fold_accuracy
        best_C[0] = C

    lock.release()


def kfold_cv(c_values, l, m, k, train_set, folds, max_epochs, threshold, score_function, runs=1, choose_by_vote=False, parallel=-1,
            store=False, name=''):
    # PRECONDITION: score_function must return a value that must be maximised (e.g. accuracy ok, error not ok)

    logging.basicConfig(filename='./logging/kfold_info_' + name + '_' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                                 '.log', level=logging.DEBUG)
    logging.info(
        'Starting k fold CV with parameters ' + str(c_values) + " " + str(l) + " " + str(m) + " " + str(k) + " " +
        str(folds) + " " + str(max_epochs) + " " + str(threshold) + " " + str(runs) + " " + str(choose_by_vote)
        + " " + str(parallel) + " " + str(store))

    # Simple hold-out validation

    # An ugly trick to pass integers by reference
    best_C = [-1]
    tr_expected_complete_log_likelihood = [0.]
    val_accuracy = [0.]

    lock = Lock()

    if parallel <= 1:
        for C in c_values:
            print("Training for C value ", C)
            __parallel_kfold_computation(C, l, m, k, train_set, folds,
                                           max_epochs, threshold, runs, choose_by_vote,
                                           best_C, score_function, tr_expected_complete_log_likelihood, val_accuracy,
                                           lock)

    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            # Start the load operations and mark each future with its URL
            future_to_C = {executor.submit(__parallel_kfold_computation, C, l, m, k, train_set, folds,
                                           max_epochs, threshold, runs, choose_by_vote,
                                           best_C, score_function, tr_expected_complete_log_likelihood, val_accuracy,
                                           lock): C for C in c_values}
            for future in concurrent.futures.as_completed(future_to_C):
                C = future_to_C[future]
                try:
                    future.result()
                except Exception as exc:
                    print('%r execution generated an exception: %s' % (C, exc))
                else:
                    print('Execution for %d is completed' % C)

    logging.info("Best C was " + str(best_C) + " val accuracy on folds was " + str(val_accuracy)
                 + " log likelihood was " + str(tr_expected_complete_log_likelihood))

    return best_C[0], tr_expected_complete_log_likelihood[0], val_accuracy[0]


def classification_score(model, dataset, compute_confusion=False, choose_by_vote=False):

    acc = 0.
    if compute_confusion:
        confusion = np.zeros((model.K, model.K), dtype='int32')  # predicted x actual

    for sample in dataset:
        predictions, predicted_votes = model.predict(sample, print_votes=False)
        predicted_root = predictions[0]

        predicted = predicted_votes if choose_by_vote else predicted_root

        if predicted == sample.y:
            acc = acc + 1.0
            if compute_confusion:
                confusion[sample.y, sample.y] = confusion[sample.y, sample.y] + 1
        elif compute_confusion:
            confusion[predicted, sample.y] = confusion[predicted, sample.y] + 1

    acc = (acc / len(dataset))*100.0

    if compute_confusion:
        return acc, confusion
    else:
        return acc


def transduction_score(model, dataset, choose_by_vote=False):  # last param needed for code reuse
    N = 0.
    correct = 0.

    for sample in dataset:
        id_to_node, Un, In, X, Y, Pos, Parents = sample.dag_ordering()
        N = N + Un
        predictions, tree_votes = model.predict(sample)

        correct += np.sum(predictions == Y)
        '''
        for u in range(0, Un):
            node_u = id_to_node[u]
            if predictions[u] == Y[u]:
                correct = correct + 1
        '''
    return (correct/N)*100.0







