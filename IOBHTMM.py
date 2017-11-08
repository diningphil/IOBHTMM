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

import numpy as np
import pickle
import time


def current_milli_time():
    return int(round(time.time() * 1000))


class IOBHTMM:
    POSITIONAL = 'positional'
    FULL = 'full'
    EPS = np.float64(0.00001)  # ADDITIVE SMOOTHING

    def __init__(self, l, c, m, k, stationarity='positional'):
        """
        An IOBHTMM model with a multinomial input/output alphabet
        :param l: the max outdegree of the tree
        :param c: the number of hidden states
        :param m: dimension of input alphabet
        :param k: dimension of output alphabet
        """
        self.L = l
        self.C = c  # model positional state transitions adding a special state, which encodes null children.
        self.M = m
        self.K = k
        self.stationarity = stationarity

        # Initialisation of the model's parameters.
        # Notice: the sum-to-1 requirement has been naively satisfied.

        if stationarity == self.POSITIONAL:
            self.priors = np.empty((self.L, self.M, self.C))
            for l1 in range(0, self.L):
                for m in range(0, self.M):
                    pr = np.random.uniform(size=self.C)
                    pr = pr / np.sum(pr)
                    self.priors[l1, m, :] = pr

            self.state_trans = np.empty((self.L, self.M, self.C, self.C + 1),
                                        dtype='float32')  # j has C+1 values (handle i0)
            # "l" stands for the position of the child
            # the former "c" stands for the "i" index, the latter stands for "j"

            for l1 in range(0, self.L):
                for m in range(0, self.M):
                    for j in range(0, self.C + 1):
                        pst = np.random.uniform(size=self.C)
                        pst = pst / np.sum(pst)
                        self.state_trans[l1, m, :, j] = pst

        elif stationarity == self.FULL:
            self.priors = np.empty((self.M, self.C))
            for m in range(0, self.M):
                pr = np.random.uniform(size=self.C)
                pr = pr / np.sum(pr)
                self.priors[m, :] = pr

            self.state_trans = np.empty((self.M, self.C, self.C + 1))  # j has C+1 values (handle i0)
            # the former "c" stands for the "i" index, the latter stands for "j"

            for m in range(0, self.M):
                for j in range(0, self.C + 1):
                    pst = np.random.uniform(size=self.C)
                    pst = pst / np.sum(pst)
                    self.state_trans[m, :, j] = pst

        else:
            raise Exception("Stationarity parameter not valid")

        sp = np.random.uniform(size=self.L)
        sp = sp / np.sum(sp)
        self.switching_parents = sp

        self.emissions = np.empty((self.M, self.K, self.C))
        for m in range(0, self.M):
            for i in range(0, self.C):
                em = np.random.uniform(size=self.K)
                em = em / np.sum(em)
                self.emissions[m, :, i] = em

    def get_parameters(self):
        """
        Return model's parameters
        :return: priors, state_transitions, emissions, switching_parents
        """
        return self.priors, self.state_trans, self.emissions, self.switching_parents


    def predict(self, node, print_votes=False):
        """
        Updates the entire tree's "prediction" and "state" attributes with the most likely output and hidden state
        :param print_votes:
        :param node: the root of the tree, where each y is unknown
        :return: root_output, vote_output
        """
        votes = np.zeros(self.K)

        id_to_node, Un, In, X, Y, Pos, Parents = node.dag_ordering()
        deltas = np.zeros((Un, self.C))
        gamma = np.zeros((Un, self.C, 2), dtype='int32')  # Indices of child state and emission

        states = np.zeros(Un, dtype='int32')
        predictions = np.zeros(Un, dtype='int32')

        # --------------------------------------------------------------------------#
        #                           Deltas on leaves                                #
        # --------------------------------------------------------------------------#
        if self.stationarity == self.POSITIONAL:
            deltas[In:Un, :] = self.priors[Pos[In:Un], X[In:Un], :]
        else:
            deltas[In:Un, :] = self.priors[X[In:Un], :]

        # --------------------------------------------------------------------------#
        #                       Deltas on internal nodes                            #
        # --------------------------------------------------------------------------#
        U_levels, Children_pos_levels, \
        Children_ids_levels, Null_Pos_levels, U_children_levels, U_null_levels, tree_depth = \
            node.get_internal_nodes_by_level()

        for level in range(tree_depth - 1, -1, -1):
            u_level = np.array(U_levels[level], dtype='int')
            u_null_level = np.array(U_null_levels[level], dtype='int')
            u_children_level = np.array(U_children_levels[level], dtype='int')

            null_pos_level = np.array(Null_Pos_levels[level], dtype='int')
            children_pos_level = np.array(Children_pos_levels[level], dtype='int')
            children_ids_level = np.array(Children_ids_levels[level], dtype='int')

            # for non-null children
            if self.stationarity == self.POSITIONAL:
                sp = np.reshape(self.switching_parents[children_pos_level],
                                (len(self.switching_parents[children_pos_level]), 1))  # [?x1]

                tmp = deltas[children_ids_level, :] * sp  # [?xC]

                tmp_bcast = np.reshape(tmp, (tmp.shape[0], 1, self.C))  # broadcast for each i  [?x1xC]

                increment = np.multiply(self.state_trans[children_pos_level, X[u_children_level], :, 0:self.C],
                                        tmp_bcast)  # [?xCxC]
            else:
                sp = np.reshape(self.switching_parents[children_pos_level],
                                (len(self.switching_parents[children_pos_level]), 1))

                tmp = deltas[children_ids_level, :] * sp  # [?xC]

                tmp_bcast = np.reshape(tmp, (tmp.shape[0], 1, self.C))  # broadcast for each i  [?x1xC]

                state_trans_bcast = self.state_trans[X[u_children_level], :, 0:self.C]
                increment = np.multiply(state_trans_bcast, tmp_bcast)  # [?xCxC]

            # need to broadcast over K dimension
            increment_bcast = np.reshape(increment, (increment.shape[0], 1, self.C, self.C))  # [?x1xCxC]

            # emissions does not depend on i, should be reused (which means..broadcast)
            emissions_bcast = np.reshape(self.emissions[X[children_ids_level], :, :],
                                         (len(children_ids_level), self.K, 1, self.C))

            takeMaxIndexes_new = np.multiply(emissions_bcast, increment_bcast)  # [?xKxCxC]

            # leave KxC (where C stands for index j) in the end
            takeMaxIndexes_swapped = np.swapaxes(takeMaxIndexes_new, 1, 2)

            maxIndexesReshaped = takeMaxIndexes_swapped.reshape((len(children_ids_level) * self.C, self.K * self.C))
            # ?x(K*C)

            maxIndices = maxIndexesReshaped.argmax(axis=1)  # dim [(?xK)x1] # tuples

            best_tuples_tmp = np.asarray(np.unravel_index(maxIndices, (self.K, self.C))).T  # (?xC)x2

            best_tuples_tmp = np.reshape(best_tuples_tmp, (len(children_ids_level), self.C, 2))
            # ?xCx2  ( C stands for each i ), before was ?x2 for each i

            i_best = best_tuples_tmp[:, :, 1]  # ?xC --> mi dice qual'e il j migliore dato i
            y_best = best_tuples_tmp[:, :, 0]  # ?xC --> mi dice qual'e il y migliore dato i

            gamma[children_ids_level, :, 0] = i_best
            gamma[children_ids_level, :, 1] = y_best

            # per applicare tutti gli i_best per ogni i, devo per forza fare l'unrolling.
            children_ids_repeated = np.tile(children_ids_level, self.C)
            Pos_children_repeated = Pos[children_ids_repeated]
            sp_children_repeated = np.tile(self.switching_parents[children_pos_level], self.C)  # (?xC)

            i_repeated = np.ravel(np.tile(np.arange(self.C), (i_best.shape[0], 1)),
                                  'F')  # (?xC) 00...011..122..2..(C-1)...(C-1)
            i_best_unrolled = np.ravel(i_best, 'F')  # (Cx?) ( unravel column by column )

            u_children_level_rep = np.tile(u_children_level, self.C)

            if self.stationarity == self.POSITIONAL:

                tmp = np.multiply(sp_children_repeated,
                                  self.state_trans[Pos_children_repeated, X[u_children_level_rep],
                                                   i_repeated, i_best_unrolled])
                tmp = np.reshape(tmp, (self.C, i_best.shape[0]))

                # intuitively i have to obtain a "summation" term for each u in u_level
                # raggruppando quelli che hanno stesso valore di u_children_level
                # summation += np.sum(tmp, axis=1)
                summation = tmp  # Cxu_children_level

            else:

                tmp = np.multiply(sp_children_repeated,
                                  self.state_trans[X[u_children_level_rep], i_repeated, i_best_unrolled])
                tmp = np.reshape(tmp, (self.C, i_best.shape[0]))
                # summation += np.sum(tmp, axis=1)
                summation = tmp  # Cxu_children_level

            # for null children
            summation_null = None
            if len(null_pos_level) != 0:
                sp_null_repeated = np.tile(self.switching_parents[null_pos_level], self.C)  # (null_childrenxC)

                # null_pos = np.flatnonzero(null_pos_level) NON E' PIU' BOOLEAN
                null_pos = null_pos_level

                no_of_null_nodes = len(null_pos)
                i_null_repeated = np.ravel(np.tile(np.arange(self.C), (no_of_null_nodes, 1)), 'F')

                null_pos = np.tile(null_pos, self.C)

                u_null_level_rep = np.tile(u_null_level, self.C)

                if self.stationarity == self.POSITIONAL:

                    tmp = np.multiply(sp_null_repeated,
                                      self.state_trans[null_pos, X[u_null_level_rep], i_null_repeated, self.C])
                    tmp = np.reshape(tmp, (self.C, no_of_null_nodes))
                    # summation += np.sum(tmp, axis=1)
                    summation_null = tmp
                else:
                    tmp = np.multiply(sp_null_repeated,
                                      self.state_trans[X[u_null_level_rep], i_null_repeated, self.C])
                    tmp = np.reshape(tmp, (self.C, no_of_null_nodes))
                    # summation += np.sum(tmp, axis=1)
                    summation_null = tmp

                    # prod = np.multiply(prod, 1) does not change

            y_best_unrolled = np.ravel(y_best, 'F')  # (Cx?) ( unravel column by column )

            tmp = np.multiply(self.emissions[X[children_ids_repeated], y_best_unrolled, i_best_unrolled],
                              deltas[children_ids_repeated, i_best_unrolled])

            tmp = np.reshape(tmp, (self.C, i_best.shape[0]))

            # prod = np.multiply.reduce(tmp, axis=1)
            prod = tmp

            # summation, summation_null and prod have shape Cxi_best.shape[0] or Cxno_of_null_nodes
            # I must separate again info for each node u in u_level.
            # the structure of u_children_level tells us how many (consecutive) terms belong to a specific u

            unique, counts = np.unique(u_children_level, return_counts=True)
            counts = np.cumsum(counts)  # now I have indices telling me the exact separation
            summation_list = np.split(summation, counts, axis=1)  # list of ndarray
            # summation_list = summation_list[:-1]  # do not consider the last element (Cx0)
            prod_list = np.split(prod, counts, axis=1)  # list of ndarray
            # prod_list = prod_list[:-1]  # do not consider the last element (Cx0)

            if summation_null is not None:
                unique, counts = np.unique(u_null_level, return_counts=True)
                counts = np.cumsum(counts)  # now I have indices telling me the exact separation
                summation_null_list = np.split(summation_null, counts, axis=1)  # list of ndarray
                # summation_null_list = summation_null_list[:-1]  # do not consider the last element (Cx0)

            # --------------------------------------------------------------------------#
            #                  Now updating Deltas on internal nodes                    #
            # --------------------------------------------------------------------------#
            for i in range(0, len(u_level)):
                if summation_null is not None:
                    deltas[u_level[i], :] = \
                        np.multiply(np.sum(summation_list[i], axis=1) + np.sum(summation_null_list[i], axis=1),
                                    np.multiply.reduce(prod_list[i], axis=1))
                else:
                    deltas[u_level[i], :] = \
                        np.multiply(np.sum(summation_list[i], axis=1),
                                    np.multiply.reduce(prod_list[i], axis=1))

        # ------ Compute best state and emission label for the root ------ #
        takeMaxIndexes = np.multiply(self.emissions[X[0], :, :], np.transpose(deltas[0, :]))  # [KxC]
        maxIdxTuple = np.unravel_index(takeMaxIndexes.argmax(), takeMaxIndexes.shape)
        y1_best, i1_best = maxIdxTuple[0], maxIdxTuple[1]

        states[0] = i1_best
        predictions[0] = y1_best

        votes[predictions[0]] = votes[predictions[0]] + 1

        if Un == 1:
            return np.array([i1_best]), np.array([i1_best])

        # ------ Compute best state and emission label for the other nodes ------ #

        # Downward recursion to propagate the optimal node at the root to the internal nodes
        for level in range(1, tree_depth + 1):

            if level == tree_depth:
                u_level = np.array(range(In, Un))
            else:
                u_level = U_levels[level]

            states[u_level] = gamma[u_level, states[Parents[u_level]], 0]
            predictions[u_level] = gamma[u_level, states[Parents[u_level]], 1]
            votes[predictions[u_level]] = votes[predictions[u_level]] + 1

        if print_votes:
            print("Votes are: ", votes)

        return predictions, np.argmax(votes)

    def train(self, dataset, threshold=0, max_epochs=30, print_score=False, pred_set=None, score_fun=None):
        """
        Training algorithm: updates model's parameters
        :param dataset: a list of nodes which are root of the N trees in the dataset
        :param threshold: stopping criterion wrt log Lc
        :param max_epochs: the max number of epochs
        :param score_fun:
        :param pred_set:
        :param print_score:
        :return: expected complete log likelihood's history
        """
        N = len(dataset)

        # EM algorithm
        epoch = 1
        expected_log_likelihood = -np.inf
        training_log_Lc_history = []
        delta = threshold + 1

        batch_dim = 250
        batches = int(np.floor(N / batch_dim))
        batches = 1 if batches == 0 else batches
        batches = batches + 1 if N % batches != 0 else batches

        # num = numerator | den = denominator
        if self.stationarity == self.POSITIONAL:
            num_priors = np.full((self.L, self.M, self.C), self.EPS)
            den_priors = np.full((self.L, self.M, self.C), np.multiply(self.C, self.EPS))
            num_state_trans = np.full((self.L, self.M, self.C, self.C + 1), self.EPS)
            den_state_trans = np.full((self.L, self.M, self.C, self.C + 1), np.multiply(self.C, self.EPS))
        elif self.stationarity == self.FULL:
            num_priors = np.full((self.M, self.C), self.EPS)
            den_priors = np.full((self.M, self.C), np.multiply(self.C, self.EPS))
            num_state_trans = np.full((self.M, self.C, self.C + 1), self.EPS)
            den_state_trans = np.full((self.M, self.C, self.C + 1), np.multiply(self.C, self.EPS))
        else:
            raise Exception("Stationarity parameter not valid")

        num_switching_parents = np.full(self.L, self.EPS)
        den_switching_parents = np.full(self.L, np.multiply(self.L, self.EPS))
        num_emissions = np.full((self.M, self.K, self.C), self.EPS)
        den_emissions = np.full((self.M, self.K, self.C), np.multiply(self.K, self.EPS))

        print("Training with C = ", self.C)  # , ". Epoch: ", epoch)

        while epoch <= max_epochs and delta > threshold:

            # debugging constants for average computation time
            timeEpoch = current_milli_time()
            totalE = 0
            totalM = 0
            totalEll = 0

            # needed to compute the delta
            old_expected_log_likelihood = expected_log_likelihood
            expected_log_likelihood = 0.

            for mini_batch in range(0, batches):  # batches+1, in case there is a last smaller mini batchs

                # t = current_milli_time()

                # ######################################################################### #
                #                                                                           #
                #        Preparing auxiliary data structures for mini batch computing       #
                #                                                                           #
                # ######################################################################### #

                # print("Batch n ", mini_batch+1)
                leaves_n_tree = []  # n-th element contains the number of leaves of the n-th tree
                first_n_Un = []  # n-th element contains the sum of the first "n" trees' nodes
                X_batch = np.array([], dtype='int16')  # concatenate the X of each tree
                Y_batch = np.array([], dtype='int16')  # concatenate the Y of each tree
                Pos_batch = np.array([], dtype='int16')  # concatenate the Pos of each tree
                Parents_batch = np.array([], dtype='int16')  # concatenate the Parents of each tree (shift Us)

                # aux data structure for batch processing per level
                u_by_level_batch = []  # concatenate Us for each level and tree (shift Us by CURRENT total_nodes)

                # for these holds the same concept of concatenation by level, below we describe what is concatenated
                children_positions_batch = []  # each node u has an array of positions of its children
                children_ids_batch = []  # each node u has an array of IDs of its children
                null_positions_batch = []  # each node u has an array of null positions
                u_rep_for_children_batch = []  # each node u is repeated for the number of children (u gets shifted)
                u_rep_for_null_batch = []  # each node u is repeated for the number of NULL children (u gets shifted)
                max_depth = 1  # I need to keep track of the max depth existing

                batch_threshold = N - mini_batch * batch_dim
                batch_size = batch_threshold if batch_threshold < batch_dim else batch_dim

                total_nodes = 0
                start_n = batch_size * mini_batch

                for n in range(0, batch_size):

                    # ######################################################################### #
                    #                         Notation:                                         #
                    #                            u: id of a node                                #
                    #                            In: # of internal nodes (of a tree)            #
                    #                            Un: # of total nodes    (of a tree)            #
                    #                                                                           #
                    # ######################################################################### #

                    id_to_node, Un, In, X, Y, Pos, Parents = dataset[start_n + n].dag_ordering()

                    u_by_level, children_positions, \
                    children_ids, null_positions, u_rep_for_children, u_rep_for_null, tree_depth = \
                        dataset[start_n + n].get_internal_nodes_by_level()  # requires a dag ordering

                    if max_depth < tree_depth:
                        max_depth = tree_depth

                    # this loop simply concatenates level. if a level has never been reached, we "add it" (append)
                    for level in range(0, tree_depth):

                        if len(u_by_level_batch) <= level:  # this checks holds for all the others
                            u_by_level_batch.append(
                                np.array(u_by_level[level]) + total_nodes)

                            children_positions_batch.append(children_positions[level])
                            children_ids_batch.append(
                                np.array(children_ids[level]) + total_nodes)
                            null_positions_batch.append(null_positions[level])
                            u_rep_for_children_batch.append(
                                np.array(u_rep_for_children[level]) + total_nodes)
                            u_rep_for_null_batch.append(
                                np.array(u_rep_for_null[level]) + total_nodes)

                        else:
                            u_by_level_batch[level] = \
                                np.concatenate([u_by_level_batch[level], np.array(u_by_level[level]) + total_nodes])

                            children_positions_batch[level].extend(children_positions[level])

                            children_ids_batch[level] = \
                                np.concatenate([children_ids_batch[level],
                                                np.array(children_ids[level]) + total_nodes])

                            null_positions_batch[level].extend(null_positions[level])

                            u_rep_for_children_batch[level] = \
                                np.concatenate([u_rep_for_children_batch[level],
                                                np.array(u_rep_for_children[level]) + total_nodes])

                            u_rep_for_null_batch[level] = \
                                np.concatenate([u_rep_for_null_batch[level],
                                                np.array(u_rep_for_null[level]) + total_nodes])

                    leaves_n_tree.append(Un - In)

                    Parents_batch = np.concatenate((Parents_batch, Parents + total_nodes))

                    total_nodes = total_nodes + Un

                    first_n_Un.append(Un)
                    X_batch = np.concatenate((X_batch, X))
                    Y_batch = np.concatenate((Y_batch, Y))

                    Pos_batch = np.concatenate((Pos_batch, Pos))

                leaves_n_tree = np.array(leaves_n_tree)

                # now accumulate Un to obtain what we want
                first_n_Un = np.cumsum(
                    np.array(first_n_Un))  # now -> ith element: the total sum of nodes of the first i trees' nodes

                if batches <= 0:
                    Un_total = first_n_Un
                else:
                    Un_total = first_n_Un[-1]

                # now compute boolean masks to select specific subsets of nodes
                mask_leaves = np.zeros(Un_total, dtype='bool')
                mask_In = np.zeros(Un_total, dtype='bool')
                mask_all_but_root = np.ones(Un_total, dtype='bool')

                for n in range(0, batch_size):
                    mask_leaves[first_n_Un[n] - leaves_n_tree[n]: first_n_Un[n]] = True
                    if n == 0:
                        mask_In[: first_n_Un[0] - leaves_n_tree[0]] = True
                        mask_all_but_root[0] = False
                    else:
                        mask_In[first_n_Un[n - 1]: first_n_Un[n] - leaves_n_tree[n]] = True
                        mask_all_but_root[first_n_Un[n - 1]] = False

                # print("Helper structure prep in ", timeE1-t, " ms")

                # ######################################################################### #
                #                                                                           #
                #                       E-STEP  (batch version, but for a loop)             #
                #                                                                           #
                # ######################################################################### #

                timeE1 = current_milli_time()

                betas = np.zeros((Un_total, self.C))
                betas_aux = np.zeros((Un_total, self.L, self.C))

                posteriors = np.zeros((Un_total, self.C))
                state_transition_posteriors = np.zeros((Un_total, self.L, self.C, self.C + 1))

                # --------------------------------------------------------------------------#
                #                     Betas on leaves (batch version)                       #
                # --------------------------------------------------------------------------#
                if self.stationarity == self.POSITIONAL:
                    priors = self.priors[Pos_batch[mask_leaves], X_batch[mask_leaves], :]
                else:
                    priors = self.priors[X_batch[mask_leaves], :]
                emissions = self.emissions[X_batch[mask_leaves], Y_batch[mask_leaves], :]

                Nu = np.sum(np.multiply(emissions, priors), axis=1)

                betas[mask_leaves, :] = np.divide(np.multiply(priors, emissions),
                                                  np.reshape(Nu, (len(Nu), 1)))

                # --------------------------------------------------------------------------#
                #           Betas on int. nodes (batch version, per level, bottom-up)       #
                # --------------------------------------------------------------------------#
                for level in range(max_depth - 1, -1, -1):  # depth 1 (level 0) included
                    # intuitively, levels are in range from 0 to max_depth - 1

                    u_level = np.array(u_by_level_batch[level], dtype='int')
                    u_null_level = np.array(u_rep_for_null_batch[level], dtype='int')
                    u_children_level = np.array(u_rep_for_children_batch[level], dtype='int')

                    null_pos_level = np.array(null_positions_batch[level], dtype='int')
                    children_pos_level = np.array(children_positions_batch[level], dtype='int')
                    children_ids_level = np.array(children_ids_batch[level], dtype='int')

                    X_level = X_batch[u_level]
                    Y_level = Y_batch[u_level]

                    if self.stationarity == self.POSITIONAL:

                        # --------------- Auxiliary Beta (null case) --------------- #
                        betas_aux[u_null_level, null_pos_level, :] = \
                            self.state_trans[null_pos_level, X_batch[u_null_level], :, self.C]
                        # ---------------------------------------------------------- #

                        # children case
                        right_factor = np.reshape(betas[children_ids_level, :],
                                                  (len(children_ids_level), 1, self.C))
                        # needed to broadcast same matrix across "i" values (the 1)

                        summation_body = np.multiply(
                            self.state_trans[children_pos_level, X_batch[u_children_level], :, 0:self.C],
                            right_factor)  # broadcasting -->  NodesPerLevelxCxC

                    else:  # full stationarity

                        # null state case
                        betas_aux[u_null_level, null_pos_level, :] = \
                            self.state_trans[X_batch[u_null_level], :, self.C]

                        # children case
                        right_factor = np.reshape(betas[children_ids_level, :],
                                                  (len(children_ids_level), 1, self.C))  # NodesPerLevx1xC
                        # needed to broadcast same matrix across "i" values (the 1)

                        summation_body = np.multiply(
                            self.state_trans[X_batch[u_children_level], :, 0:self.C],
                            right_factor)  # broadcasting --> NodesPerLevelxCxC

                    # --------------- Auxiliary Beta --------------- #
                    betas_aux[u_children_level, children_pos_level, :] = np.sum(summation_body, axis=2)
                    # ---------------------------------------------- #

                    # dot for N dimensions it is a sum product over the last axis of a and the second-to-last of b
                    betas_right_factor = np.dot(self.switching_parents, betas_aux[u_level, :, :])  # NodesPerLevelxC
                    emissions = self.emissions[X_level, Y_level, :]  # NodesPerLevelxC

                    Nu = np.sum(np.multiply(emissions, betas_right_factor), axis=1)  # shape (NodesPerLevel,)
                    Nu = np.reshape(Nu, (len(Nu), 1))  # broadcast per row: NodesPerLevelx1 (column vec)

                    # --------------- Beta --------------- #
                    betas[u_level, :] = np.divide(np.multiply(emissions, betas_right_factor), Nu)
                    # ------------------------------------ #

                # --------------------------------------------------------------------------#
                #           Eps state transition (batch version, per level, top-down)       #
                # --------------------------------------------------------------------------#

                only_roots = np.logical_not(mask_all_but_root)

                # --------------- State occupancy posterior (eps) for root --------------- #
                posteriors[only_roots, :] = betas[only_roots, :]
                # ---------------------------------------------------------------------------- #

                for level in range(1, max_depth + 1):  # root not included
                    if level == max_depth:
                        u_level = np.nonzero(mask_leaves)[0]
                    else:
                        u_level = np.array(u_by_level_batch[level])

                    Parents_level = Parents_batch[u_level]
                    Pos_level = Pos_batch[u_level]

                    # ----------- Children case ----------- #

                    beta_chl_u_level = betas[u_level, :]  # [NodePerLevelxC] does not depend on i

                    sp_l = self.switching_parents[Pos_level]  # (NodeperLevel,) does not depend on i

                    eps_u = posteriors[Parents_level, :]  # [NodePerLevelxC]

                    if self.stationarity == self.POSITIONAL:
                        # do not consider state i0
                        st_trans = self.state_trans[Pos_level, X_batch[Parents_level], :, 0:self.C]
                    else:
                        st_trans = self.state_trans[X_batch[Parents_level], :, 0:self.C]  # do not consider state i0

                    denominator = np.dot(self.switching_parents,
                                         betas_aux[Parents_level, :, :])  # [NodePerLevel*C]

                    denominator = np.reshape(denominator, (len(u_level), self.C, 1))

                    # --------------- State transition posterior (eps_u_ch) --------------- #
                    state_transition_posteriors[Parents_level, Pos_level, :, 0:self.C] = np.divide(
                        np.multiply(
                            np.multiply(
                                np.multiply(np.reshape(beta_chl_u_level, (len(u_level), 1, self.C)), st_trans),
                                np.reshape(eps_u, (len(u_level), self.C, 1))),
                            np.reshape(sp_l, (len(u_level), 1, 1))), denominator)  # [NodePerLevelxCxC]
                    # --------------------------------------------------------------------- #

                    # ----------- Null Children case ----------- #

                    # I need null positions' info for each parent --> i have it, in the above level!
                    parent_null_level = np.array(u_rep_for_null_batch[level - 1])
                    parent_null_pos_level = np.array(null_positions_batch[level - 1])

                    eps_u = posteriors[parent_null_level, :]  # RepParentsxC rep stands for repeated
                    sp_l = self.switching_parents[parent_null_pos_level]  # RepParentsx1

                    if self.stationarity == self.POSITIONAL:
                        # consider only state i0, ParentsPerLevelxC
                        st_trans_null = self.state_trans[parent_null_pos_level, X_batch[parent_null_level], :, self.C]
                    else:
                        st_trans_null = self.state_trans[X_batch[parent_null_level], :, self.C]
                        # consider only state i0, ParentsPerLevelxC

                    # as for internal nodes' beta
                    denominator = np.dot(np.reshape(self.switching_parents, (1, self.L)),
                                         betas_aux[parent_null_level, :, :])  # [RepParents*C]

                    num = np.multiply(st_trans_null, eps_u)
                    num = np.multiply(num, np.reshape(sp_l, (1, len(sp_l), 1)))
                    # the first 1 stands "for each repeated parent"
                    # each col, corresponds to diff positions, must be multiplied by sp_l)

                    # --------------- State transition posterior (eps_u_ch) NULL CASE --------------- #
                    state_transition_posteriors[parent_null_level, parent_null_pos_level, :, self.C] = \
                        np.divide(num, denominator)
                    # ------------------------------------------------------------------------------- #

                    # --------------------------------------------------------------------------#
                    #           Eps state occupancy (batch version, per level, top-down)       #
                    # --------------------------------------------------------------------------#

                    normalize = np.sum(state_transition_posteriors[Parents_level, Pos_level, :, :],
                                       axis=(1, 2))
                    normalize = np.reshape(normalize, (len(u_level), 1))

                    posteriors[u_level, :] = np.divide(
                        np.sum(state_transition_posteriors[Parents_level, Pos_level, :, 0:self.C], axis=1),
                        normalize)

                timeStepE = current_milli_time()
                totalE += timeStepE - timeE1

                # ######################################################################### #
                #                                                                           #
                #                                M-STEP                                     #
                #                                                                           #
                # ######################################################################### #

                # --------------------------------------------------------------------------#
                #                  M-step for null children (simple for loop)               #
                # --------------------------------------------------------------------------#
                for n in range(0, batch_size):

                    id_to_node, Un, In, X, Y, Pos, Parents = dataset[start_n + n].dag_ordering()

                    if n > 0:
                        tree_state_transition_posteriors = \
                            state_transition_posteriors[first_n_Un[n - 1]:first_n_Un[n], :, :, :]
                    else:  # n == 0
                        tree_state_transition_posteriors = \
                            state_transition_posteriors[:first_n_Un[0], :, :, :]

                    # Iterating over all parents === Internal Nodes!
                    for u in range(0, In):
                        node_u = id_to_node[u]

                        null_bool_arr, children_bool_arr, children_ids = node_u.get_children_info()

                        # M-step increments for null children
                        if self.stationarity == self.POSITIONAL:

                            num_state_trans[null_bool_arr, X[u], :, :] += \
                                tree_state_transition_posteriors[u, null_bool_arr, :, :]

                            den = np.sum(tree_state_transition_posteriors[u, null_bool_arr, :, :], axis=1)

                            den = np.reshape(den, (den.shape[0], 1, den.shape[1]))
                            den_state_trans[null_bool_arr, X[u], :, :] += den

                        else:  # full stationariety

                            num_state_trans[X[u], :, :] += \
                                np.sum(tree_state_transition_posteriors[u, null_bool_arr, :, :], axis=0)

                            den = np.sum(tree_state_transition_posteriors[u, null_bool_arr, :, :], axis=(0, 1))
                            den = np.reshape(den, (1, den.shape[0]))
                            den_state_trans[X[u], :, :] += den

                # --------------------------------------------------------------------------#
                #                           M-step for children                             #
                # --------------------------------------------------------------------------#
                leaves_pos = Pos_batch[mask_leaves]
                not_root_pos = Pos_batch[mask_all_but_root]
                leaves_x = X_batch[mask_leaves]
                not_root_parent_x = X_batch[Parents_batch[mask_all_but_root]]
                leaves_parents = Parents_batch[mask_leaves]
                not_root_parents = Parents_batch[mask_all_but_root]

                if self.stationarity == self.POSITIONAL:

                    # --------- PRIORS  --------- #
                    sum_num_pr = np.sum(state_transition_posteriors[leaves_parents,
                                        leaves_pos, :, 0:self.C], axis=1)
                    np.add.at(num_priors, (leaves_pos, leaves_x), sum_num_pr)

                    den = np.sum(sum_num_pr, axis=1)
                    np.add.at(den_priors, (leaves_pos, leaves_x), np.reshape(den, (len(den), 1)))
                    # --------------------------- #

                    # --------- STATE TRANSITION  --------- #
                    np.add.at(num_state_trans, (not_root_pos, not_root_parent_x),
                              state_transition_posteriors[not_root_parents, not_root_pos, :, :])

                    den = np.sum(state_transition_posteriors[not_root_parents, not_root_pos, :, :], axis=1)
                    np.add.at(den_state_trans, (not_root_pos, not_root_parent_x),
                              np.reshape(den, (den.shape[0], 1, den.shape[1])))
                    # ------------------------------------- #

                else:  # full stationariety
                    # NOTE: add.at can be slow here, more than in the POSITIONAL case.
                    # It may be useful to have a function that assumes an ordering to do
                    # the same thing, and then provide the data ordered.Copyright © 2007 Free Software Foundation, Inc. <https://fsf.org/>

                    # --------- PRIORS  --------- #
                    sum_num_pr = np.sum(state_transition_posteriors[leaves_parents, :, :, 0:self.C], axis=(1, 2))
                    np.add.at(num_priors, leaves_x, sum_num_pr)

                    den = np.sum(sum_num_pr, axis=1)
                    np.add.at(den_priors, leaves_x, np.reshape(den, (len(den), 1)))
                    # --------------------------- #

                    # --------- STATE TRANSITION  --------- #
                    sum_num_st = np.sum(state_transition_posteriors[not_root_parents, :, :, :], axis=1)
                    np.add.at(num_state_trans, not_root_parent_x, sum_num_st)

                    # den = np.sum(state_transition_posteriors[not_root_parents, :, :, :], axis=(1, 2))
                    den = np.sum(sum_num_st, axis=1)

                    # this is expensive!
                    den = np.reshape(den, (den.shape[0], 1, den.shape[1]))
                    np.add.at(den_state_trans, not_root_parent_x, den)
                    # ------------------------------------- #

                # -------- SWITCHING PARENTS -------- #
                # np.add.at is used to accumulate results for same values in the indexing array
                sum_num_sp = np.sum(state_transition_posteriors[not_root_parents, not_root_pos, :, :], axis=(1, 2))
                np.add.at(num_switching_parents, not_root_pos, sum_num_sp)

                den_switching_parents += np.sum(sum_num_sp)
                # ----------------------------------- #

                # --------- EMISSION ------------ #
                np.add.at(num_emissions, (X_batch, Y_batch), posteriors)

                tmp_post = np.reshape(posteriors, (posteriors.shape[0], 1, self.C))
                np.add.at(den_emissions, X_batch, tmp_post)
                # -------------------------------- #

                timeM = current_milli_time()
                totalM += np.subtract(timeM, timeStepE)

                # ######################################################################### #
                #                                                                           #
                #             Expected complete Log-Likelihood computation                  #
                #                                                                           #
                # ######################################################################### #

                expected_log_likelihood += np.sum(
                    np.multiply(posteriors, np.log(self.emissions[X_batch, Y_batch, :])))

                if self.stationarity == self.POSITIONAL:
                    expected_log_likelihood += np.sum(
                        np.multiply(posteriors[mask_leaves, :],
                                    np.log(self.priors[Pos_batch[mask_leaves], X_batch[mask_leaves], :])))
                else:
                    expected_log_likelihood += np.sum(
                        np.multiply(posteriors[mask_leaves, :], np.log(self.priors[X_batch[mask_leaves], :])))

                r = np.reshape(np.log(self.switching_parents), (1, self.L, 1, 1))
                expected_log_likelihood += np.sum(np.multiply(state_transition_posteriors[mask_In, :, :, :], r))

                # reshape is different by swapaxes! reshape scrambles the tensor if not used with care!
                if self.stationarity == self.POSITIONAL:
                    log = np.log(self.state_trans[:, X_batch[mask_In], :, :])
                    log = np.swapaxes(log, 0, 1)
                    expected_log_likelihood += np.sum(
                        np.multiply(state_transition_posteriors[mask_In, :, :, :], log))
                else:
                    log = np.log(self.state_trans[X_batch[mask_In], :, :])

                    log = np.swapaxes(np.reshape(log, (1, log.shape[0], self.C, self.C + 1)), 0, 1)
                    expected_log_likelihood += np.sum(
                        np.multiply(state_transition_posteriors[mask_In, :, :, :], log))

                totalEll += current_milli_time() - timeM

            # --------------------------------------------------------------------------#
            #                           End of the EM epoch                             #
            # --------------------------------------------------------------------------#

            print("C = ", self.C, " Expected complete log likelihood at iteration ", epoch, ": ",
                  expected_log_likelihood)
            training_log_Lc_history.append(expected_log_likelihood)

            delta = expected_log_likelihood - old_expected_log_likelihood

            # print the score according to computed parameters
            if print_score:
                print("Score is ", score_fun(pred_set))

            # Update parameters
            if delta > 0:
                np.divide(num_priors, den_priors, out=self.priors)  # element-wise division
                np.divide(num_state_trans, den_state_trans, out=self.state_trans)
                np.divide(num_emissions, den_emissions, out=self.emissions)
                np.divide(num_switching_parents, den_switching_parents, out=self.switching_parents)
                num_priors.fill(self.EPS)
                den_priors.fill(np.multiply(self.EPS, self.C))
                num_state_trans.fill(self.EPS)
                den_state_trans.fill(np.multiply(self.EPS, self.C))
                num_emissions.fill(self.EPS)
                den_emissions.fill(np.multiply(self.EPS, self.K))
                num_switching_parents.fill(self.EPS)
                den_switching_parents.fill(np.multiply(self.EPS, self.L))

            epoch = epoch + 1

            '''
            print("Epoch in ", current_milli_time() - timeEpoch)
            print("Avg E in ", totalE / N)
            print("Avg M in ", totalM / N)
            print("Avg Ell in ", totalEll / N)
            '''

        return training_log_Lc_history

    def save_model(self, filename):
        """
        Store the model into a file
        :param filename:
        :return:
        """
        with open(filename, 'wb') as f:
            pickle.dump(
                [self.L, self.C, self.M, self.K, self.stationarity, self.priors, self.state_trans, self.emissions,
                 self.switching_parents], f)

    def load_model(self, filename):
        """
        Load the model into a file
        :param filename:
        :return:
        """
        with open(filename, 'rb') as f:
            self.L, self.C, self.M, self.K, self.stationarity, self.priors, self.state_trans, \
            self.emissions, self.switching_parents = pickle.load(f)
