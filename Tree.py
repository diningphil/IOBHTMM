import numpy as np


class Node(object):
    """
    This class represents a tree with an id called "u",
    a generic input value and a generic output value.
    It is used to preprocess an input-output tree sample from the dataset
    """

    def __init__(self, x, y, l, parent, position):
        """
        :param x: the input value
        :param y: the output value
        :param parent: the parent Node
        :param l: the max outdegree. It is the user's responsibility not to add too many children
        :param position: the position of the node wrt the father in {0..L-1}
        """
        self.dagDone = False
        self.dagResults = None

        self.L = l  # The max outdegree
        self.u = -1  # The id of the node
        self.x = x  # The input label
        self.y = y  # The output label
        self.prediction = None  # Output, used for prediction
        self.state = None  # Hidden state, filled by a model's prediction at test time
        self.parent = parent  # The parent Node
        self.position_as_child = position
        self.children = []

        if parent is None:
            self.depth = 1
        else:
            self.depth = parent.depth + 1

        # These are updated when dag_ordering is called
        self.id_to_node = None
        self.number_of_nodes = 0
        self.number_of_leaves = 0

        self.X = None
        self.Y = None
        self.Pos = None
        self.Parents = None

    def children_size(self):
        return len([1 for el in self.children if el is not None])

    def add_child(self, x, y, pos):  # The position in self.children does not reflect "l"
        """
        Add a node to the children.
        :param x:
        :param y:
        :param pos: the position of the node wrt the parent
        :return:
        """
        if len(self.children) == 0:
            self.children = [None] * self.L

        self.children[pos] = Node(x, y, self.L, self, pos)

    def get_lth_child(self, l):
        """
        :param l: position in the children array
        :return: the l-th child or None if it is not present
        """
        # filtered = filter(lambda x: x.position_as_child == l, self.children)
        # return None if len(filtered) == 0 else filtered[0]
        return self.children[l]

    def get_children_info(self):
        children_ids = []
        a = np.zeros(self.L, dtype=bool)
        for i in range(0, self.L):
            if self.children[i] is None:
                a[i] = True
            else:
                children_ids.append(self.get_lth_child(i).u)
        if self.is_a_leaf():
            return []
        else:
            return a, np.logical_not(a), np.array(children_ids)

    def get_null_children_positions(self):
        if self.is_a_leaf():
            return []
        return [i for i in range(0, self.L) if self.children[i] is None]

    def set_u(self, u):
        self.u = u

    def is_a_leaf(self):
        return len(self.children) == 0

    def complete_tree(self, m, k):
        """
        If an internal node has less than L children, it adds them with special x,y values
        :param m: the M parameter of the IO-BHTMM model
        :param k: the K parameter of the IO-BHTMM model
        :return:
        """
        visited_children = []  # The stack, elements are (position, node)

        if not self.is_a_leaf():
            for i in range(0, self.L):
                if self.get_lth_child(i) is None:
                    self.add_child(m, k, i)

        for child in self.children:
            visited_children.append(child)

        while len(visited_children) != 0:
            first_node = visited_children.pop(0)  # pos goes from {0..L-1}

            if not first_node.is_a_leaf():
                for i in range(0, self.L):
                    if first_node.get_lth_child(i) is None:
                        first_node.add_child(m, k, i)

                for child in first_node.children:
                    visited_children.append(child)

    def dag_ordering(self):
        """
        Assign an incremental index to each node, from left to right and from level to level
        PRECONDITION: must be called on the root and each internal node must have L children
        POSTCONDITION: the ordering is the same of the IO_BHTMM paper
        :return: map id_to_node, Un, In, map id_to_parentId
        """
        if self.dagDone:
            return self.dagResults

        self.dagDone = True

        number_of_leaves = 0
        number_of_nodes = 1
        idx = 0
        self.set_u(idx)
        visited_children = []  # The stack, elements are (position, node)
        id_to_node = [self]
        leaves = []  # Contains the leaves to which an index will be given only at the end

        X = [self.x]
        Y = [self.y]
        Pos = [-1]
        Parents = [-1]

        if len(self.children) == 0:
            number_of_leaves = 1
            return id_to_node, number_of_nodes, number_of_leaves, np.array([self.x]), np.array([self.y]), \
                   np.array([-1])

        elif len(self.children) != self.L:
            raise Exception('Each int. node should have L children. \
            Use complete_tree to insert fake nodes with special values')

        pos = 0
        for child in self.children:
            if child is not None:
                visited_children.append((pos, child))
                number_of_nodes = number_of_nodes + 1
            pos = pos + 1

        while len(visited_children) != 0:
            pos, first_node = visited_children.pop(0)  # pos goes from {0..L-1}

            if first_node is not None:

                if first_node.is_a_leaf():
                    leaves.append((pos, first_node))  # process later

                else:
                    idx = idx + 1
                    first_node.set_u(idx)
                    id_to_node.append(first_node)

                    X.append(first_node.x)
                    Y.append(first_node.y)
                    Pos.append(pos)
                    Parents.append(first_node.parent.u)

                    pos = 0
                    for child in first_node.children:
                        if child is not None:
                            visited_children.append((pos, child))
                            number_of_nodes = number_of_nodes + 1
                        pos = pos + 1

        for pos, leaf_node in leaves:
            idx = idx + 1
            leaf_node.set_u(idx)
            number_of_leaves = number_of_leaves + 1
            id_to_node.append(leaf_node)

            X.append(leaf_node.x)
            Y.append(leaf_node.y)
            Pos.append(pos)
            Parents.append(leaf_node.parent.u)

        # Update the state
        self.id_to_node = id_to_node
        self.number_of_nodes = number_of_nodes
        self.number_of_leaves = number_of_leaves

        self.X = np.array(X)
        self.Y = np.array(Y)
        self.Pos = np.array(Pos)
        self.Parents = np.array(Parents)

        Un = number_of_nodes
        In = number_of_nodes - number_of_leaves
        self.dagResults = self.id_to_node, Un, In, self.X, self.Y, self.Pos, self.Parents

        return self.dagResults

    def get_internal_nodes_by_level(self):
        # PRECONDITION: a dag ordering has already been done

        U_levels = []

        Children_ids_levels = []
        Children_pos_levels = []
        Null_pos_levels = []
        U_children_levels = []  # each u must be repeated for the number of its children
        U_null_levels = []  # each u must be repeated for the number of its null positions

        # Dag ordering corresponds to a BFS: hence the ordering in the depth for internal nodes
        In = self.number_of_nodes - self.number_of_leaves

        last_depth = 1
        u = 0

        while u < In:

            u_level = []

            children_pos_level = []
            children_ids_level = []
            null_pos_level = []
            u_children_level = []  # each u must be repeated for the number of its children
            u_null_level = []  # each u must be repeated for the number of its null positions

            node_u = self.id_to_node[u]

            while last_depth == node_u.depth and u < In:

                null_bool_arr, children_bool_arr, children_ids = node_u.get_children_info()

                null_pos_arr = np.where(null_bool_arr)[0].tolist()  # needed a list
                null_pos_len = len(null_pos_arr)
                children_ids_len = len(children_ids)
                # repeat u null_pos_len times and children_ids_len times
                u_null_level.extend([u]*null_pos_len)
                null_pos_level.extend(null_pos_arr)
                u_children_level.extend([u]*children_ids_len)
                children_pos_arr = np.where(children_bool_arr)[0].tolist()  # needed a list
                children_pos_level.extend(children_pos_arr)
                children_ids_level.extend(children_ids)

                u_level.append(u)

                u += 1
                node_u = self.id_to_node[u]

            if u < In:
                last_depth = node_u.depth

            U_levels.append(u_level)

            Children_pos_levels.append(children_pos_level)
            Children_ids_levels.append(children_ids_level)
            Null_pos_levels.append(null_pos_level)
            U_children_levels.append(u_children_level)
            U_null_levels.append(u_null_level)

        # last depth ora contiene la max depth dei nodi interni == il numero di livelli dei nodi interni!
        return U_levels, Children_pos_levels, Children_ids_levels, Null_pos_levels, \
               U_children_levels, U_null_levels, last_depth
