from Tree import Node
class Parser:

    @staticmethod
    def parse(filename, maxOutdegree):
        roots = []

        with open(filename) as f:
            for line in f:
                line = line.strip()
                dim = len(line)
                y = -1  # Use it as y for all the nodes of the tree
                if line[1] == ":":
                    y = int(line[0]) - 1
                    idx = 2
                else:
                    y = int(line[0:2])  - 1  # 2 not included
                    idx = 3

                curr = None
                root = None

                stop = False
                while not stop:
                    oldidx = idx
                    while idx < dim and line[idx] != '(' and line[idx] != ')':
                        idx = idx+1

                    if idx < dim and line[idx] == "(":

                        x = int(line[oldidx:idx]) - 1  # idx not included

                        if root is None:
                            root = Node(x, y, maxOutdegree, None, None)
                            roots.append(root)
                            curr = root
                        else:

                            l = curr.children_size()
                            # the position in the children list in this case corresponds to the true position
                            curr.add_child(x, y, l)
                            curr = curr.get_lth_child(l)

                        if line[idx+1] == "$" and line[idx+2] == ")":
                            curr = curr.parent
                            idx = idx + 3  # positioning the cursor after the ")"

                        else:
                            idx = idx + 1

                    elif idx < dim and line[idx] == ")":
                        curr = curr.parent
                        idx = idx + 1

                    if idx >= dim:
                        stop = True
            print("Parsed.")

        return roots