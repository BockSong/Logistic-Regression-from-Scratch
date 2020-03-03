import sys
import math
import numpy as np

Debug = False

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sparse_add(X, W):
           for i, v in X.items():
              W[i] += v
           return W

def sparse_sub(X, W):
    for i, v in X.items():
        W[i] -= v
    return W

def sparse_dot(X, W):
    product = 0.0
    for i, v in X.items():
        product += W[i] * v
    return product

class lr(object):
    def __init__(self):
        self.root = None

    def train_epoch(self, dataset, available_nodes, depth = 0):
        # special stopping rules
        pred = majority_vote(dataset)
        _, label = gini_impurity(dataset)
        if (len(available_nodes) == 0) or (depth >= self.max_depth):
            if Debug:
                print("stoped: special rules", available_nodes, depth)
            return self.make_leaf(pred, label)

        gg_max, split_idx = 0, -1
        dataset_0, dataset_1 = None, None
        for idx in available_nodes:
            gg_cur, node_info = gini_gain(dataset, idx)
            # Is there any possibility to have two identical gini from two attris?
            if (gg_cur > gg_max):
                gg_max = gg_cur
                split_idx = idx
                split_info = node_info

        if split_idx != -1:
            # split and create a node
            node = tree_node(split_idx)
            node.split_info = split_info
            dataset_0, dataset_1 = node.split_info["left_ds"], node.split_info["right_ds"]
            
            # must make a new copy in order to don't affect other sub-trees
            unused_nodes = available_nodes.copy()
            unused_nodes.remove(split_idx)
            next_depth = depth
            next_depth += 1

            if Debug:
                print("left chd:\n dataset: ", len(dataset_0))
                print("right chd:\n dataset: ", len(dataset_1))

            # build sub trees
            left_chd = self.train_stump(dataset_0, unused_nodes, next_depth)
            if left_chd:
                node.left = left_chd
            else:
                node.left = tree_node(majority_vote(dataset_0), True)

            if Debug:
                print("from here is the right tree")
            right_chd = self.train_stump(dataset_1, unused_nodes, next_depth)
            if right_chd:
                node.right = right_chd
            else:
                node.right = tree_node(majority_vote(dataset_1), True)

            return node
        else:
            # touch stoping rule, no split
            if Debug:
                print("stoped: regular rules")
            return self.make_leaf(pred, label)

    def train_model(self, train_file, num_epoch):
        dataset = []

        # read from dataset
        with open(train_file, 'r') as f:
            idx = 0
            for line in f:
                split_line = line.strip().split('\t')
                if idx == 0:
                    self.attriName = split_line
                else:
                    dataset.append(split_line)
                    if split_line[-1] not in self.labelName:
                        self.labelName.add(split_line[-1])
                idx += 1

        self.dataset = dataset

        # use length of first data line to generate available attributes # set
        self.root = self.train_stump(dataset, set(range(len(dataset[0]) - 1)))

    # Use decision tree to predict y for a single data line
    def predict(self, node, ele):
        if node.isleaf:
            return node.val
        elif node.split_info["left_value"] == ele[node.val]:
            return self.predict(node.left, ele)
        elif node.split_info["right_value"] == ele[node.val]:
            return self.predict(node.right, ele)
        else:
            print("Error! Unknown value " + ele[node.val] + "for attribute " + self.attriName[node.val])
            exit(-1)

    def evaluate(self, in_path, out_path):
        error = 0
        total = 0

        with open(in_path, 'r') as f_in:
            with open(out_path, 'w') as f_out:
                for line in f_in:
                    if total == 0:
                        total += 1
                        continue
                    split_line = line.strip().split('\t')

                    pred = self.predict(self.root, split_line)
                    if pred != split_line[-1]:
                        error += 1
                    f_out.write(pred + "\n")
                    total += 1

        return error / (total - 1) # len(data)


if __name__ == '__main__':
    formatted_train_input = sys.argv[1]  # path to the formatted training input .tsv file
    formatted_val_input = sys.argv[2] # path to the formatted validation input .tsv file
    formatted_test_input = sys.argv[3] # path to the formatted test input .tsv file
    dict_input = sys.argv[4] # path to the dictionary input .txt file
    train_out = sys.argv[5] # path to output .labels file which predicts on trainning data
    test_out = sys.argv[6] #  path to output .labels file which predicts on test data
    metrics_out = sys.argv[7] # path of the output .txt file to write metrics
    num_epoch = int(sys.argv[8]) # an integer specifying the number of times SGD loops

    model = lr()

    # training: build the lr model
    model.train_model(train_file, num_epoch)

    # testing: evaluate and write labels to output files
    train_error = model.evaluate(train_file, train_out)
    test_error = model.evaluate(test_file, test_out)

    print("train_error: ", train_error)
    print("test_error: ", test_error)

    # Output: Metrics File
    with open(metrics_out, 'w') as f_metrics:
        f_metrics.write("error(train): " + str(train_error) + "\n")
        f_metrics.write("error(test): " + str(test_error))
