import sys
import math
import numpy as np

Debug = False
learning_rate = 0.1

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

def build_dic(dict_input):
    dic = dict()
    with open(dict_input, 'r') as f_dict:
        for line in f_dict:
            split_line = line.strip().split(' ')
            dic[split_line[0]] = split_line[1]
    return dic


class lr(object):
    def __init__(self, dic):
        self.param = 0
        self.learning_rate = learning_rate
        self.dic = dic

    def SGD_step(self, feature, label):
        # TODO: update param by taking one SGD step
        pass

    def train_model(self, train_file, num_epoch):
        self.dataset = []
        # read from dataset
        with open(train_file, 'r') as f:
            idx = 0
            for line in f:
                split_line = line.strip().split('\t')
                if idx == 0:
                    self.attriName = split_line
                else:
                    self.dataset.append(split_line)
                    if split_line[-1] not in self.labelName:
                        self.labelName.add(split_line[-1])
                idx += 1

        for i in range(num_epoch):
            for feature in self.dataset:
                self.SGD_step(feature[1], feature[0])

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
    train_input = sys.argv[1]  # path to the formatted training input .tsv file
    val_input = sys.argv[2] # path to the formatted validation input .tsv file
    test_input = sys.argv[3] # path to the formatted test input .tsv file
    dict_input = sys.argv[4] # path to the dictionary input .txt file
    train_out = sys.argv[5] # path to output .labels file which predicts on trainning data
    test_out = sys.argv[6] #  path to output .labels file which predicts on test data
    metrics_out = sys.argv[7] # path of the output .txt file to write metrics
    num_epoch = int(sys.argv[8]) # an integer specifying the number of times SGD loops

    # read and build dic
    dic = build_dic(dict_input)

    model = lr(dic)

    # training: build the lr model
    model.train_model(train_input, num_epoch)

    # testing: evaluate and write labels to output files
    train_error = model.evaluate(train_input, train_out)
    test_error = model.evaluate(test_input, test_out)

    print("train_error: ", train_error)
    print("test_error: ", test_error)

    # Output: Metrics File
    with open(metrics_out, 'w') as f_metrics:
        f_metrics.write("error(train): " + str(train_error) + "\n")
        f_metrics.write("error(test): " + str(test_error))
