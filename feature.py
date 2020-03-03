import sys
import numpy as np

Debug = False

def sparse_formatting(input_file):
    value_left = None
    left_count, right_count = 0., 0.

    with open(input_file, 'r') as f:
        idx = -1
        for line in f:
            if idx == -1:
                idx += 1
                continue
            split_line = line.strip().split('\t')
            # set the first y value met as for left edge
            if idx == 0:
                value_left = split_line[-1]

            if split_line[-1] == value_left:
                left_count += 1
            else:
                right_count += 1
            idx += 1

    gini = 1 - (left_count / idx) ** 2 - (right_count / idx) ** 2
    error = min(left_count, right_count) / idx

    if Debug:
        print("Dataset size: ", idx)
        print("For value ", value_left, ": ", left_count)
        print("For the other value: ", right_count)
        print("gini_impurity: " + str(gini))
        print("error: " + str(error))

    return representation


if __name__ == '__main__':
    train_input = sys.argv[1]  # path to the training input .tsv file
    validation_input = sys.argv[2] # path to the validation input .tsv file
    test_input = sys.argv[3] # path to the test input .tsv file
    dict_input = sys.argv[4] # path to the dictionary input .txt file
    formatted_train_out = sys.argv[5] # path to output .tsv file
    formatted_validation_out = sys.argv[6] # path to output .tsv file
    formatted_test_out = sys.argv[7] # path to output .tsv file
    feature_flag = int(sys.argv[8]) # an integer which specifies which model to construct

    train_formatted = sparse_formatting(train_input)

    with open(formatted_train_out, 'w') as train_out:
        train_out.write("gini_impurity: " + str(gini) + "\n")
        train_out.write("error: " + str(error))
