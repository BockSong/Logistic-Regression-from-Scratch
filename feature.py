import sys
import numpy as np

Debug = True
trim_th = 4

def build_dic(dict_input):
    dic = dict()
    with open(dict_input, 'r') as f_dict:
        for line in f_dict:
            split_line = line.strip().split(' ')
            dic[split_line[0]] = split_line[1]
    return dic

def formatted_output(input_file, output_file, dic, flag):
    features = []

    with open(input_file, 'r') as f:
        for line in f:
            feature = []
            split_line = line.strip().split('\t')
            feature.append(int(split_line[0])) # feature[0] is label

            words = split_line[1].split(' ')
            sparse_dict = dict()
            for word in words:
                if word in dic:
                    if flag == 1:
                        # just add occured words if they are in the dic
                        sparse_dict[dic[word]] = 1
                    else:
                        # count the words
                        sparse_dict[dic[word]] = sparse_dict.get(dic[word], 0)
            
            # do trimming for model 2
            if flag == 2:
                for key in sparse_dict:
                    if sparse_dict[key] >= trim_th:
                        sparse_dict.pop(key)

            feature.append(sparse_dict) # feature[1] is the sparse_dict
            features.append(feature)

            if Debug:
                print("Length of feature: ", len(feature))
                print(feature)

    if Debug:
        print("Length of features: ", len(features))

    with open(output_file, 'w') as f_out:
        for feature in features:
            # write the label
            f_out.write(str(feature[0]) + "\t")

            # write the attributes
            for key in feature[1]:
                f_out.write(str(key) + ":" + str(feature[1][key]) + "\t")
            f_out.write("\n")


if __name__ == '__main__':
    train_input = sys.argv[1]  # path to the training input .tsv file
    val_input = sys.argv[2] # path to the validation input .tsv file
    test_input = sys.argv[3] # path to the test input .tsv file
    dict_input = sys.argv[4] # path to the dictionary input .txt file
    formatted_train_out = sys.argv[5] # path to output .tsv file
    formatted_val_out = sys.argv[6] # path to output .tsv file
    formatted_test_out = sys.argv[7] # path to output .tsv file
    feature_flag = int(sys.argv[8]) # an integer which specifies which model to construct

    # read and build dic
    dic = build_dic(dict_input)

    # output
    formatted_output(train_input, formatted_train_out, dic, feature_flag)
    formatted_output(val_input, formatted_val_out, dic, feature_flag)
    formatted_output(test_input, formatted_test_out, dic, feature_flag)

