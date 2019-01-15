import sys
import numpy as np

# def parse_namespace(namespace):
#     namespace = namespace[9:]
#     namespace.replace('(', '{').replace('(', '{')

if __name__ == '__main__':
    log_file = sys.argv[1]
    print(log_file)
    val_set = []
    test_set = []
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # if "Namespace" in line:
            #     print(line)
            if 'val set' in line:
                val_set.append(line)
            elif 'test set' in line:
                test_set.append(line)

    # for (lines, set_name)in zip([val_set, test_set], ['val', 'test']):
    #     f1s = [float(line[-6:]) for line in lines]
    #     index = np.argmax(f1s)
    #     print("Best {} result at number {} epoch: ".format(set_name, index + 1))
    #     print(lines[index])
    # print("{} epoches in total.".format(len(val_set)))

    accs = [line.split()[11] for line in val_set]
    print(accs)
    index = np.argmax(accs)
    print("Best val and test result at number {} epoch: ".format(index + 1))
    print(val_set[index])
    print(test_set[index])
    print("{} epoches in total.".format(len(val_set)))