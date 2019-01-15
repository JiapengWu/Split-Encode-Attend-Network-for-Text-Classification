import sys

if __name__ == '__main__':
    args = sys.argv[1:]
    average = 'macro'
    fload_list = list(map(lambda x: float(x), args))
    tn1, fp1, fn1, tp1 = fload_list[0], fload_list[1], fload_list[2], fload_list[3]# 0: negative, 1: positive
    precision_1 = tp1 / (tp1 + fp1)
    recall_1 = tp1 / (tp1 + fn1)
    f1_score_1 = 2 * (recall_1 * precision_1) / (recall_1 + precision_1)
    print(precision_1, recall_1, f1_score_1)

    tp2, fn2, fp2, tn2 = fload_list[0], fload_list[1], fload_list[2], fload_list[3]# 1: negative, 0: positive
    precision_2 = tp2 / (tp2 + fp2)
    recall_2 = tp2 / (tp2 + fn2)
    f1_score_2 = 2 * (recall_2 * precision_2) / (recall_2 + precision_2)
    print(precision_2, recall_2, f1_score_2)

    if average == 'macro':
        precision = (precision_1 + precision_2) / 2
        recall = (recall_1 + recall_2) / 2
        f1_score = 2 * (recall * precision) / (recall + precision)
        print(precision, recall, f1_score)
