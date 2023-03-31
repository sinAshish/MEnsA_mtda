####################################################################################
# Author: Ashish Sinha
# Disc: Utility functions for calculating per-sample accuracy and per-class accuracy
####################################################################################

import torch
import torch.nn.functional as F


def calculate_accuracy(pred_list, label_list, num_class, mean_method='sample'):
    """
    :param pred_list: (N) Prediction
    :param label_list: (N) Label
    :param num_class: (int) Number of class
    :param mean_method: (str) Accuracy mean method ['sample', 'class']
    :return: Accuracy
    """
    if mean_method == 'sample':
        correct = (pred_list == label_list).sum().to(float)
        total = label_list.size(0)
        accuracy = correct / total
    elif mean_method == 'class':
        accuracy = torch.tensor(calculate_accuracy_per_class(pred_list, label_list, num_class, normalize=True)).mean()
    else:
        raise NotImplementedError

    return accuracy.item()


def calculate_accuracy_per_class(pred_list, label_list, num_class, normalize=True):
    """
    :param pred_list: (N) Prediction
    :param label_list: (N) Label
    :param num_class: (int) Number of class
    :param normalize: (bool) If False, return the number of correctly classified samples. Otherwise,
                       return the fraction of correctly classified samples.
    :return: (list) Accuracy per class
    """
    accuracy = F.one_hot(label_list[pred_list == label_list], num_class).sum(dim=0)
    total = F.one_hot(label_list, num_class).sum(dim=0)

    if normalize:
        accuracy = accuracy.to(float) / total.to(float)
        accuracy[torch.isnan(accuracy)] = 0

    return accuracy.tolist()


def calculate_accuracy_all(pred_list, label_list, num_class):
    sample_accuracy = calculate_accuracy(pred_list, label_list, num_class, mean_method='sample')
    class_accuracy = calculate_accuracy(pred_list, label_list, num_class, mean_method='class')
    accuracy_per_class = calculate_accuracy_per_class(pred_list, label_list, num_class, normalize=True)

    return sample_accuracy, class_accuracy, accuracy_per_class