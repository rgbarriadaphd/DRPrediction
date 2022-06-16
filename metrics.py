"""
# Author: ruben 
# Date: 1/2/22
# Project: CACFramework
# File: metrics.py

Description: Functions to provide performance metrics
"""
import logging
import statistics
import math
import numpy as np
import os

import seaborn as sns
import matplotlib.pyplot as plt



class PerformanceMetrics:
    """
    Class to compute model performance
    """
    def __init__(self, ground, prediction, output_folder, classes, percent=False, formatted=False):
        """
        PerformanceMetrics class constructor
        :param ground: input array of ground truth
        :param prediction: input array of prediction values
        :param percent: (bool) whether if data is provided [0-1] or [0%-100%]
        :param formatted: (bool) Decimals to format.
            If activated return string instead of float
        """
        print("Executing metrics!")
        assert (len(ground) == len(prediction))
        self._ground = np.array(ground)
        self._prediction = np.array(prediction)
        self._n_classes = len(classes)
        self._class_distribution = {}
        self._class_metrics = {}

        self._percent = percent
        self._formatted = formatted
        self._classes = classes
        self._out_folder = output_folder
        self._confusion_matrix = np.zeros((self._n_classes,self._n_classes), dtype=np.int)
        self._compute_measures()

    def _compute_measures(self):
        """
        Compute performance measures
        """
        self._compute_confusion_matrix()
        self._compute_metrics()
        self._compute_output()

    def _compute_confusion_matrix(self):
        """
        Computes the confusion matrix of a model
        """

        for i in range(len(self._prediction)):
            self._confusion_matrix[self._prediction[i], self._ground[i]] += 1

        for current, cls in enumerate(self._classes):
            self._class_distribution[cls] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

            # True positives
            self._class_distribution[cls]['tp'] = self._confusion_matrix[current, current]

            # True negatives
            for i in range(self._n_classes):
                for j in range(self._n_classes):
                    if i != current and j != current:
                        self._class_distribution[cls]['tn'] += self._confusion_matrix[i, j]

            # False positives
            for i in range(self._n_classes):
                if i != current:
                    self._class_distribution[cls]['fp'] += self._confusion_matrix[current, i]

            # False negatives
            for i in range(self._n_classes):
                if i != current:
                    self._class_distribution[cls]['fn'] += self._confusion_matrix[i, current]

    def _compute_metrics(self):

        import pprint

        for cls in self._classes:
            self._class_metrics[cls] = {'recall': None, 'precision': None, 'f1': None}

            tn = self._class_distribution[cls]['tn']
            tp = self._class_distribution[cls]['tp']
            fn = self._class_distribution[cls]['fn']
            fp = self._class_distribution[cls]['fp']

            try:
                precision = tp / (tp + fp)
            except ZeroDivisionError:
                precision = '0.0'
            try:
                recall = tp / (tp + fn)
            except ZeroDivisionError:
                recall = '0.0'
            try:
                f1 = 2 * (precision * recall / (precision + recall))
            except ZeroDivisionError:
                f1 = '0.0'

            self._class_metrics[cls]['precision'] = f'{precision:0.2f}'
            self._class_metrics[cls]['recall'] = f'{recall:0.2f}'
            self._class_metrics[cls]['f1'] = f'{f1:0.2f}'

    def _compute_output(self):

        with open(os.path.join(self._out_folder, 'metrics.latex'), 'w') as f_out:
            f_out.write('\\begin{table}[H]\n')
            f_out.write('\caption{Model metrics by classes.}\n')
            f_out.write('\centering\n')
            f_out.write('\\begin{tabular}{c|c c c }\n')
            f_out.write(
                '\t\\textbf{Class} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-score}\\\\ \n')
            f_out.write('\t\hline\n')
            f_out.write('\t\hline\n')

            for cls, metrics in self._class_metrics.items():
                formatted = cls
                if '_' in cls:
                    formatted = formatted.replace('_', '\_')
                f_out.write(f'\t{formatted} & {metrics["precision"]} & {metrics["recall"]} & {metrics["f1"]}\\\\ \n')
                f_out.write(f'\t\hline\n')

            f_out.write('\end{tabular}\n')
            f_out.write('\label{tab:metrics}\n')
            f_out.write('\end{table}\n')

        cm = self._confusion_matrix
        class_names = self._classes

        # Plot confusion matrix in a beautiful manner
        fig = plt.figure(figsize=(16, 14))
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, fmt='g', annot_kws={"size": 35})  # annot=True to annotate cells
        # labels, title and ticks
        ax.set_xlabel('True', fontsize=20)
        ax.xaxis.set_label_position('bottom')
        plt.xticks(rotation=90)
        ax.xaxis.set_ticklabels(class_names, fontsize=20)
        ax.xaxis.tick_bottom()

        ax.set_ylabel('Predicted', fontsize=20)
        ax.yaxis.set_ticklabels(class_names, fontsize=20)
        plt.yticks(rotation=0)

        plt.title('Confusion Matrix', fontsize=30)

        plt.savefig(os.path.join(self._out_folder, 'confusion_matrix.png'))

        cm = self._confusion_matrix/self._confusion_matrix.sum(axis=0)
        class_names = self._classes

        # Plot confusion matrix in a beautiful manner
        fig = plt.figure(figsize=(16, 14))
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, fmt='g', annot_kws={"size": 15})  # annot=True to annotate cells
        # labels, title and ticks
        ax.set_xlabel('True', fontsize=20)
        ax.xaxis.set_label_position('bottom')
        plt.xticks(rotation=90)
        ax.xaxis.set_ticklabels(class_names, fontsize=20)
        ax.xaxis.tick_bottom()

        ax.set_ylabel('Predicted', fontsize=20)
        ax.yaxis.set_ticklabels(class_names, fontsize=20)
        plt.yticks(rotation=0)

        plt.title('Normalized Confusion Matrix', fontsize=30)

        plt.savefig(os.path.join(self._out_folder, 'normalized_confusion_matrix.png'))



if __name__ == '__main__':
    # Test functions
    ground = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 2]
    prediction = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
                  2, 2]
    pm = PerformanceMetrics(ground,
                            prediction,
                            output_folder='outputs/',
                            # classes=['class_0', 'class_1', 'class_2', 'class_3', 'class_4'],
                            classes=['apple', 'orange', 'mango'],
                            percent=True,
                            formatted=2
                            )


