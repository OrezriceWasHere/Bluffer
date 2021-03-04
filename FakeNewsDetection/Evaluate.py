import matplotlib.pyplot as plt

import Parameters
from SaveLoad import load_metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from Parameters import DEVICE
import seaborn as sns


def display_loss_graph(metric_file_location, title=""):
    train_loss_list, test_loss_list, global_steps_list = load_metrics(metric_file_location)
    plt.plot(global_steps_list, train_loss_list, label='Train')
    plt.plot(global_steps_list, test_loss_list, label='Test')
    plt.xlabel('Global Steps')
    plt.ylabel('Loss')
    plt.legend()
    if len(title) > 0:
        plt.title(title)
    plt.show()


def evaluate(model, test_loader, title="", criterion=nn.BCELoss()):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for line in test_loader:
            (title_text, labels), _ = line
            labels = labels.to(DEVICE)
            title_text = title_text.to(DEVICE)
            result = model(title_text)
            prediction = torch.argmax(result, 1).long()
            y_pred.extend(prediction.tolist())
            y_true.extend(labels.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix ' + title)

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    ax.yaxis.set_ticklabels(['FAKE', 'REAL'])
    print(cm)
