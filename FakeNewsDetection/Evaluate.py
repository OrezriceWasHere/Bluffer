import matplotlib.pyplot as plt
from SaveLoad import load_metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import Parameters
import seaborn as sns


def display_loss_graph(metric_file_location):
    train_loss_list, valid_loss_list, global_steps_list = load_metrics(metric_file_location)
    plt.plot(global_steps_list, train_loss_list, label='Train')
    plt.plot(global_steps_list, valid_loss_list, label='Valid')
    plt.xlabel('Global Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def evaluate(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (labels, tweet_text), _ in test_loader:
            labels = labels.to(Parameters.DEVICE)
            tweet_text = title.to(Parameters.DEVICE)
            output = model(tweet_text, labels)
            _, output = output
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(labels.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    ax.yaxis.set_ticklabels(['FAKE', 'REAL'])