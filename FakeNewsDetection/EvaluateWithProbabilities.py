from os.path import join
from BERT import BERT
import SaveLoad
import DatasetPrepare
import Parameters
from sklearn.metrics import classification_report, confusion_matrix
import torch
from Parameters import DEVICE
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_with_probability(model, test_loader, p):
    y_pred = []
    y_true = []
    count_success = 0
    count_total = 0
    model.eval()
    with torch.no_grad():
        for line in test_loader:
            (title_text, labels), _ = line
            labels = labels.to(DEVICE)
            title_text = title_text.to(DEVICE)
            result = model(title_text)
            probabilities, classes = torch.max(result, 1)
            for index, (predict_class, predict_prob, actual_label) in enumerate(zip(classes, probabilities, labels)):
                if predict_prob.item() >= p:
                    count_total += 1
                    y_pred.append(predict_class.item())
                    y_true.append(actual_label.item())
                    if predict_class.item() == actual_label.item():
                        count_success += 1

    print(f'Statistics of success when p = {100 * p}%:')
    print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))
    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix for threshold = ' + str(p))

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    ax.yaxis.set_ticklabels(['FAKE', 'REAL'])
    print(cm)

    return count_success / count_total if count_total > 0 else 0


def display_accuracy_graph(probability_threshold_list, accuracy_list, image_output_file=""):
    plt.plot(probability_threshold_list, accuracy_list, label='p / accuracy')
    plt.xlabel('Probability Threshold')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("Accuracy for given threshold")
    if len(image_output_file) > 0:
        plt.savefig(image_output_file)
    plt.show()


model_location = join(Parameters.OUTPUT_4_FOLDER, Parameters.MODEL_FILE_NAME)
model = BERT()
model = model.to(Parameters.DEVICE)
SaveLoad.load_checkpoint(model_location, model)
output_image_file = join(model_location, "accuracies.png")
dataset = {
    "data_file": join(Parameters.SOURCE_4_FOLDER, "output.tsv"),
    "output_dir": join(Parameters.SOURCE_4_FOLDER, "output")
}
iterator = DatasetPrepare.create_iterators(dataset["data_file"], split_to_train_and_test=False)
thresholds = [t / 100 for t in range(50, 100, 2)]
accuracies = [evaluate_with_probability(model=model, test_loader=iterator, p=p) for p in thresholds]
display_accuracy_graph(thresholds, accuracies, image_output_file=output_image_file)
