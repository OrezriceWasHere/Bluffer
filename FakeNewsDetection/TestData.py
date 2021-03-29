import torch
import math
import Parameters
from os.path import join
import SaveLoad
from BERT import BERT
import DatasetPrepare


def calculate_avg(float_list):
    return sum(float_list) / len(float_list)


def calculate_stddev(float_list):
    avg = calculate_avg(float_list)
    variance = sum(map(lambda item: item * item, map(lambda item: item - avg, float_list))) / len(float_list)
    return math.sqrt(variance)


model_location = join(Parameters.OUTPUT_4_FOLDER, Parameters.MODEL_FILE_NAME)
model = BERT()
SaveLoad.load_checkpoint(model_location, model)
dataset = {
    "data_file": join(Parameters.SOURCE_4_FOLDER, "output.tsv"),
    "output_dir": join(Parameters.SOURCE_4_FOLDER, "output")
}
results = {
    "train": {
        "0": list(),
        "1": list()
    },
    "test": {
        "0": list(),
        "1": list()
    }
}
train_iterator, test_iterator = DatasetPrepare.create_iterators(dataset["data_file"])


stars = "*" * 15
model.eval()
with torch.no_grad():
    for data_type, prob_dict in results.items():
        print('{stars}\t working on {dataType} data \t {stars}'.format(stars=stars, dataType=data_type))
        for line in train_iterator:
            (title_text, labels), _ = line
            labels = labels.to(Parameters.DEVICE)
            title_text = title_text.to(Parameters.DEVICE)
            result = model(title_text)
            for index, prob_prediction in enumerate(result):
                class_prediction = torch.argmax(prob_prediction, 0)
                class_prediction_int = class_prediction.item()
                prob = prob_prediction[class_prediction_int]
                prob_list = prob_dict[str(class_prediction.item())]
                prob_list.append(prob)

for data_type, classes in results.items():
    print('{stars}\t result for dataset {dataType} \t {stars}'.format(stars=stars, dataType=data_type))
    for class_name, class_probabilities in classes.items():
        print('result for class {class_name}:'.format(class_name=class_name))
        print(f'\t avg: {calculate_avg(class_probabilities)}')
        print(f'\t min: {min(class_probabilities)}')
        print(f'\t max: {max(class_probabilities)}')
        print(f'\t stddev: {calculate_stddev(class_probabilities)}')
        print("\n")

print("done")
