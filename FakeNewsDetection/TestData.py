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
model = model.to(Parameters.DEVICE)
SaveLoad.load_checkpoint(model_location, model)
dataset = {
    "data_file": join(Parameters.SOURCE_4_FOLDER, "output.tsv"),
    "output_dir": join(Parameters.SOURCE_4_FOLDER, "output")
}
train_iterator, test_iterator = DatasetPrepare.create_iterators(dataset["data_file"])
results = {
    "train": {
        "iterator": train_iterator,
        "probabilities": {
            "true_prediction": {
                "0": list(),
                "1": list()
            },
            "false_prediction": {
                "0": list(),
                "1": list()
            }
        }
    },
    "test": {
        "iterator": test_iterator,
        "probabilities": {
            "true_prediction": {
                "0": list(),
                "1": list()
            },
            "false_prediction": {
                "0": list(),
                "1": list()
            }
        }
    }
}

stars = "*" * 15
model.eval()
with torch.no_grad():
    for data_type, data_type_dict in results.items():
        print('{stars}\t working on {dataType} data \t {stars}'.format(stars=stars, dataType=data_type))
        for line in data_type_dict["iterator"]:
            prob_dict = data_type_dict["probabilities"]
            (title_text, labels), _ = line
            labels = labels.to(Parameters.DEVICE)
            title_text = title_text.to(Parameters.DEVICE)
            model_predictions = model(title_text)
            for index, combo in enumerate(zip(model_predictions, labels)):
                prob_prediction, class_prediction = torch.max(combo[0], 0)
                real_prediction = combo[1]
                if class_prediction == real_prediction:
                    prob_dict["true_prediction"][str(class_prediction.item())].append(prob_prediction)
                else:
                    prob_dict["false_prediction"][str(class_prediction.item())].append(prob_prediction)

                # class_prediction = torch.argmax(prob_prediction, 0).item()
                # prob = prob_prediction[class_prediction].item()
                # prob_list = prob_dict[str(class_prediction)]
                # prob_list.append(prob)


for data_type, classes in results.items():
    print('{stars}\t result for dataset {dataType} \t {stars}'.format(stars=stars, dataType=data_type))
    for prediction_result, prob_prediction in classes["probabilities"].items():
        for class_name, class_probabilities in prob_prediction.items():
            print('result for class {success}, {class_name}:'.format(success=prediction_result,class_name=class_name))
            print(f'\t avg: {calculate_avg(class_probabilities)}')
            print(f'\t min: {min(class_probabilities)}')
            print(f'\t max: {max(class_probabilities)}')
            print(f'\t stddev: {calculate_stddev(class_probabilities)}')
            print(f'\t count items: {len(class_probabilities)}')
            print("\n")

print("done")
