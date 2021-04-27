from flask import Flask, request, jsonify, json
from BERT import BERT
import DatasetPrepare
import Parameters
import SaveLoad
import os

model = BERT().to(Parameters.DEVICE)
model_load_path = os.path.join("Data", "Dataset4", "output", "model.pt")
SaveLoad.load_checkpoint(load_path=model_load_path, model=model)
app = Flask(__name__)


@app.route('/patternModel', methods=["POST"])
def index():
    body = request.get_json()
    titles = body['titles']
    tokenized_titles = DatasetPrepare.encode_bert(titles)
    prediction_tensor = model(tokenized_titles)
    prediction_of_true_label = [pred_tensor[Parameters.TRUE_LABEL_INDEX] for pred_tensor in prediction_tensor.squeeze().tolist()]
    return {"prediction": prediction_of_true_label}

if __name__ == "__main__":
    app.run()