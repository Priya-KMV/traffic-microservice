from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np

# Define the same model structure as in your training notebook
class LSTMModel(nn.Module):
    def __init__(self, input_size=207, hidden_size=128, output_size=207, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.2)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        output, _ = self.lstm(x)
        preds = self.linear(output[-1])
        return preds

# Load the model
model = LSTMModel()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json['input']  # Shape: [sequence_len, sensors]
        input_array = np.array(input_data).astype(np.float32)
        input_tensor = torch.tensor(input_array).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(input_tensor)
            prediction = output.numpy().tolist()[0]  # First batch, full sensor prediction

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
