import torch, torch.nn as nn

class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)
    def forward(self, x):
        return torch.softmax(self.fc2(self.relu(self.fc1(x))), dim=-1)

torch.onnx.export(SmallModel().eval(), torch.randn(1, 4), "models/mlp.onnx",
                  input_names=["input"], output_names=["output"], dynamo=False)
