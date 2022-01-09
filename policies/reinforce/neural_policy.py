import torch.nn as nn
import torch.nn.functional as F


class NeuralPolicy(nn.Module):
    """
    Simple neural network policy chooser
    """

    def __init__(self):
        super(NeuralPolicy, self).__init__()

        # 4 being the observation space (cart position, velocity, pole angle and angular velocity)
        self.input_layer = nn.Linear(4, 5)
        self.output_layer = nn.Linear(5, 1)

    def forward(self, inputs):
        x = F.elu(self.input_layer(inputs))
        return F.sigmoid(self.output_layer(x))
