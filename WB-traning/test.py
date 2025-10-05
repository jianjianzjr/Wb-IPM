import torch
import torch.nn as nn
import torch.optim as optim

# Simple model with a learnable threshold
class LearnableThresholdLayer(nn.Module):
    def __init__(self,in_channel=100, initial_threshold=0.5,batchOn=True, ReluOn=True):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor([initial_threshold]))

        self.norm = nn.BatchNorm2d(num_features=in_channel)
        nn.init.constant_(self.norm.weight, 1)
        nn.init.constant_(self.norm.bias, 0)

        self.relu = nn.ReLU()

    def forward(self, input_data, excitation):
        # Assuming excitation is the same shape as input_data
        # Find the maximum value per channel or across the entire tensor, depending on your needs
        max_value = excitation.max()  # or excitation.max() for a single global max value
        thresholded_excitation = nn.ReLU(
            excitation - self.threshold * max_value)  # Apply threshold in a differentiable way
        out = input_data * thresholded_excitation  # Mask input data using the thresholded excitation
        return out

    # def forward(self, input_data, excitation, batchOn=True, ReluOn=True):
    #     # Applying a threshold that's similar to ReLU but learnable
    #     mask = excitation > (self.threshold * excitation.max())
    #     out = input_data * mask.float()
    #     # if batchOn:
    #     #     out = self.norm(out)
    #     # if ReluOn:
    #     #     out = self.relu(out)
    #     return out

# Example usage
model = LearnableThresholdLayer()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Dummy input and target
input_x = torch.rand(100,55,55)
input_y = torch.rand(100,55,55)
target = torch.rand(100,55,55)

for epoch in range(5):
    optimizer.zero_grad()
    output = model(input_x,input_y)
    loss = ((output - target) ** 2).mean()  # Adding threshold to loss
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Threshold Grad: {model.threshold.item()}, '
          f'Threshold Grad: {model.threshold.grad}')



