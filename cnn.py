from turtle import forward
import torch
import torch.nn.functional as F

class CNN(torch.nn.Module):

	def __init__(self) -> None:
		super(CNN, self).__init__()

		self.relu = torch.nn.ReLU(inplace=True)
		self.max_pool = torch.nn.MaxPool2d(2)

		self.conv1 = torch.nn.Conv2d(3, 32, 3) # 3 input channels, 32 output, 3x3
		self.bn1 = torch.nn.BatchNorm2d(32)
		self.conv2 = torch.nn.Conv2d(32, 32, 5) # 32 inputs, 32 outputs, 5x5
		self.bn2 = torch.nn.BatchNorm2d(32)
		self.conv3 = torch.nn.Conv2d(32, 64, 3) # 32 inputs, 64 outputs, 5x5
		self.bn3 = torch.nn.BatchNorm2d(64)
		self.conv4 = torch.nn.Conv2d(64, 64, 5) # 64 inputs, 64 outputs, 5x5
		self.bn4 = torch.nn.BatchNorm2d(64)
		self.conv5 = torch.nn.Conv2d(64, 128, 3) # 64 inputs, 128 outputs, 3x3
		self.bn5 = torch.nn.BatchNorm2d(128)
		self.conv6 = torch.nn.Conv2d(128, 128, 5) # 128 inputs, 128 outputs, 5x5
		self.bn6 = torch.nn.BatchNorm2d(128)
		self.linear1 = torch.nn.LazyLinear(256)
		self.linear2 = torch.nn.Linear(256, 256)
		self.fc = torch.nn.LazyLinear(1)

	def features(self, x):
		x = x.float()
		x = self.relu(self.bn1(self.conv1(x)))
		x = self.max_pool(self.bn2(self.conv2(x)))
		x = self.relu(self.bn3(self.conv3(x)))
		x = self.max_pool(self.bn4(self.conv4(x)))
		x = self.relu(self.bn5(self.conv5(x)))
		x = self.max_pool(self.bn6(self.conv6(x)))
		x = self.linear1(x)
		x = self.linear2(x)
		return x

	def logits(self, features):
		x = features.view(features.size(0), -1)
		x = self.fc(x)
		return x

	def forward(self, input):
		x = self.features(input)
		x = self.logits(x)
		return x


def cnn(**kwargs):
    """
    Construct CNN.
    """
    model = CNN(**kwargs)
    return model
