import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)

print(model)
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())