from torchvision import transforms

cnn_transforms = transforms.Compose([
    #transforms.Resize((299, 299)),
    transforms.ToTensor(),
    #transforms.Normalize([0.5] * 3, [0.5] * 3)
])