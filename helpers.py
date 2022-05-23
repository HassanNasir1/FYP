import numpy as np
import torch
import torchvision.transforms as transforms


def preprocess(image):
    array = np.array(image)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    processed = transform(array)
    return torch.unsqueeze(processed, 0)


def postprocess(processed):
    count = 0

    output_cnn = processed.tolist()
    for out1 in output_cnn:
        for out2 in out1:
            if out2 == 0:
                count += 1
    return count
