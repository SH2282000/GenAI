
import numpy as np
import os
import torch
from torch.optim import Adam, SGD, Adagrad
from torchvision import models
from  aux_ops import preprocess_image, recreate_image, save_image


# Initialize GPU if available
use_gpu = False
if torch.cuda.is_available():
    use_gpu = True
# Select device to work on.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Total Variation Loss
def total_variation_loss(img, weight):
     # Your code here
    pass

def visualise_layer_filter_custom(model, layer_nmbr, filter_nmbr, opt, num_optim_steps=26, rnd=True, lr=0.1):

    # Generate a random image
    if rnd:
        img = np.uint8(np.random.uniform(low=120,
                                            high=190,
                                            size=(224, 224, 3)))
    else:
        img = np.full((224, 224, 3), 128, dtype=np.uint8)

    # Process image and return variable
    processed_image = preprocess_image(img, False)
    processed_image = torch.tensor(processed_image, device=device).float()
    processed_image.requires_grad = True

    # Define optimizer for the image
    if opt == "Adam":
        optimizer = Adam([processed_image], lr=lr, weight_decay=1e-5)
    elif opt == "SGD":
        optimizer = SGD([processed_image], lr=lr, weight_decay=1e-5)
    elif opt == "Adagrad":
        optimizer = Adagrad([processed_image], lr=lr, weight_decay=1e-5)
    else:
        print("Optimizer unknown\n")

    for i in range(1, num_optim_steps):
        optimizer.zero_grad()
        # Assign create image to a variable to move forward in the model
        x = processed_image
        for index, layer in enumerate(model):
            # Forward pass layer by layer
            x = layer(x)
            if index == layer_nmbr:
                # Only need to forward until the selected layer is reached
                # Now, x is the output of the selected layer
                break

        conv_output = x[0, filter_nmbr]
        # Loss function is the mean of the output of the selected layer/filter
        # We try to minimize the mean of the output of that specific filter
        loss = -torch.mean(conv_output)
        # You may need to add total variation loss later
        # loss_tv = total_variation_loss(processed_image, 500.)
        # loss = -torch.mean(conv_output) + loss_tv*1.

        # print(f'Step {i:05d}. Loss:{loss.data.cpu().numpy():0.2f}')
        # Compute gradients
        loss.backward()
        # Apply gradients
        optimizer.step()
        # Recreate image
        optimized_image = recreate_image(processed_image.cpu())

    return optimized_image


if __name__ == '__main__':
    layer_nmbr = 28
    filter_nmbr = 228

    # Fully connected layer is not needed
    model = models.vgg16(pretrained=True).features
    model.eval()
    # Fix model weights
    for param in model.parameters():
        param.requires_grad = False
    # Enable GPU
    if use_gpu:
        model.cuda()

    # use this output in some way
    visualise_layer_filter_custom(model,
                           layer_nmbr=layer_nmbr,
                           filter_nmbr=filter_nmbr)
