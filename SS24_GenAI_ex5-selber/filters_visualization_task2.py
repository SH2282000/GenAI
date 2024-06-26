
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
    widthVar = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    heightVar = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    tvLoss = weight*torch.sqrt(heightVar + widthVar)
    return tvLoss 

def visualise_layer_filter_task2(model, layer_nmbr, 
                                 filter_nmbr, opt="Adam", 
                                 num_optim_steps=26, 
                                 inputMode="Random", lr=0.1, 
                                 img=None, 
                                 preProc=False,
                                 filterMode="Custom",
                                 highestFilter=1,
                                 addTvLoss=False,
                                 tvLossWeight=500.):

    # Generate a random image
    if inputMode == "Random":
        img = np.uint8(np.random.uniform(low=120,
                                            high=190,
                                            size=(224, 224, 3)))
    elif inputMode == "128":
        img = np.full((224, 224, 3), 128, dtype=np.uint8)
    elif inputMode == "Image":
        img = img
    else:
        print("Not a valid input mode\n")

    # Process image and return variable
    processed_image = preprocess_image(img, resize_im=preProc)
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


    # If filterMode == "Highest", do forward pass and pick one of the most activated filters
    # for further optimization of the input image
    if filterMode == "Highest":
        optimizer.zero_grad()
        x = processed_image

        # Do one single forward pass 
        for index, layer in enumerate(model):
            # Forward pass layer by layer
            x = layer(x)
            if index == layer_nmbr:
                # Only need to forward until the selected layer is reached
                # Now, x is the output of the selected layer
                break

        filter_means = x.mean(dim=[2, 3])
        _, sorted_filter_indices = filter_means.sort(descending=True)
        filter_nmbr = sorted_filter_indices[0][highestFilter-1]
        # print(f"filter_means: {filter_means}\n")
        # print(f"filter_means.shape: {filter_means.shape}\n")
        # print(f"sorted_filter_indices: {sorted_filter_indices}\n")
        # print(f"sorted_filter_indices.shape: {sorted_filter_indices.shape}\n")
        # print(f"filter_nmbr: {filter_nmbr}\n")

    # print(f"Do optimization on filter {filter_nmbr}\n:")
    # Optimze input image 
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


        # Loss function is the mean of the output of the selected layer/filter
        # We try to minimize the mean of the output of that specific filter
        if filterMode == "All":
            conv_output = x[0, :]  # Select all filters
            loss = -torch.norm(conv_output, p=2)
        else:
            conv_output = x[0, filter_nmbr]
            loss = -torch.mean(conv_output)
        # You may need to add total variation loss later
        if addTvLoss:
            loss_tv = total_variation_loss(processed_image, tvLossWeight)
            loss = loss + loss_tv*1.

        # print(f'Step {i:05d}. Loss:{loss.data.cpu().numpy():0.2f}')
        # Compute gradients
        loss.backward()
        # Apply gradients
        optimizer.step()
        # Recreate image
        optimized_image = recreate_image(processed_image.cpu())

    return optimized_image


def visualise_layer_filter_task2_multLayers(model, layer_nmbrs, 
                                 filter_nmbr, opt="Adam", 
                                 num_optim_steps=26, 
                                 inputMode="Random", lr=0.1, 
                                 img=None, 
                                 preProc=False,
                                 filterMode="Custom",
                                 highestFilter=1,
                                 addTvLoss=False,
                                 tvLossWeight=500.):

    # Generate a random image
    if inputMode == "Random":
        img = np.uint8(np.random.uniform(low=120,
                                            high=190,
                                            size=(224, 224, 3)))
    elif inputMode == "128":
        img = np.full((224, 224, 3), 128, dtype=np.uint8)
    elif inputMode == "Image":
        img = img
    else:
        print("Not a valid input mode\n")

    # Process image and return variable
    processed_image = preprocess_image(img, resize_im=preProc)
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


    # print(f"Do optimization on filter {filter_nmbr}\n:")
    # Optimze input image 
    for i in range(1, num_optim_steps):
        optimizer.zero_grad()
        # Assign create image to a variable to move forward in the model
        x = processed_image
        firstHit = True
        for index, layer in enumerate(model):
            # Forward pass layer by layer
            x = layer(x)
            if index in layer_nmbrs:
                # Only need to forward until the selected layer is reached
                # Now, x is the output of the selected layer
            
                # Loss function is the mean of the output of the selected layer/filter
                # We try to minimize the mean of the output of that specific filter
                if filterMode == "All":
                    conv_output = x[0, :]  # Select all filters
                    if firstHit:
                        loss = -torch.norm(conv_output, p=2)
                        firstHit = False
                    else:
                        loss += -torch.norm(conv_output, p=2)

                else:
                    conv_output = x[0, filter_nmbr]
                    if firstHit:
                        loss = -torch.mean(conv_output)
                        firstHit = False
                    else:
                        loss += -torch.mean(conv_output)
        # You may need to add total variation loss later
        if addTvLoss:
            loss_tv = total_variation_loss(processed_image, tvLossWeight)
            loss = loss + loss_tv*1.

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
    visualise_layer_filter_task2(model,
                           layer_nmbr=layer_nmbr,
                           filter_nmbr=filter_nmbr)
