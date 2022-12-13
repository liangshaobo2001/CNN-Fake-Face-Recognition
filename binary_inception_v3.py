# Code partly from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

## Train or test models for a binary classification task with inception v3

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import trange
from time import sleep
from tqdm import tqdm

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, device = torch.device("cpu"), save_path = None, continue_training = False):
    if continue_training:
        checkpoint = torch.load(f"{save_path}.checkpoint.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    starting_epoch = 0
    curr_epoch = 0

    since = time.time()

    log_file = open(f"{save_path}.log.txt", "w")

    val_acc_history = []
    train_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in trange(num_epochs - starting_epoch, desc = f"Training", unit = "epochs"):
        print()
        print('Epoch {}/{}'.format(epoch + starting_epoch, num_epochs - 1))
        print('-' * 10)

        log_file.write('Epoch {}/{}\n'.format(epoch + starting_epoch, num_epochs - 1))
        log_file.write('-' * 10)
        log_file.write('\n')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
            # for inputs, labels in dataloaders[phase]:
                sleep(0.1)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            log_file.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                train_acc_history.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

            curr_epoch += 1

        # Save the model for the current epoch
        torch.save(model.state_dict(), save_path + f"epoch{epoch + starting_epoch}.pt")

        # Save the checkpoint
        torch.save({
                    'epoch': epoch + starting_epoch, 
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(), 
                    'loss': loss,}, f"{save_path}.checkpoint.pt")

        print()
        log_file.write('\n')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    log_file.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    log_file.write('Best val Acc: {:4f}\n'.format(best_acc))

    log_file.close()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history

def test_model(model, dataloaders, model_path = None):
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders['test']):
            sleep(0.1)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    test_acc = running_corrects.double() / len(dataloaders['test'].dataset)
    print('Acc: {:.4f}'.format(test_acc))
  
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def main():
    num_classes = 2
    batch_size = 8
    num_epochs = 50

    use_pretrained = True # Use pretrained Inception V3 (on imagenet data) if this parameter is true.
    continue_training = False # Continue training from a checkpoint if this parameter is true.
    testing = False # Set this to be false for training. If this parameter is true, the model will be tested.

    # The path to your dataset directory.
    # If training, the dataset should be a folder containing two sub-folders named "train" and "val"
    # containing the training and validation data. Each of these two sub-folders should contain two arbitrarily
    # named sub-folders containing the images of the two classes.
    # If testing, the dataset should be a folder containing one sub-folder named "test", which contains
    # two sub-folders with the same name respectively as the trained model's image classes.
    data_dir = "./Dataset_TpDne_Small"
    # data_dir = "./Dataset_TpDne_Test"

    # The path to save your trained models. Only used in training.
    save_path = "./Trained_Models/Scratch_200epochs_BadBh/" # Path to save the model, logs, and checkpoints.
    # save_path = "./Trained_Models/Pretrained_50epochs_BadBh/" # Path to save the model, logs, and checkpoints.
    # save_path = "./Trained_Models/Pretrained_50epochs_TpDne/"
    # save_path = "./Trained_Models/Scratch_50epochs_TpDne/"
    # save_path = "./Trained_Models/Pretrained_50epochs_Mixed/"
    # save_path = "./Trained_Models/Scratch_50epochs_Mixed/"
    # save_path = "./Trained_Models/Pretrained_50epochs_TpDne_Small/"

    # The path of your trained models to be tested. Only used in testing.
    model_path = "./Trained_Models/Pretrained_50epochs_BadBh/epoch29.pt" # Path to the model to be tested.
    # model_path = "./Trained_Models/Scratch_200epochs_BadBh/epoch61.pt" # Path to the model to be tested.
    # model_path = "./Trained_Models/Pretrained_50epochs_TpDne/epoch13.pt" # Path to the model to be tested.
    # model_path = "./Trained_Models/Scratch_50epochs_TpDne/epoch46.pt" # Path to the model to be tested.
    # model_path = "./Trained_Models/Pretrained_50epochs_Mixed/epoch25.pt" # Path to the model to be tested.
    # model_path = "./Trained_Models/Scratch_50epochs_Mixed/epoch38.pt" # Path to the model to be tested.

    if testing: # Testing
        # Set up the model.
        if use_pretrained: 
            feature_extract = True
            model = models.inception_v3(pretrained=use_pretrained)
            set_parameter_requires_grad(model, feature_extract)
            # Handle the auxilary net
            num_ftrs = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs,num_classes)
        else:
            feature_extract = False
            model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', num_classes=num_classes, pretrained=False)
            set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)

        # Get and preprocess the data.
        data_transforms = {
            'test': transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        }
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['test']}
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['test']}
        test_model(model, dataloaders_dict, model_path)
    else: # Training
        # Set up the model.
        if use_pretrained:
            feature_extract = True
            model = models.inception_v3(pretrained=use_pretrained)
            set_parameter_requires_grad(model, feature_extract)
            # Handle the auxilary net
            num_ftrs = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs,num_classes)
        else:
            feature_extract = False
            model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', num_classes=num_classes, pretrained=False)
            set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)

        # The data preprocessor.
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            'val': transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        }

        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
        # Create training and validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        params_to_update = model.parameters()
        if feature_extract:
            params_to_update = []
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)

        optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        # train and evaluate
        model, val_hist, train_hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs, is_inception=True, device=device, save_path=save_path, continue_training=continue_training)

        # Plot train and val accuracies.
        val_accuracy = [h.cpu().numpy() for h in val_hist]
        train_accuracy = [h.cpu().numpy() for h in train_hist]

        plt.title("Accuracy vs. Number of Training Epochs")
        plt.xlabel("Training Epochs")
        plt.ylabel("Accuracy")
        plt.plot(range(1,num_epochs+1), val_accuracy,label="Validation Accuracy")
        plt.plot(range(1,num_epochs+1), train_accuracy,label="Training Accuracy")
        plt.ylim((0,1.))
        plt.xticks(np.arange(1, num_epochs+2, 10.0))
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()

# # Download ImageNet labels
# !wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

# # Read the categories
# with open("imagenet_classes.txt", "r") as f:
#     categories = [s.strip() for s in f.readlines()]
# # Show top categories per image
# top5_prob, top5_catid = torch.topk(probabilities, 5)
# for i in range(top5_prob.size(0)):
#     print(categories[top5_catid[i]], top5_prob[i].item())

# """### Model Description

# Inception v3: Based on the exploration of ways to scale up networks in ways that aim at utilizing the added computation as efficiently as possible by suitably factorized convolutions and aggressive regularization. We benchmark our methods on the ILSVRC 2012 classification challenge validation set demonstrate substantial gains over the state of the art: 21.2% top-1 and 5.6% top-5 error for single frame evaluation using a network with a computational cost of 5 billion multiply-adds per inference and with using less than 25 million parameters. With an ensemble of 4 models and multi-crop evaluation, we report 3.5% top-5 error on the validation set (3.6% error on the test set) and 17.3% top-1 error on the validation set.

# The 1-crop error rates on the imagenet dataset with the pretrained model are listed below.

# | Model structure | Top-1 error | Top-5 error |
# | --------------- | ----------- | ----------- |
# |  inception_v3        | 22.55       | 6.44        |

# ### References

#  - [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567).
# """