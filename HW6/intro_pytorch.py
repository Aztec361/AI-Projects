import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training=True):
    if training:
        dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    else:
        dataset = datasets.FashionMNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=(training == True))
    return loader


def build_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model


def train_model(model, train_loader, criterion, T):
    """
    Trains the given model on the given train DataLoader for T epochs using the given criterion.

    INPUT:
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy
        T - number of epochs for training

    RETURNS:
        None
    """

    # Define optimizer
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model for T epochs
    for epoch in range(T):
        correct = 0
        total_loss = 0
        total = 0

        # Set the model to train mode
        model.train()

        for images, labels in train_loader:
            opt.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            # Backward pass and optimization
            loss.backward()
            opt.step()

            # Calculate number of correct predictions
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # Update total
            total += labels.size(0)

        # Print training status after every epoch
        accuracy = 100. * correct / total
        avg_loss = total_loss / total
        print('Train Epoch: {} Accuracy: {}/{} ({:.2f}%) Loss: {:.3f}'.format(
            epoch, correct, total, accuracy, avg_loss))


def evaluate_model(model, test_loader, criterion, show_loss=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item() * target.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    if show_loss:
        print(f"Average loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")


def predict_label(model, test_images, index):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle Boot']
    model.eval()
    with torch.no_grad():
        logits = model(test_images[index])
        probabilities = F.softmax(logits, dim=1)
        sorted_probs, sorted_classes = torch.sort(probabilities, descending=True)
        for i in range(3):
            class_name = class_names[sorted_classes[0][i].item()]
            prob = sorted_probs[0][i].item() * 100
            print(f"{class_name}: {prob:.2f}%")


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
