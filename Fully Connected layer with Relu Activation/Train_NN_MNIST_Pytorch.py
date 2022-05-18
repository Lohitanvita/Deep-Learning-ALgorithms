import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn
from torch import optim


# Defining a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
training_data = datasets.MNIST(r'C:\Users\rlohi\Downloads', download=True, train=True, transform=transform)
validation_data = datasets.MNIST(r'C:\Users\rlohi\Downloads', download=True, train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(training_data, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(validation_data , batch_size=128, shuffle=True)

data_iterator = iter(train_loader)
images, labels = data_iterator.next()
print(type(images))
print(images.shape)
print(labels.shape)

file = r'C:\Users\rlohi\Downloads'
figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
plt.show()
plt.savefig(file+'\\fig2.png')
plt.close('all')

# Layer details for the neural network
input_size = 784
hidden_sizes = [200, 50]
output_size = 10

# Build a feed-forward network
NN_architecture = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
print(NN_architecture)

criterion = nn.NLLLoss()
images, labels = next(iter(train_loader))
images = images.view(images.shape[0], -1)

probability = NN_architecture(images)
loss = criterion(probability, labels)

print('Before backward pass: \n', NN_architecture[0].weight.grad)
loss.backward()
print('After backward pass: \n', NN_architecture[0].weight.grad)



# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(NN_architecture.parameters(), lr=0.01, momentum=0.5)

print('Initial weights - ', NN_architecture[0].weight)

images, labels = next(iter(train_loader))
images.resize_(128, 784)

# Clear the gradients, do this because gradients are accumulated
optimizer.zero_grad()

# Forward pass, then backward pass, then update weights
output = NN_architecture(images)
loss = criterion(output, labels)
loss.backward()
print('Gradient -', NN_architecture[0].weight.grad)

# Take an update step and few the new weights
optimizer.step()
print('Updated weights - ', NN_architecture[0].weight)

#Core Training of NN
optimizer = optim.SGD(NN_architecture.parameters(), lr=0.01, momentum=0.5)
time0 = time()
epochs = 10
for e in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = NN_architecture(images)
        loss = criterion(output, labels)

        # This is where the NN_architecture learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(train_loader)))
print("\nTraining Time (in minutes) =", (time() - time0) / 60)



def view_classify(self,img, ps):
    ''' Function for viewing an image and it's predicted classes.'''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()

    images, labels = next(iter(val_loader))

    img = images[0].view(1, 784)
    # Turn off gradients to speed up this part
    with torch.no_grad():
        probability = NN_architecture(img)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(probability)
    probab = list(ps.numpy()[0])
    print("Predicted Digit =", probab.index(max(probab)))
    self.view_classify(img.view(1, 28, 28), ps)


#NN_architecture Evaluation
correct_count, all_count = 0, 0
for images,labels in val_loader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    # Turn off gradients to speed up this part
    with torch.no_grad():
        probability = NN_architecture(img)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(probability)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))