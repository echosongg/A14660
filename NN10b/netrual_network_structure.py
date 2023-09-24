## nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#Levenberg Marquardt loss function

class NeuralNetwork(nn.Module):
    def __init__(self, input_size = 12):
        super(NeuralNetwork, self).__init__()
        self.input_layer = nn.Linear(input_size, 10)  # Input layer with 12 neurons
        self.hidden_layer = nn.Linear(10, 10)  # Hidden layer with 10 neurons
        self.output_layer = nn.Linear(10, 3)  # Output layer with 3 neurons

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

def train_network(net, X, Y, n_epochs=20, learning_rate=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    losses = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        #print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")

    return losses

#use top 12 to train a new model
def retrain_with_top_features(top_indices, X_train_tensor, Y_train_tensor, n_epochs=20,
                                    learning_rate=0.01):

    # Select top features from the original data
    X_train_selected = X_train_tensor[:, top_indices]

    # Initialize a new neural network with reduced input size
    net_selected = NeuralNetwork(input_size=len(top_indices))

    # Train the new network with the selected features
    train_network(net_selected, X_train_selected, Y_train_tensor, n_epochs=n_epochs, learning_rate=learning_rate)

    return net_selected

