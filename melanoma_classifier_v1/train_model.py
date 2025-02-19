import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from net_class import Net

# 50px x 50px
img_size = 50

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# load melanoma training data
training_data = np.load("melanoma_training_data.npy", allow_pickle=True)

print(f"Training data loaded: {len(training_data)} samples")
print(f"First image shape: {training_data[0][0].shape}, First label: {training_data[0][1]}")

train_X = torch.tensor(np.array([item[0] for item in training_data]), dtype=torch.float32) / 255    # print image arrays into this tensor
train_Y = torch.tensor(np.array([item[1] for item in training_data]), dtype=torch.float32)          # one-hot vector labels tensor

# Ensure proper shape for CNN
train_X = train_X.unsqueeze(1)  # Add channel dim (Batch, 1, 50, 50)


# Move tensors to device
train_X, train_Y = train_X.to(device), train_Y.to(device)


# Hyperparameters
net = Net().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)


# Check model output shape
outputs = net(train_X[:5])
print(f"Model output shape: {outputs.shape}, Example: {outputs}")

loss_function = nn.MSELoss()
batch_size = 100
epochs = 2


#Training loop
for epoch in range(epochs):

    # 100 imgs per pass
    for i in range(0, len(train_X), batch_size):
        print(f"EPOCH {epoch + 1}, fraction complete: {i/len(train_X)}")
        batch_X = train_X[i: i+batch_size]
        batch_Y = train_Y[i: i+batch_size]

        optimizer.zero_grad() # reset gradients of model params to zero before this pass
        outputs = net(batch_X)

        # Debugging: Check output
        print(f"Batch_X shape: {batch_X.shape}, Model outputs: {outputs[:5]}")

        loss = loss_function(outputs, batch_Y) # loss between predicted outputs and actual image one-hot vectors

        # Debugging: Check loss progression
        print(f"Epoch {epoch + 1}, Batch {i//batch_size}: Loss = {loss.item()}")

        loss.backward() # backprop


        # Debugging: Check if gradients are updating
        for name, param in net.named_parameters():
            if param.requires_grad:
                print(f"{name}: Gradient Norm = {param.grad.norm()}")

        optimizer.step() # update model params


torch.save(net.state_dict(), "saved_model.pth")
print("Model saved successfully!")
