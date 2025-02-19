import numpy as np
import torch
from net_class import Net

# 50px x 50px
img_size = 50


net = Net()
net.load_state_dict(torch.load('saved_model.pth'))
net.eval()

testing_data = np.load("melanoma_testing_data.npy", allow_pickle=True)

test_X = torch.tensor(np.array([item[0] for item in testing_data]), dtype=torch.float32) / 255    # print image arrays into this tensor
test_Y = torch.tensor(np.array([item[1] for item in testing_data]), dtype=torch.float32)          # one-hot vector labels tensor


correct = 0
total = 0

# no auto tracking grads
with torch.no_grad():

    for i in range(len(test_X)):
        output = net(test_X[i].view(-1, 1, img_size, img_size))[0]

        if output[0] >= output[1]:
            guess = "Benign"
        else:
            guess = "Malignant"
        
        real_label = test_Y[i] # real label
        
        if real_label[0] >= output[1]:
            real_class = "Benign"
        else:
            real_class = "Malignant"
        
        if guess == real_class:
            correct += 1
        
        total += 1

print(f"Accuracy: {round(correct/total, 3)}") # output 0.843
