
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 14:59:34 2019

@author: samar
"""


import torch
import numpy as np
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import timeit
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib.pyplot as plt

# Define CNN architecture and forward pass
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 12, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(12),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,stride=2))
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(12, 24, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(24),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,stride=2))
        self.fc1 = torch.nn.Linear(4*4*24, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
        
    def forward_pass(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = (self.fc1(out))
        out = (self.fc2(out))
        out = (self.fc3(out))
        return out
    

class ConvNet:
    # Initialize the class
    def __init__(self, X, Y):  
        #print(self.device)

#        # Check if there is a GPU available
#        if torch.cuda.is_available() == True:
#            self.dtype_double = torch.cuda.FloatTensor
#            self.dtype_int = torch.cuda.LongTensor
#        else:
        self.dtype_double = torch.FloatTensor
        self.dtype_int = torch.LongTensor
        
        # Define PyTorch dataset
        X = torch.from_numpy(X).type(self.dtype_double) # num_images x num_pixels_x x num_pixels_y
        Y = torch.from_numpy(Y).type(self.dtype_int) # num_images x 1
        self.train_data = torch.utils.data.TensorDataset(X, Y)
        
        # Define architecture and initialize

        self.net=CNN()
        
        # Define the loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        
           
    # Trains the model by minimizing the Cross Entropy loss
    def train(self, num_epochs = 20, batch_size = 128):
        
        # Create a PyTorch data loader object
        self.trainloader = torch.utils.data.DataLoader(self.train_data, 
                                                  batch_size=batch_size, 
                                                  shuffle=True)

        start_time = timeit.default_timer()
        for epoch in range(num_epochs):
            for it, (images, labels) in enumerate(self.trainloader):
                images = Variable(images)
                labels = Variable(labels)
        
                # Reset gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.net.forward_pass(images)
                
                # Compute loss
                loss = self.loss_fn(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                self.optimizer.step()

                if (it+1) % 100 == 0:
                    elapsed = timeit.default_timer() - start_time
                    print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Time: %2fs' 
                           %(epoch+1, num_epochs, it+1, len(self.train_data)//batch_size, loss.cpu().data, elapsed))
                    start_time = timeit.default_timer()             
                    
    def test(self, X, Y):
        # Define PyTorch dataset
        X = torch.from_numpy(X).type(self.dtype_double) # num_images x num_pixels_x x num_pixels_y
        Y = torch.from_numpy(Y).type(self.dtype_int) # num_images x 1
        test_data = torch.utils.data.TensorDataset(X, Y)
       
        # Create a PyTorch data loader object
        test_loader = torch.utils.data.DataLoader(test_data, 
                                                  batch_size=128, 
                                                  shuffle=True)
        
        # Test prediction accuracy
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = Variable(images)
            outputs = self.net.forward_pass(images)
            _, predicted = torch.max(outputs.cpu().data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        print('Test Accuracy of the model on the %d test images: %.5f %%' % (len(test_data), 100.0 * correct / total))
        print('correct:%d total:%d'%(correct,total))
        
    
    # Evaluates predictions at test points    
    def predict(self, X_star):
        X_star = torch.from_numpy(X_star).type(self.dtype_double) 
        X_star = Variable(X_star, requires_grad=False)
        y_star = self.net.forward_pass(X_star)
        y_star = y_star.cpu().data.numpy()
        return y_star
    
    
if __name__ == "__main__": 
    
    def plot_random_sample(images, labels):
        idx = np.random.randint(images.shape[0])
        plt.figure(1)
        plt.clf()
        img = images[idx,:,:,:]
        plt.imshow(np.transpose(img,(1,2,0)))
        plt.title('This is a %d' % labels[idx])
        plt.show()
        

    # Load the dataset
    def load_cfar10_train_batch(cifar10_dataset_folder_path, batch_id):
        with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
            # note the encoding type is 'latin1'
            batch = pickle.load(file, encoding='latin1')
        batch_temp=np.array(batch['data'])    
        features = batch_temp.reshape((len(batch_temp), 3, 32, 32))
        labels = np.array(batch['labels'])
        return features, labels
    
    def load_cfar10_test_batch(cifar10_dataset_folder_path):
        with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
            # note the encoding type is 'latin1'
            batch = pickle.load(file, encoding='latin1')
        batch_temp=np.array(batch['data'])    
        features = batch_temp.reshape((len(batch_temp), 3, 32, 32))
        labels = np.array(batch['labels'])
        return features, labels
    
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for i in range(1,6):
        batch,l=load_cfar10_train_batch("cifar-10-batches-py",i)
        l=np.array(l)
        if i==1:
            train_images=batch
            train_labels=l
        else:
            train_images=np.concatenate((train_images, batch), axis=0)
            train_labels=np.concatenate((train_labels, l), axis=0)
    
    test_images,test_label=load_cfar10_test_batch("cifar-10-batches-py")
    test_labels=np.array(test_label)    

    # Check a few samples to make sure the data was loaded correctly
    plot_random_sample(train_images, train_labels)
    
    # Define model
    model = ConvNet(train_images, train_labels)
    # Train
    model.train(num_epochs = 20, batch_size = 128)
    # Evaluate test performance
    model.test(test_images, test_labels)
    print("done 1")
    #Predict
    predicted_labels = np.argmax(model.predict(test_images),1)
    
    print("done 2")
    
    # Plot a random prediction
    idx = np.random.permutation(predicted_labels.shape[0])[0]
    plt.figure(1)
    img = test_images[idx,:,:,:]
    plt.imshow(np.transpose(img,(1,2,0)))
    print('Correct label: %d, Predicted label: %d' % (test_labels[idx], predicted_labels[idx]))
    plt.title('This is a %d' % predicted_labels[idx])
    plt.show()
    
        #find confusion matrix        
    confuse=confusion_matrix(test_labels, predicted_labels)
    