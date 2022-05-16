import numpy as np
import torch
from torchvision import datasets, transforms
"""
This code will be used to process the data from the MNIST data set. 
Data referring to images of zeros and ones digits will be considered.
"""

def dataMNIST2(ntrain,ntest):
	'''
	input
		ntrain:(int) number of training data, for example, if ntrain=10, 10 images of zero digits and 10 of digits 1 will be used
		ntest: (int) number of validation data, for example, if ntrain=4, 4 images of zero digits and 4 of digits 1 will be used
	'''
	################################### data train #########################################3

	
	n_samples = ntrain

	img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
	])

	X_train = datasets.MNIST(root='./data', train=True, download=True,
	                         transform=img_transform)


	# Leaving only labels 0 and 1 
	idx = np.append(np.where(X_train.targets == 0)[0][:n_samples], 
	                np.where(X_train.targets == 1)[0][:n_samples])

	X_train.data = X_train.data[idx]
	X_train.targets = X_train.targets[idx]

	train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)



	#############################  data  teste   ###############################################

	n_samples = ntest

	X_test = datasets.MNIST(root='./data', train=False, download=True,
	                        transform=img_transform)

	idx = np.append(np.where(X_test.targets == 0)[0][:n_samples], 
	                np.where(X_test.targets == 1)[0][:n_samples])

	X_test.data = X_test.data[idx]
	X_test.targets = X_test.targets[idx]

	test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)

	return train_loader,test_loader



def targetPro(x):
    if x.item() == 0:
        y = torch.zeros((1,2))
        y[0][0] = 1
        return y

    elif x.item() == 1:
        y = torch.zeros((1,2))
        y[0][1] = 1
        return y
  
