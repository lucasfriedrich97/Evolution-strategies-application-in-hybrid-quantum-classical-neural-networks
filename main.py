import numpy as np
import math as m
import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import quantumModel
import cv2
import os
from qiskit import  Aer
import DATA


from tqdm import trange
from tqdm import tqdm
from time import sleep

from qiskit import IBMQ
from qiskit.providers.aer.noise import NoiseModel


def Tresize(x,m,n):
    '''
    function used to shrink images
    input
        x: torch dim(28,28)
        m: int
        m: int
    output
        res: array dim(m,n)
    '''
    x = x.numpy()
    x = x.reshape(28,28)
    res = cv2.resize(x, dsize=(m, n), interpolation=cv2.INTER_CUBIC)
    return torch.from_numpy(res).reshape(m,n)
 


class Model1(nn.Module):
    def __init__(self,nq,dl):
        super(Model1, self).__init__()
        '''
        model 1 used in the article 
        '''
        self.q1 = quantumModel.Qlayer(5,4,3,encoder='RealAmplitude',re_uploading=False,input_grad=False,shots=200,l=dl)
        self.q2 = quantumModel.Qlayer(3,4,2,encoder='qubitEncoder',re_uploading=False,input_grad=True,shots=200,l=dl)

        

    def forward(self, x):

        x = self.q1(x)
        x = x.reshape(3)
        x = self.q2(x)
        
        return x



class Model2(nn.Module):
    def __init__(self,nq,dl):
        super(Model2, self).__init__()
        '''
        model 2 used in the article 
        '''
        self.q2 = quantumModel.Qlayer(4,4,4,encoder='qubitEncoder',re_uploading=False,input_grad=True,shots=200,l=dl)
        self.l1 = nn.Linear(784,4)
        self.l2 = nn.Linear(4,2)
        self.f = nn.Tanh()
        

    def forward(self, x):

        x = self.l1(x)
        x = self.f(x)
        x = x.reshape(4)
        x = self.q2(x)
        x = self.l2(x)
        x = self.f(x)
        
        return x




############################################ test function #############################

def test(model,nq, loss ,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            
            
            
            #data = Tresize(data,nq,nq)
            data = data.reshape(1,784)
            output = model(data)
            
            target1 = DATA.targetPro(target)

            test_loss += loss(output, target1).item() 
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss  , correct / len(test_loader.dataset)


############################################ Train function #############################

def train(model,train_loader,test_loader,optimizer,loss_func,nq,epochs):
    trainloss = []
    testeloss = []
    validation = []
    for epoch in range(1, epochs+1):
        sloss = 0
        acc = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f" {epoch}/{epochs}  ")
                    
                optimizer.zero_grad()
                    
                    
                
                #data = Tresize(data,nq,nq)# use this line for model 1
                data = data.reshape(1,784)# use this line for model 2
                output = model(data)

                target = DATA.targetPro(target)
                loss = loss_func(output, target)
                
                loss.backward()
                optimizer.step()
                sloss += loss.item()
                tepoch.set_postfix(loss=loss.item())
                    
            losst,acc=test(model,nq,loss_func,test_loader)
                
        print( '\nTrain set: Average loss: {:.4f} \n'.format( sloss/len(train_loader)))
        trainloss.append( sloss/len(train_loader) )
        validation.append(acc)
        testeloss.append(losst)

    validation = np.array(validation)
    trainloss = np.array(trainloss)
    testeloss = np.array(testeloss)
    return validation,trainloss,testeloss
        

######################################################################################

def main(epochs,lr,nq,train_loader,test_loader,dl,saveM):

    if not os.path.exists('./{}'.format(saveM)):
        os.mkdir('./{}'.format(saveM))


    if not os.path.exists('./{}/dataModel_Q.nq{}.lr{}.l{}'.format(saveM,nq,lr,dl)):
        os.mkdir('./{}/dataModel_Q.nq{}.lr{}.l{}'.format(saveM,nq,lr,dl))


    loss_func = nn.MSELoss()

    acc = []
    trainLoss = []
    testLoss = []
    epochsT = 4
    for ep in range(epochsT):
        print("###########################################################################")
        print("########################### ep: {}  lr: {} l: {} ##################################".format(ep+1,lr,dl))
        print("###########################################################################")

        model = Model1(nq,dl)
        #model = Model2(nq,dl)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        a1,a2,a3=train(model,train_loader,test_loader,optimizer,loss_func,nq,epochs)
        acc.append( a1 )
        trainLoss.append(a2)
        testLoss.append(a3)
        if ep <epochsT-1:
            del model
        else:
            torch.save(model.state_dict(), './{}/dataModel_Q.nq{}.lr{}.l{}/model.pth'.format(saveM,nq,lr,dl))
            del model



    np.savetxt( './{}/dataModel_Q.nq{}.lr{}.l{}/acc.txt'.format(saveM,nq,lr,dl), acc )
    np.savetxt( './{}/dataModel_Q.nq{}.lr{}.l{}/trainLoss.txt'.format(saveM,nq,lr,dl), trainLoss )
    np.savetxt( './{}/dataModel_Q.nq{}.lr{}.l{}/testLoss.txt'.format(saveM,nq,lr,dl), testLoss )


    param = np.array( [ nq,lr ] )
    np.savetxt( './{}/dataModel_Q.nq{}.lr{}.l{}/param.txt'.format(saveM,nq,lr,dl), param )


#######################################  data ##########################################

train_loader,test_loader = DATA.dataMNIST2(1000,100)


################################### main ################################### 

epochs = 100

nq = 4# used for the name of the directory where it will be saved

model = 'directory_name'

main(epochs,0.01,nq,train_loader,test_loader,2,model)
main(epochs,0.001,nq,train_loader,test_loader,2,model)
main(epochs,0.0001,nq,train_loader,test_loader,2,model)
