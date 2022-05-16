import numpy as np
import math as m
import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from qiskit import *
import qiskit
from qiskit import assemble,Aer
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
from tqdm import trange
import matplotlib.pyplot as plt
import quantumModel as ql



def myloss(x):
    return torch.mean(x)


NL = [2,4,8,15,30]


for nl in NL:
    dx = []
    dy = []
    for i in range(2,11):
        d = []
        x = torch.ones(i)*(m.pi/4)
        tp =  trange(1000)
        for j in tp:
            tp.set_description(f" {i}/{nl}  ")
            model = ql.Qlayer( i, nl , i )
            out = model(x)
            l = myloss(out)
            dw = torch.autograd.grad(l,model.parameters())
            d.append(dw[0][0])
        dx.append(i)
        dy.append(np.var(d))

    dx = np.array(dx)
    dy = np.array(dy)

    plt.plot(dx,dy,label='L = {} '.format(nl))
    np.savetxt( './dataVarNL_{}.txt'.format(nl), dy )

plt.legend()
plt.xlabel('n')
plt.ylabel('var')
plt.savefig('variancia_')
#plt.show()
