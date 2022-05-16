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
from qiskit.visualization import *
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator

##################################### quantum model ####################################

class QuantumClass:

    def __init__(self, nq,nl,nout,encoder = 'qubitEncoder', re_uploading=False , backend = AerSimulator(), shot=1024):
        '''
        nq: (int) number of qubit to be used
        nl: (int) number of layer to be used
        nout: (int) output size
        encoder: (string) encoding type
        re_uploading: boolean 
        '''
        self.nq = nq
        self.nl = nl
        self.nout = nout
        self.shot = shot
        self.cc = []
        self.backend = backend
        
        
        
        
        if encoder == 'qubitEncoder':
            self.zmap = False
        elif encoder == 'RealAmplitude':
            self.zmap = True
            
       
        if self.zmap:
            self.imput = { k : Parameter('imput{}'.format(k)) for k in range(self.nq**2) }
        else:
            self.imput = { k : Parameter('imput{}'.format(k)) for k in range(self.nq) }

               
        self.theta = { k : Parameter('theta{}'.format(k)) for k in range(self.nq*self.nl) }
            
        for cc in range(self.nout):
            self.q = QuantumRegister(self.nq)
            self.c = ClassicalRegister(1)
            self.qc = QuantumCircuit(self.q,self.c)
            if not re_uploading: 
                if self.zmap:
                    for i in range(self.nq):
                        for j in range(self.nq):
                            self.qc.ry( self.imput[i*self.nq+j] , self.q[j] )
                        if i+1 < self.nq:
                            for jj in range(self.nq-1):
                                for k in range(self.nq):
                                    if k>jj:
                                        self.qc.cx( self.q[jj], self.q[k] )   
                else:
                    for i in range(self.nq):
                        self.qc.ry( 2*self.imput[i] , self.q[i] )
                
                for i in range(self.nl):
                    for j in range(self.nq):
                        self.qc.ry( self.theta[i*self.nq+j] , self.q[j] )
                    for j in range(self.nq-1):
                        self.qc.cx(self.q[j],self.q[j+1])
                    
            else:
                for i in range(self.nl):

                    if self.zmap:
                        for ii in range(self.nq):
                            for j in range(self.nq):
                                self.qc.ry( self.imput[ii*self.nq+j] , self.q[j] )
                            if ii+1 < self.nq:
                                for j in range(self.nq-1):
                                    self.qc.cx( self.q[j], self.q[j+1] )   
                    else:
                        for ii in range(self.nq):
                            self.qc.ry( 2*self.imput[ii] , self.q[ii] )
                   

                    for j in range(self.nq):
                        self.qc.ry( self.theta[i*self.nq+j] , self.q[j] )
                    for j in range(self.nq-1):
                        self.qc.cx(self.q[j],self.q[j+1])
                    


            self.qc.measure(self.q[-self.nout+cc],self.c)
            self.cc.append(self.qc)
        
            
    def run(self,imput,theta):
        
        if self.zmap:
            imput = imput.reshape(self.nq*self.nq)
            params = { self.imput[k] : imput[k].item() for k in range(self.nq*self.nq) }
        else:
            imput = imput.reshape(self.nq)
            params = { self.imput[k] : imput[k].item() for k in range(self.nq) }
            
        theta = theta.reshape(self.nq*self.nl)
        params1 = { self.theta[k] : theta[k].item() for k in range(self.nq*self.nl) }
        params.update(params1)
        
        qobj = assemble(self.cc,shots=self.shot, parameter_binds = [ params ])
        
        job = self.backend.run(qobj)
        
        
            
        re = job.result().get_counts()
    
        
        if self.nout ==1:
            
            result = torch.zeros((1,1))
            try:
                
                result[0][0] = re['0']/self.shot
            except:
                result[0][0] = 0
            return result
        else:
            result = torch.zeros((1,len(re)))
            for i in range(len(re)):
                try:
                    result[0][i] = re[i]['0']/self.shot
                except:
                    result[0][i] = 0
            return result
                    
      
      
class TorchCircuitNES(Function):
    @staticmethod
    def forward(self, imput ,theta , quantumcircuit, input_grad = True, sigma = m.pi/24 , dl=None ):
        self.quantumcircuit = quantumcircuit
        result = self.quantumcircuit.run(imput,theta)
        self.n_qubit = self.quantumcircuit.nq
        self.layer = self.quantumcircuit.nl
        self.zmap = self.quantumcircuit.zmap
        self.input_grad = input_grad
        self.sigma = sigma
        self.dl = dl
        self.save_for_backward(result,imput, theta)
        

        return result.float()

    @staticmethod
    def backward(self, grad_output):

        forward_tensor,imput1, theta1 = self.saved_tensors

        gradInput = None
        if self.input_grad :
            input_numbers1 = theta1
            input_numbers1 = input_numbers1.reshape(self.n_qubit*self.layer)

            input_numbers2 = imput1
            if self.zmap:
                input_numbers2 = input_numbers2.reshape(self.n_qubit**2)
            else:
                input_numbers2 = input_numbers2.reshape(self.n_qubit)


            sigma = self.sigma
            if not self.dl:
                if  self.n_qubit*self.layer >= self.n_qubit**2 :
                    l = int( (4+3*np.log(self.n_qubit*self.layer)) )
                else:
                    l = int( (4+3*np.log(self.n_qubit**2))  )
            else:
                l = self.dl

            
            media1 = 0
            soma1 = 0
            media2 = 0
            soma2 = 0
            mm1 = torch.distributions.multivariate_normal.MultivariateNormal(input_numbers1,torch.eye( len(input_numbers1) )*(sigma**2) )
            mm2 = torch.distributions.multivariate_normal.MultivariateNormal(input_numbers2,torch.eye( len(input_numbers2) )*(sigma**2) )
            w1 = input_numbers1.reshape(len(input_numbers1),1)
            w2 = input_numbers2.reshape(len(input_numbers2),1)
            for k in range( l ):
                xi1 = mm1.sample()
                xi2 = mm2.sample()
                d0 =  self.quantumcircuit.run(xi2,xi1)
                xi1 = xi1.reshape(len(input_numbers1),1)
                xi2 = xi2.reshape(len(input_numbers2),1)
                soma1+= (xi1-w1)*d0
                soma2+= (xi2-w2)*d0
                
            media1 = soma1/(l*sigma**2)
            media1 = media1.float()
            
            result = torch.matmul( media1, grad_output.T)
            result = result.reshape(self.n_qubit*self.layer)
        
            
            media2 = soma2/(l*sigma**2)
            media2 = media2.float()
            result1 = torch.matmul( media2, grad_output.T)
            result1 = result1.reshape( len(input_numbers2) )
        
          
            
            gradInput = result1


        else:
            input_numbers1 = theta1
            input_numbers1 = input_numbers1.reshape(self.n_qubit*self.layer)
            sigma = self.sigma
            if not self.dl:
                l = int( (4+3*np.log(self.n_qubit*self.layer))  )
            else:
                l = self.dl
            
            
            media1 = 0
            soma1 = 0
            mm1 = torch.distributions.multivariate_normal.MultivariateNormal(input_numbers1,torch.eye( len(input_numbers1) )*(sigma**2) )
            w1 = input_numbers1.reshape(len(input_numbers1),1)
            for k in range( l ):
                xi1 = mm1.sample()
                d0 =  self.quantumcircuit.run(imput1,xi1)
                xi1 = xi1.reshape(len(input_numbers1),1)
                soma1+= (xi1-w1)*d0

                
                
            media1 = soma1/(l*sigma**2)
            media1 = media1.float()
            result = torch.matmul( media1, grad_output.T)
            result = result.reshape(self.n_qubit*self.layer)
        
        
        
        
        return  gradInput,result,None,None,None,None






class Qlayer(nn.Module):

   
    def __init__(self,n_qubit,n_layer,
                 nout,
        encoder = 'qubitEncoder',
        re_uploading = False,
        input_grad = False ,
        backend=AerSimulator(),
        shots = 1024,
        sigma  = m.pi/24,
                l=None):
        super(Qlayer, self).__init__()
        """
        input
            n_qubit: (int)  number of qubit to be used
            n_layer: (int)  number of layer to be used
            encoder: (str) encoding type
              'qubitEncoder': |x> =  Ry(x_i)|0>

              'RealAmplitude':  ┌──────────┐ ░                 ░ ┌──────────┐ ░                 ░ ┌──────────┐
                                ┤ RY(x[0]) ├─░───■────■────────░─┤ RY(x[3]) ├─░───■────■────────░─┤ RY(x[6]) ├
                                ├──────────┤ ░ ┌─┴─┐  │        ░ ├──────────┤ ░ ┌─┴─┐  │        ░ ├──────────┤
                                ┤ RY(x[1]) ├─░─┤ X ├──┼────■───░─┤ RY(x[4]) ├─░─┤ X ├──┼────■───░─┤ RY(x[7]) ├
                                ├──────────┤ ░ └───┘┌─┴─┐┌─┴─┐ ░ ├──────────┤ ░ └───┘┌─┴─┐┌─┴─┐ ░ ├──────────┤
                                ┤ RY(x[2]) ├─░──────┤ X ├┤ X ├─░─┤ RY(x[5]) ├─░──────┤ X ├┤ X ├─░─┤ RY(x[8]) ├
                                └──────────┘ ░      └───┘└───┘ ░ └──────────┘ ░      └───┘└───┘ ░ └──────────┘
                re_uploading: (boolean) if True reload data after each layer
                                        if False just load data on input
                                        more information: https://arxiv.org/pdf/1907.02085.pdf

                input_grad: (boolean) if True derive with respect to input data
                                      if False do not derive with respect to input data
                backend: choose simulator()
                shots: (int)number of repetitions needed to obtain the averages
                sigma: (float) 
                
    """
       
            
        if  nout>n_qubit:
            raise ValueError(
                "nout must be less than or equal to n_qubit"
            )
        else:
            self.quantum_circuit = QuantumClass(n_qubit,n_layer,nout,encoder, re_uploading,backend,shots)
            self.alfa = torch.nn.Parameter(torch.FloatTensor(n_qubit*n_layer).uniform_(-m.pi, m.pi))
            self.input_grad = input_grad
            self.sigma = sigma
            self.l = l

        
        
       

    def forward(self,input):
        
        return TorchCircuitNES.apply( input,self.alfa,self.quantum_circuit,self.input_grad,self.sigma,self.l )
      
          
            

