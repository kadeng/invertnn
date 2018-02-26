
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import OrderedDict
import pandas
import torch.distributions as distributions
import seaborn as sns
from sklearn import datasets

from torch.nn.utils.clip_grad import clip_grad_norm
# In[19]:

import invertnn.pytorch.invertible_transforms as invt
import invertnn.pytorch.weight_reparametrization as weights


# In[3]:

def random_covariance(n):
    tmp = torch.rand((n,n)).sqrt().mul(2.0).sub(1.0).triu()
    k = torch.matmul(tmp, tmp.t())
    for i in range(n):
        k[i,i] = abs(k[i,i])
    return k
    


# In[4]:

prior = distributions.MultivariateNormal(torch.rand((3)),covariance_matrix=random_covariance(3))
prior.covariance_matrix


# In[5]:

xyz = prior.sample((10000,))
np_xyz = xyz.numpy()
data = pandas.DataFrame({ 'x' : np_xyz[:,0], 'y' : np_xyz[:,1], 'z' : np_xyz[:,2]})


# In[7]:

sns.jointplot(x=data.x, y=data.y, kind="hex", color="k");


# In[8]:

sns.jointplot(x=data.y, y=data.z, kind="hex", color="k");


# In[9]:

sns.jointplot(x=data.x, y=data.z, kind="hex", color="k");


# In[10]:

import math


# In[11]:

def actual_transform(xyz):
    xyz[:,2] = torch.sin(xyz[:,0]*math.pi*2.0+xyz[:,1]*math.pi)+xyz[:,2]*0.1
    return xyz


# In[12]:

txyz = actual_transform(xyz.clone())


# In[13]:

np_txyz = txyz.numpy()
tdata = pandas.DataFrame({ 'x' : np_txyz[:,0], 'y' : np_txyz[:,1], 'z' : np_txyz[:,2]})


# In[14]:

sns.jointplot(x=data.x, y=data.z, kind="hex", color="k");


# In[15]:

sns.jointplot(x=tdata.x, y=tdata.z, kind="hex", color="k");


# In[16]:

sns.jointplot(x=tdata.y, y=tdata.z, kind="hex", color="k");


# In[17]:

noisy_circles = datasets.make_circles(n_samples=10000, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=10000, noise=.05)
blobs = datasets.make_blobs(n_samples=10000, random_state=8)


# In[33]:

class SimpleMLP(nn.Module):
    
    def __init__(self, n, m, hidden=[10,10,10], final_activation=None):
        super().__init__()
        self.n = n
        self.m = m
        self.hidden = hidden
        self.activation = nn.SELU()
        layers = [ 
                  
                 ]
        for i in range(len(hidden)):
            if i==0:
                k = n
            else:
                k = hidden[i-1]
            layers.append(weights.centered_weight_norm(nn.Linear(k, hidden[i])))
            layers.append(nn.SELU())
        layers.append(nn.Linear(hidden[-1], m))
        if final_activation is not None:
            layers.append(final_activation)
        self.net = nn.Sequential(*layers)
    
    def forward(self, input):
        return self.net(input)


# In[42]:

# We constrain the output range of s via Softsign for numerical reasons, because it's result gets exponentiated
invnn = invt.InvertibleSequential(OrderedDict([
    ('l1', invt.InvertibleCouplingLayer(D=3, d=2, odd_layer=True, s=SimpleMLP(2,1, final_activation=nn.Softsign()), t=SimpleMLP(2,1))),
    ('l2', invt.InvertibleCouplingLayer(D=3, d=2, odd_layer=False, s=SimpleMLP(2,1, final_activation=nn.Softsign()), t=SimpleMLP(2,1))),
    ('l3', invt.InvertibleCouplingLayer(D=3, d=1, odd_layer=True,s=SimpleMLP(1,2, final_activation=nn.Softsign()), t=SimpleMLP(1,2))),
    ('l4', invt.InvertibleCouplingLayer(D=3, d=1, odd_layer=False,s=SimpleMLP(1,2, final_activation=nn.Softsign()), t=SimpleMLP(1,2))),
]))




prior_critic = SimpleMLP(3,1)
observation_critic = SimpleMLP(3,1)
invnn.cuda(2)
prior_critic.cuda(2)
observation_critic.cuda(2)


# In[48]:




# In[43]:

generator_opt = optim.Adam(params = invnn.parameters(), lr=0.0002)
prior_critic_opt = optim.Adam(params=prior_critic.parameters(), lr=0.0002)
observation_critic_opt = optim.Adam(params=observation_critic.parameters(), lr=0.0002)


# In[46]:

batchsize = (1000,)


# In[50]:

for i in range(10000):
    # Improve Observation Critic
    observation_critic_opt.zero_grad()
    noise = prior.sample(batchsize).cuda(2)
    generated = invnn.forward(noise)
    observation = actual_transform(prior.sample(batchsize).cuda(2))
    observation_score = observation_critic.forward(generated).mean()-observation_critic.forward(observation).mean()
    observation_score.backward()
    gnorm = clip_grad_norm(observation_critic.parameters(), 100.0)
    if np.isfinite(gnorm):
        observation_critic_opt.step()
    
    # Improve Reconstruction Critic
    prior_critic_opt.zero_grad()
    noise = prior.sample(batchsize).cuda(2)
    observation = actual_transform(prior.sample(batchsize).cuda(2))
    reconstructed_noise = invnn.invert(observation)
    reconstructed_score = prior_critic.forward(reconstructed_noise).mean()-prior_critic.forward(noise).mean()
    reconstructed_score.backward()
    gnorm = clip_grad_norm(prior_critic.parameters(), 100.0)
    if np.isfinite(gnorm):
        prior_critic_opt.step()
    
    # Improve Generator
    generator_opt.zero_grad()
    noise = prior.sample(batchsize).cuda(2)
    generated = invnn.forward(prior.sample(batchsize).cuda(2))
    observation = actual_transform(prior.sample(batchsize).cuda(2))
    reconstructed_noise = invnn.invert(observation)
    neg_generated_score = -observation_critic.forward(generated).mean()+observation_critic.forward(observation).mean()
    
    neg_reconstructed_score = -prior_critic.forward(reconstructed_noise).mean()+prior_critic.forward(prior.sample(batchsize).cuda(2)).mean()
    gen_score = neg_generated_score+neg_reconstructed_score
    gen_score.backward()
    gnorm = clip_grad_norm(invnn.parameters(), 100.0)
    if np.isfinite(gnorm):
        generator_opt.step()


    print("Scores: observation_score=%.6f - reconstructed_score: %.6f - gen_score: %.6f" % (observation_score.item(),
                                                                                        reconstructed_score.item(),
                                                                                        gen_score.item()))
    
    
    
    

