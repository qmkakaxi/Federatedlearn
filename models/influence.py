import six
import torch
from torch.autograd import grad
from  models import utility
from torch.autograd import Variable
import torch.nn.functional as F
import random

def hvp(y, w, v):
    first_grads = grad(y, w, retain_graph=True, create_graph=True)
    grad_v = 0
    for g, v in six.moves.zip(first_grads, v):
        grad_v += torch.sum(g * v)
    grad(grad_v, w, create_graph=True)
    return grad(grad_v, w, create_graph=True)

def grad_z(z, t, model, device,create_graph=True,lossfunction = F.nll_loss):
    model.eval()
    z, t = Variable(z, volatile=False).to(device), Variable(t, volatile=False).to(device)
    y = model(z)
    loss = lossfunction(y, t)
    return (grad(loss, list(model.parameters()), create_graph=create_graph))



def stest(v,model,z_loader,device,damp=0.01,scale=25.0,repeat=5,lossfunction = F.nll_loss):
    h_estimate=v
    train_set=z_loader
    for i in utility.create_progressbar(repeat, desc='s_test'):
        j=random.randint(0,len(z_loader))
        data, target= train_set.dataset[j]
        data = train_set.collate_fn([data])
        target= train_set.collate_fn([target])
        x, t = Variable(data, volatile=False).to(device), Variable(target, volatile=False).to(device)
        y = model(x)
        loss = lossfunction(y, t)
        hv = hvp(loss, list(model.parameters()), h_estimate)
        h_estimate = [_v + (1 - damp) * h_estimate - _hv / scale for _v, h_estimate, _hv in six.moves.zip(v, h_estimate, hv)]

    return h_estimate


