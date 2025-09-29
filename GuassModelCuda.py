import torch
import numpy as np
from torch import nn
import tifffile as tf
import time
from CUDA_kernal_3dgrc import AccRendering,AccRendering_Gradient,AccRendering_e2e,AccRendering_sparse_e2e


class GuassModel_cuda(nn.Module):
    def __init__(self,paras = None, size = None,max_value = None):
        super( GuassModel_cuda, self ).__init__()             
        if paras is None:
            pass
        else:
            self.num = paras.shape[1]
            self.size = size
            self.paras =  nn.Parameter(torch.tensor(paras,dtype=torch.float32).cuda().requires_grad_(True))
            self.optimizer = torch.optim.Adam([self.paras])
            self.max_value = max_value


    def create_guass(self,img,step,thredhold_value,method='mean'):
        kernal_size = step*2+1
        if torch.tensor(img.shape).min()<kernal_size:
            kernal_size = (torch.tensor(img.shape).min()//2)*2-1
        if method=='mean':
            mean = torch.nn.functional.avg_pool3d(img.unsqueeze(0).unsqueeze(0),kernel_size=int(kernal_size),stride=step,padding=int((kernal_size-1)//2)).squeeze(0).squeeze(0)
            length = mean.shape[0]*mean.shape[1]*mean.shape[2]
            value = mean.reshape([length,1])
            r = torch.ones([length,6])*step/2
        elif method=='max':
            mean = torch.nn.functional.max_pool3d(img.unsqueeze(0).unsqueeze(0),kernel_size=int(kernal_size),stride=step,padding=int((kernal_size-1)//2)).squeeze(0).squeeze(0)
            length = mean.shape[0]*mean.shape[1]*mean.shape[2]
            value = mean.reshape([length,1])
            r = torch.ones([length,6])*2
        
        
        a = torch.arange(mean.shape[0])
        b = torch.arange(mean.shape[1])
        c = torch.arange(mean.shape[2])
        a,b,c = torch.meshgrid(a,b,c)
        a = ((a+1)*step - step//2 ).reshape([length,1])
        b = ((b+1)*step - step//2 ).reshape([length,1])
        c = ((c+1)*step - step//2 ).reshape([length,1])


        
        r[:,1] = 1e-5
        r[:,2] = 1e-5
        r[:,4] = 1e-5

        paras = torch.cat([a.cuda(),b.cuda(),c.cuda(),value,r.cuda()],dim=1)
        mask = paras[:,3]>thredhold_value

        return paras[mask].transpose(1,0)

    def init_paras(self,img,step = 4,thredhold_value=0.001):

        paras = self.create_guass(img,step,thredhold_value)
      
        self.paras = paras
        self.num = self.paras.shape[1]
        self.size = img.shape
        self.optimizer = torch.optim.Adam([self.paras])

    def forward(self,distance,thredhold):
        if self.num==0:
            return torch.zeros([*self.size]).cuda()
        paras_ = self.paras.clone()
        position = paras_[:3,:].contiguous()
        value = paras_[3,:].reshape(1,self.num).contiguous()
        r = paras_[4:,:].contiguous()
        return AccRendering(*self.size,position,value,r,distance,thredhold)
        
    def backward(self,rgrad,distance,thredhold):
        if self.num==0:
            return torch.zeros_like(self.paras).cuda()
        paras_ = self.paras.clone()
        position = paras_[:3,:].contiguous()
        value = paras_[3,:].reshape(1,self.num).contiguous()
        r = paras_[4:,:].contiguous()
        return AccRendering_Gradient(rgrad,position,value,r,distance,thredhold)
    
    def e2e(self,gt,distance,thredhold):
        if self.num==0:
            return torch.zeros_like(self.paras).cuda()
        paras_ = self.paras.clone()
        position = paras_[:3,:].contiguous()
        value = paras_[3,:].reshape(1,self.num).contiguous()
        r = paras_[4:,:].contiguous()
        return AccRendering_e2e(gt,position,value,r,distance,thredhold,thredhold)
    
    def sparse_e2e(self,gt,distance,thredhold,i,PT):
        if self.num==0:
            return torch.zeros_like(self.paras).cuda()
        paras_ = self.paras.clone()
        position = paras_[:3,:].contiguous()
        value = paras_[3,:].reshape(1,self.num).contiguous()
        r = paras_[4:,:].contiguous()
        return AccRendering_sparse_e2e(gt,position,value,r,distance,thredhold,thredhold,PT,i%PT)

        
    def paras_step(self,grad):
        self.paras.grad = grad
        self.optimizer.step()

 
    def value_prune(self,value_min):
        mask = self.paras[3]>value_min
        paras = self.paras[:,mask].data.clone()
        self.num = paras.shape[1]
        self.paras =  nn.Parameter(torch.tensor(paras,dtype=torch.float32).cuda().requires_grad_(True))
        self.optimizer = torch.optim.Adam([self.paras])

    def value_new_pool(self,pred,gt,step,thredhold_value):
        diff = gt - pred
        new_guass = self.create_guass(diff,step,thredhold_value,'mean')
        paras = self.paras.data.clone()
        paras_new = torch.cat([paras,new_guass.cuda()],dim=1)
        self.num = paras_new.shape[1]
        self.paras =  nn.Parameter(torch.tensor(paras_new,dtype=torch.float32).cuda().requires_grad_(True))
        self.optimizer = torch.optim.Adam([self.paras])


    def get_expon_lr_func(self,lr_init=10, lr_final=0.1, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
        """
        Copied from Plenoxels

        Continuous learning rate decay function. Adapted from JaxNeRF
        The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
        is log-linearly interpolated elsewhere (equivalent to exponential decay).
        If lr_delay_steps>0 then the learning rate will be scaled by some smooth
        function of lr_delay_mult, such that the initial learning rate is
        lr_init*lr_delay_mult at the beginning of optimization but will be eased back
        to the normal learning rate when steps>lr_delay_steps.
        :param conf: config subtree 'lr' or similar
        :param max_steps: int, the number of steps during optimization.
        :return HoF which takes step as input
        """

        def helper(step):
            if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
                # Disable this parameter
                return 0.0
            if lr_delay_steps > 0:
                # A kind of reverse cosine decay.
                delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                    0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
                )
            else:
                delay_rate = 1.0
            t = np.clip(step / max_steps, 0, 1)
            log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return delay_rate * log_lerp

        return helper
    
        

    

        

