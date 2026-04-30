import torch
import numpy as np
import GuassModelCuda
import h5py
import time
from collections import defaultdict
import hdf5plugin
import numpy as np
import tifffile as tf
import os
import shutil
class Run():
    def __init__(self,img,config):
        self.max_value = img.max()
        self.min_value = img.min()
        self.img = ((img-self.min_value)/(self.max_value-self.min_value)).float().cuda()+1e-5
        self.loss_aim = config['paras']['loss_aim']
        self.shape = img.shape
        self.step = config['paras']['init_step']
        self.Max = config['data']['Max']
        self.lr_init = config['paras']['lr_init']
        self.lr_final = config['paras']['lr_final']
        self.iternum = config['paras']['iternum']
        self.thredhold = config['paras']['thredhold']
        self.prune_step = config['paras']['prune_step']
        self.distance = config['paras']['distance_scale']*self.step
        
        self.prune_threhold = config['paras']['prune_threhold']/self.max_value
        self.prune_start_loss = config['paras']['prune_start_loss']
        self.every_n = config['paras']['every_n']
        self.compress_method = config['data']['compress_method']

    def Run(self,paras = None):
        if paras ==None:
            GM = GuassModelCuda.GuassModel_cuda()
            GM.init_paras(self.img,step = self.step,thredhold_value = self.prune_threhold)
        else:
            GM = GuassModelCuda.GuassModel_cuda(paras,self.shape,self.max_value)
        
        lr_get = GM.get_expon_lr_func(lr_init=self.lr_init, lr_final=self.lr_final,max_steps=self.iternum)

        t0 = time.time()
        loss = torch.tensor(1)
        if os.path.exists('Temp'):
            shutil.rmtree('Temp')
        os.mkdir('Temp')
        for i in range (self.iternum):    
            #grad = GM.e2e(gt = self.img, distance = self.distance , thredhold = self.thredhold)    
            grad = GM.sparse_e2e(gt = self.img, distance = self.distance , thredhold = self.thredhold,PT = self.every_n,i=i)
            grad = torch.nan_to_num(grad, nan=0.0)
            grad = torch.clip(grad,-10,10)
            
            
            if np.mod(i+1,100) == 0:
                ## 每100轮输出一次
                pred = GM.forward(distance = self.distance , thredhold = self.thredhold)
                loss = torch.mean((pred*self.max_value/self.Max-self.img*self.max_value/self.Max)**2)
                print('Compress step:%d, time:%.4fs, loss:%8.3e, kernal number:%d        '%(i+1,time.time()-t0,loss.detach().cpu().half().numpy(),GM.num))
                if i == 99:
                    with open('Temp/log.txt','w') as f:
                        f.write('%d %.4f %.8f %d \n'%(i+1,time.time()-t0,loss.detach().cpu().half().numpy(),GM.num))
                else:
                    with open('Temp/log.txt','a') as f:
                        f.write('%d %.4f %.8f %d \n'%(i+1,time.time()-t0,loss.detach().cpu().half().numpy(),GM.num))
            if np.mod(i+1,1000) == 0:
                pred_np = (pred*(self.max_value-self.min_value)+self.min_value).detach().cpu().numpy()
                tf.imwrite('Temp/pred_%d.tif'%i,pred_np)
                paras = GM.paras.detach().cpu().numpy()
                np.savetxt('Temp/paras.txt',np.round(paras.transpose(1,0),decimals=2),fmt='%.2f', delimiter='\t')
                

                

            if loss>self.prune_start_loss:
                GM.paras_step(grad)
                GM.optimizer.zero_grad()   
            else:
                if loss<self.loss_aim: break
                if np.mod(i,self.prune_step)!=0:
                    GM.paras_step(grad)
                    GM.optimizer.zero_grad()
                else:
                    with torch.no_grad():
                        GM.value_prune(self.prune_threhold)
                        GM.value_new_pool(pred,self.img,self.step,self.prune_threhold/2)
                        

            # 学习率更新
            for param_group in GM.optimizer.param_groups:
                param_group['lr'] = lr_get(i)

            ## 错误判断
            if torch.isnan(GM.paras.data).any() | torch.isnan(grad).any(): 
                print('errors!')
                break
            if torch.isinf(loss).any() | torch.isinf(grad).any(): 
                print('errors!')
                break
        self.paras =  GM.paras.data

    def compress_methods(self):
        blosc_settings = defaultdict(lambda: {})
        for ctype in ['lz4', 'lz4hc', 'zlib', 'zstd']:
            for shuffle,shufflep in [('shuffle', 1), ('bitshuffle', 2)]:
                for clevel in [3,5,8]: #[3,5,8]:
                    key = f'BLOSC_{clevel}_{shuffle}_{ctype}'
                    toadd = {**blosc_settings[key], **hdf5plugin.Blosc(cname=ctype, clevel=clevel, shuffle=shufflep)}
                    
                    blosc_settings[key] = toadd
                    
        gzip_settings = defaultdict( lambda: {'compression':'gzip'})
        for shuffle,shufflep in [(None, False), ('shuffle', True)]:
            for clevel in [3,5,8]: #[3,5,8]:
                key = f'GZIP_{clevel}_{shuffle}'
                toadd = {**gzip_settings[key], 'shuffle':shufflep, 'compression_opts':clevel}
                gzip_settings[key] = toadd

        settings = {
            'Uncompressed':{},
            'LZF': {'compression':'lzf'},
            'LZF Shuffle': {'compression':'lzf', 'shuffle':True},
            'LZ4': {**hdf5plugin.LZ4(nbytes=0)},
            **{f'ZSTD_{cl}':{**hdf5plugin.Zstd(clevel=cl)} for cl in [3, 10, 15, 20, 22]},
            **gzip_settings,
            **blosc_settings
        }
        self.methods = settings
    def save_h5(self,filename):
        self.compress_methods()
        with h5py.File(filename, 'w') as h5file:
            grp = h5file.create_group(str(0))
            tmp = np.around(self.paras.detach().cpu().half().numpy(),decimals=2)
            grp.create_dataset("data", data=tmp, **self.methods[self.compress_method])
            h5file.attrs['max_value'] = self.max_value.detach().cpu().half().numpy()
            h5file.attrs['min_value'] = self.min_value.detach().cpu().half().numpy()
            h5file.attrs['shape'] = self.shape
            h5file.attrs['distance'] = self.distance
        return 
    
    def load_h5(filename):

        with h5py.File(filename, 'r') as h5file:
            data = h5file[str(0)]['data'][:]
        return data
    
    def Render(self,filename):
        with h5py.File(filename, 'r') as h5file:
            paras = h5file[str(0)]['data'][:]
            max_value = h5file.attrs['max_value']
            min_value = h5file.attrs['min_value']
            shape = h5file.attrs['shape']
            distance = h5file.attrs['distance']

        GM = GuassModelCuda.GuassModel_cuda(paras,shape,max_value)
        res = GM.forward(distance,1e-8)

        return res.detach().cpu().numpy()*(max_value-min_value)+min_value

