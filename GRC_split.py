# python GRC.py -c config/default.yaml -g 0
from GRC_Run import Run
from omegaconf import OmegaConf
from datetime import datetime
import os
import tifffile as tf
import shutil
import argparse
import numpy as np
import torch
import warnings
import evaluate
import split_overlap
import time
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3DGRC")
    parser.add_argument('-c', 
                        type=str, 
                        default='config/default_split.yaml', help='Path to config file')
    parser.add_argument('-g',type=str, default='0', help='gpu')
    args = parser.parse_args()
    print("config_path")
    config_path = os.path.abspath(args.c)
    # Make the gpu index used by CUDA_VISIBLE_DEVICES consistent with the gpu index shown in nvidia-smi
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Specify the gpu index to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = args.g

    config = OmegaConf.load(config_path)
    timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S.%f")[:-3]
    output_dir = config.data.output_dirname + timestamp
    os.makedirs(output_dir)
    print(f"All results wll be saved in {output_dir}")
    OmegaConf.save(config, output_dir+"/config.yaml")
    raw = np.array(tf.imread(config.data.path),dtype=np.float32)

    if config.data.splitwidth != 0:
        raws, shape, block_size, overlap, num_blocks = split_overlap.split_3d_image_overlap(raw, 
                                                                        block_size=config.data.splitwidth, 
                                                                        overlap=config.paras.init_step)
        t0 = time.time()
        t_render = 0
        Ress = []
        Original_gaussian = 0
        FInal_gaussian = 0
        for i in range(len(raws)):
            print("Processing %d/%d"%(i+1,len(raws)))
            img = torch.from_numpy(raws[i]).float().cuda()
            Com = Run(img,config)
            original_number, final_number = Com.Run()
            Original_gaussian += original_number
            FInal_gaussian += final_number
            Com.save_h5(output_dir+'/3dgrc_%d.h5'%i)
            t_render_0 = time.time()
            Res = Com.Render(output_dir+'/3dgrc_%d.h5'%i)
            t_render += time.time()-t_render_0
            Res = np.array(Res,dtype=np.float32)
            Ress.append(Res)
            tf.imwrite(output_dir+'/3dgrc_%d.tif'%i,Res)
        Res = split_overlap.combine_3d_blocks_overlap(Ress, shape, block_size, overlap, num_blocks)
        t1 = time.time()
        print("Time: %.2f Render: %.2f Original_gaussian: %d Final_gaussian: %d"%(t1-t0,t_render*1000, Original_gaussian,FInal_gaussian))

    psnr = evaluate.calc_psnr(raw,Res,config.data.Max)
    ssim = evaluate.calc_ssim(raw,Res,config.data.Max)

    h5list = os.listdir(output_dir)
    compression_size = 0
    for h5 in h5list:
        if h5.endswith('.h5'):
            compression_size += os.path.getsize(output_dir+'/'+h5)
    

    tf.imwrite(output_dir+'/org.tif',raw.astype(np.uint16))
    ratio = os.path.getsize(output_dir+'/org.tif')/compression_size
    
    tf.imwrite(output_dir+'/3dgrc_result.tif',Res.astype(np.uint16))
    print("PSNR: %.4f, SSIM: %.4f, Ratio: %.4f"%(psnr,ssim,ratio))
    np.savetxt(output_dir+'/3dgrc_ratio_%.4f_psnr_%.4f_ssim_%.4f.txt'%(ratio,psnr,ssim),np.array([ratio,psnr,ssim]))
    shutil.copytree('Temp',output_dir+'/Temp')
    
    print("Done")
