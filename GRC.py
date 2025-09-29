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

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3DGRC")
    parser.add_argument('-c', 
                        type=str, 
                        default='config/default.yaml', help='Path to config file')
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
    tf.imwrite(output_dir+'/org.tif',raw.astype(np.uint8))
    img = torch.from_numpy(raw).float().cuda()
    Com = Run(img,config)
    Com.Run()
    Com.save_h5(output_dir+'/3dgrc.h5')
    Res = Com.Render(output_dir+'/3dgrc.h5')

    psnr = evaluate.calc_psnr(raw,Res,config.data.Max)
    ssim = evaluate.calc_ssim(raw,Res,config.data.Max)

    ratio = os.path.getsize(output_dir+'/org.tif')/os.path.getsize(output_dir+'/3dgrc.h5')
    
    tf.imwrite(output_dir+'/3dgrc_result.tif',Res)
    print("PSNR: %.4f, SSIM: %.4f, Ratio: %.4f"%(psnr,ssim,ratio))
    np.savetxt(output_dir+'/3dgrc_ratio_%.4f_psnr_%.4f_ssim_%.4f.txt'%(ratio,psnr,ssim),np.array([ratio,psnr,ssim]))
    shutil.copytree('Temp',output_dir+'/Temp')
    
    print("Done")