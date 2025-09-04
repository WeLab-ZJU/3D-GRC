# 3D-GRC

Welcome to 3D-GRC! This repository is aimed at helping researchers or developers interested in 3D-GRC to quickly understand and reproduce our latest research.

## Scheme
<img src="" width="80%"/>
# Quickstart
## System requirements
- Ubuntu Linux (22.04.1 LTS)
- Anaconda3
- PyTorch
## Installation

### 1. Download project
```
git clone git@github.com:/xhb1514288815/3D-GRC.git
```
### 2. Prepare the enviroments
```
cd GRC
conda create -n grc python=3.11
conda activate grc
pip3 install -r requirements.txt
```

### Compile the Cuda kernal
```
cd Cuda_kernal
python compile.py install
cp build/lib.linux-x86_64-cpython-311/CUDA_kernal_3dgrc.cpython-311-x86_64-linux-gnu.so GRC/
```
If you are running this project on Windows, you can copy the complied file from the build folder to the main folder
## Usage
```
cd GRC
python GRC.py -c config/default.yaml -g 0
```
You can also write you own .yaml file to compress other data. 
## Contact
If you need any help about 3D-GRC,please feel free to contact us.
1514288815@qq.com
