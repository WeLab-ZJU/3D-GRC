# 3D-AGR

Welcome to 3D-AGR! This repository is aimed at helping researchers or developers interested in 3D-AGR to quickly understand and reproduce our latest research.

## Scheme
<img src="docs/Fig 1.png" width="80%"/>


# Quickstart

## System requirements
- Ubuntu Linux (22.04.1 LTS)
- Anaconda3
- PyTorch

## Installation

### 1. Download project
```
git clone git@github.com:/WeLab-ZJU/3D-AGR.git
```
### 2. Prepare the enviroments
```
cd AGR
conda create -n agr python=3.11
conda activate agr
pip3 install -r requirements.txt
```

### Compile the Cuda kernal
```
cd Cuda_kernal
python compile.py install
cp build/lib.linux-x86_64-cpython-311/CUDA_kernal_3dagr.cpython-311-x86_64-linux-gnu.so AGR/
```
If you are running this project on Windows, you can copy the complied file from the build folder to the main folder
## Usage
```
cd AGR
python AGR.py -c config/default.yaml -g 0
```
You can also write you own .yaml file to compress other data. 
## Contact
If you need any help about 3D-GRC,please feel free to contact us.
haibo.xu@zju.edu.cn
