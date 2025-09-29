from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == "__main__":
    setup(
        name='GRC_Rendering',
        ext_modules=[
            CUDAExtension('CUDA_kernal_3dgrc', [
                'GRC_Rendering_Interface.cpp', 
                'GRC_Rendering_cuda.cu',             # 需要包含.cpp和.cu文件
            ]),
        ],
        cmdclass={
            'build_ext': BuildExtension
        })
