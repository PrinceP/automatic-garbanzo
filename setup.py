import os
from distutils.core import setup, Extension
import numpy as np

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"
os.environ["CUDA_HOME"] = "/usr/local/cuda-11.4"
#OPENCV_LIBS = "/usr/local/lib"
OPENCV_LIBS = "/opt/opencv/lib"
#OPENCV_INCLUDE = "/usr/local/include/opencv4"
OPENCV_INCLUDE = "/opt/opencv/include/opencv4"

TRT_INCLUDE = "/usr/include/x86_64-linux-gnu/"
TRT_LIB = "/usr/lib/x86_64-linux-gnu/"

if 'CUDA_HOME' in os.environ:
   CUDA_HOME = os.environ['CUDA_HOME']
else:
   print("Could not find CUDA_HOME in environment variables. Defaulting to /usr/local/cuda!")
   CUDA_HOME = "/usr/local/cuda"

if not os.path.isdir(CUDA_HOME):
   print("CUDA_HOME {} not found. Please update the CUDA_HOME variable and rerun".format(CUDA_HOME))
   exit(0)

if not os.path.isdir(os.path.join(CUDA_HOME, "include")):
    print("include directory not found in CUDA_HOME. Please update CUDA_HOME and try again")
    exit(0)

print(os.path.abspath(os.getcwd()))

setup(name = 'myModule', version = '1.0',  \
   ext_modules = [
      Extension('myModule', ['myModule.c'],
      include_dirs=[np.get_include(), os.path.join(CUDA_HOME, "include"), OPENCV_INCLUDE, TRT_INCLUDE],
      libraries=["preprocess", "cudart", "opencv_core", "opencv_imgproc", "opencv_imgcodecs", "nvinfer", "nvonnxparser"],
      #library_dirs = ["/home/uncanny/projects/CUDA/Python-C-API-CUDA-Tutorial_STATIC/preprocess_test/build", os.path.join(CUDA_HOME, "lib64"), OPENCV_LIBS],
      library_dirs = ["./preprocess_test/build", os.path.join(CUDA_HOME, "lib64"), OPENCV_LIBS, TRT_LIB],
      extra_compile_args=['-std=c++11', '-fPIC'],
      #extra_link_args=['-static']
    )],
)
