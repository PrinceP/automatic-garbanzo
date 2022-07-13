# Python C++ CUDA Inference 



### Steps to run the workspace
1. Clone the repo 
```
git clone https://github.com/PrinceP/automatic-garbanzo
```

2. Move to automatic-garbanzo
```
cd automatic-garbanzo
```

3. Run on the workspace docker and execute the test 
```
sudo docker run -v {$PWD}:/app -it cuda_kernel_workspace bash

cd /app

make clean

make 

```


### For generating and testing Engine [trtexec]
```
trtexec --onnx=bcc.onnx --verbose --explicitBatch --minShapes=input0:1x3x224x224 --maxShapes=input0:10x3x224x224 --optShapes=input0:5x3x224x224  --saveEngine=bcc.trt

trtexec --loadEngine=bcc.trt --shapes='input0':2x3x224x224
```

