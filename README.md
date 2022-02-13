# Run Workspace
sudo docker run -v {$PWD}:/app -it cuda_kernel_workspace bash


# For generating and testing Engine [trtexec]
trtexec --onnx=bcc.onnx --verbose --explicitBatch --minShapes=input0:1x3x224x224 --maxShapes=input0:10x3x224x224 --optShapes=input0:5x3x224x224  --saveEngine=bcc.trt
trtexec --loadEngine=bcc.trt --shapes='input0':2x3x224x224
