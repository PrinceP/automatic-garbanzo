# Run Workspace
sudo docker run -v {$PWD}:/app -it cuda_kernel_workspace bash


# For generating Engine
trtexec --onnx=bcc.onnx --verbose --explicitBatch --minShapes=input_0:0:1x224x224x3 --maxShapes=input_0:0:32x224x224x3 --optShapes=input_0:0:8x224x224x3 --shapes=input_0:0:32x224x224x3 --saveEngine=bcc.trt
