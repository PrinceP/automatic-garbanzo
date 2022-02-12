import myModule
import os
from ctypes import py_object
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# stream = cuda.Stream()

# OUT_HEIGHT = 224
# OUT_WIDTH = 224
# OUT_CHANNEL = 3

# print(stream)

#Batch of images
all_batch_images = []
all_meta_info = []
for image_path in os.listdir('/app/samples/'):
    print(image_path)
    img = np.array(cv2.imread(os.path.join("/app/samples/", image_path)))
    details = [float(s) for s in img.shape]
    # img = np.ascontiguousarray(img)

    all_batch_images.append(img)
    all_meta_info.append(np.array(details))
#     input_image = np.ascontiguousarray(img)
#     rows, cols, dims = img.shape
#     size = rows * cols * dims
#     dtype = np.uint8
#     host_mem = cuda.pagelocked_empty(size, dtype)
#     cuda_mem = cuda.mem_alloc(host_mem.nbytes)
#     np.copyto(host_mem, input_image.ravel())
#     
#     size_out = OUT_WIDTH * OUT_HEIGHT * OUT_CHANNEL
#     dtype = np.float32
#     host_mem_out = cuda.pagelocked_empty(size_out, dtype)
#     cuda_mem_out = cuda.mem_alloc(host_mem_out.nbytes)
#     cuda.memcpy_htod_async(cuda_mem, host_mem)

batch_size = 2
batch_images = all_batch_images[0:batch_size]
crop_info = all_meta_info[0:batch_size]

myModule.helloworld()
print("Start")

for single_image in batch_images:
    dims = single_image.shape
    single_image = single_image.ravel()
    myModule.add_images(dims, single_image)

trt_outputs = myModule.perform_inference()
myModule.clear_images()

print("End")

# stream.synchronize()




