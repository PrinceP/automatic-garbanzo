import myModule
import os
from ctypes import py_object
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

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


batch_size = 2
batch_images = all_batch_images[0:batch_size]
crop_info = all_meta_info[0:batch_size]

myModule.helloworld()
myModule.add_capacity(batch_size)
print("Start")

for single_image in batch_images:
    dims = single_image.shape
    single_image = single_image.ravel()
    myModule.add_images(dims, single_image)

trt_outputs = myModule.perform_inference()
myModule.clear_images()

print("End")

# stream.synchronize()




