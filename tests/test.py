import myModule
import os
from ctypes import py_object
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


batch_size = 2
fp16 = 0
onnx_path = '/app/models/reid/objects_512.onnx'
# onnx_path = '/app/models/reid/baseline_R50.onnx'
# onnx_path = '/app/models/reid/sbs_R50_batch1.onnx'
# engine_path = '/app/models/reid/objects_512.trt'
engine_path = '/app/models/objects_512.engine'
input_height =  256
input_width =  256
output_size = 512

mean_values = (0.485, 0.456, 0.406)
scale_values = (0.229, 0.224, 0.225)

# mean_values = (0,0,0)
# scale_values = (1,1,1)
# 

#if os.path.exists(engine_path):
#    os.remove(engine_path)
#Path to engine, batch size, input height, input width, output total classes of model
myModule.load_onnx(onnx_path, engine_path, batch_size, input_height, input_width, fp16)
myModule.load_engine(engine_path, batch_size, input_height, input_width, output_size)

#Batch of images
all_batch_images = []
all_meta_info = []
all_crop_index = []

img = np.array(cv2.imread(os.path.join("/app/samples/reid_data/amiya_data/car1.jpg")))
# img = np.array(cv2.imread(os.path.join("/app/samples/reid_data/amiya_data/car2.jpg")))
details = (124,47,391,260)
# details = (161, 43, 414, 289)
all_batch_images.append(img)
all_meta_info.append(details)
all_crop_index.append(len(all_batch_images)-1)


img = np.array(cv2.imread(os.path.join("/app/samples/reid_data/prince/lb/pr2/a.jpg")))
# img = np.array(cv2.imread(os.path.join("/app/samples/reid_data/prince/lb/pr2/b.jpg")))
#details = (0,0,391,260)
details = (0,0,414,289)
all_batch_images.append(img)
all_meta_info.append(details)
all_crop_index.append(len(all_batch_images)-1)

#img = np.array(cv2.imread(os.path.join("/app/samples/reid_data/crops/3454709428_20211011140532dtest-90641884854000-1.jpg")))
#img = np.array(cv2.imread(os.path.join("/app/a.jpg")))
# img = np.zeros((260,391,3))
#details = (0,0,391,260)
#img = np.ones((60, 174, 3)) * 127
# details = (0,0,60,174)
#all_batch_images.append(img)
#all_meta_info.append(details)
#all_crop_index.append(len(all_batch_images)-1)

batch_images = all_batch_images[0:batch_size]
crop_info = all_meta_info[0:batch_size]

myModule.add_capacity(batch_size)
print("Start")

#for i,single_image in enumerate(batch_images):
#    single_crop = crop_info[i]
#    dims = single_image.shape
#    print(dims)
#    print(single_crop)
#    single_image = single_image.ravel()
#    myModule.add_images(dims, single_image)
#    myModule.add_crops(single_crop)


for i,single_image in enumerate(batch_images):
    dims=single_image.shape
    single_image=single_image.ravel()
    myModule.add_images(dims,single_image)

for i,single_crop in enumerate(crop_info):
    myModule.add_crops(single_crop)

for i,crop_indx in enumerate(all_crop_index):
    myModule.add_crop_indexes(crop_indx);

trt_outputs = myModule.perform_inference(mean_values, scale_values)

myModule.clear_images()
myModule.clear_crops()
myModule.clear_cache()


for i in range(0,50):
    print(trt_outputs[0][i])

# np.save("3454697505_20211011134539100a9d69s4-19776410682000-1.jpg", trt_outputs)
# np.save("a_lws_zeros_onnx_letterboxingEnabled.npy", trt_outputs)
# np.save("a_lws_orig_onnx_letterboxingEnabled.npy", trt_outputs)
# np.save("b.npy", trt_outputs)


print("End")

# stream.synchronize()




