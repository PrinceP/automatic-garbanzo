import myModule
import os
from ctypes import py_object
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


batch_size = 1
fp16 = 0
# onnx_path = '/app/models/reid/baseline_R18_512.onnx'
onnx_path = '/app/models/reid/baseline_R50.onnx'
# onnx_path = '/app/models/reid/sbs_R50_batch1.onnx'
engine_path = '/app/models/reid/baseline_50.trt'
input_height =  384
input_width =  128
output_size = 2048

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

# img = np.array(cv2.imread(os.path.join("/app/samples/reid_data/test_prince.jpg")))
# details = (0,0,256,256)
# all_batch_images.append(img)
# all_meta_info.append(details)

# img = np.array(cv2.imread(os.path.join("/app/samples/reid_data/crops/3454709428_20211011140532dtest-90641884854000-1.jpg")))
# details = (0,0,60,174)
# all_batch_images.append(img)
# all_meta_info.append(details)

# img = np.array(cv2.imread(os.path.join("/app/samples/reid_data/prince/lb/pr2/a.jpg")))
# img = np.array(cv2.imread(os.path.join("/app/samples/reid_data/prince/lb/pr2/b.jpg")))
# details = (0,0,391,260)

# img = np.array(cv2.imread(os.path.join("/app/a.jpg")))
img = np.zeros((260,391,3))
details = (0,0,391,260)
# details = (0,0,414,289)
all_batch_images.append(img)
all_meta_info.append(details)
 


# img = np.array(cv2.imread(os.path.join("/app/samples/reid_data/amiya_data/car1.jpg")))
# img = np.array(cv2.imread(os.path.join("/app/samples/reid_data/amiya_data/car2.jpg")))
# details = (124,47,391,260)
# details = (161, 43, 414, 289)
# all_batch_images.append(img)
# all_meta_info.append(details)
# 
# 
# img = np.array(cv2.imread(os.path.join("/app/samples/reid_data/0022_c6s1_002976_01.jpg")))
# details = (0,0,64,128)
# all_batch_images.append(img)
# all_meta_info.append(details)
# 
# img = np.array(cv2.imread(os.path.join("/app/samples/reid_data/0022_c6s1_002976_01.jpg")))
# details = (0,0,64,128)
# all_batch_images.append(img)
# all_meta_info.append(details)
# 
# img = np.array(cv2.imread(os.path.join("/app/samples/reid_data/0022_c6s1_002976_01.jpg")))
# details = (0,0,64,128)
# all_batch_images.append(img)
# all_meta_info.append(details)
# 
# 

batch_images = all_batch_images[0:batch_size]
crop_info = all_meta_info[0:batch_size]
myModule.add_capacity(batch_size)
print("Start")

for i,single_image in enumerate(batch_images):
    single_crop = crop_info[i]
    dims = single_image.shape
    print(dims)
    print(single_crop)
    single_image = single_image.ravel()
    myModule.add_images(dims, single_image)
    myModule.add_crops(single_crop)

trt_outputs = myModule.perform_inference(mean_values, scale_values)
# for i in range(0,50):
#     print(trt_outputs[0][i])

# np.save("3454709428_20211011140532dtest-90641884854000-1.jpg", trt_outputs)
# np.save("car1.npy", trt_outputs)
np.save("car2.npy", trt_outputs)


myModule.clear_images()
myModule.clear_crops()
myModule.clear_cache()

print("End")

# stream.synchronize()




