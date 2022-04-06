import myModule
import os
from ctypes import py_object
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


batch_size = 5
fp16 = 0
onnx_path = '/app/models/vehicle_make/vehicle_make_march.onnx'
engine_path = '/app/models/vehicle_make/make.trt'+str(batch_size)
name_path = '/app/models/vehicle_make/make.names'
input_height = 224
input_width = 224
output_size = 48

mean_values = (0.485, 0.456, 0.406)
scale_values = (0.229, 0.224, 0.225)

# crop_img = crop_img.astype(np.float32) / 255
# crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
# # normalize and convert to Channel Width Height
# mean = np.array(mean_values, dtype=np.float32)
# scale = np.array(scale_values, dtype=np.float32)
# crop_img = (crop_img - mean) / scale
# images = crop_img.transpose((2, 0, 1))

output_names = {}
if os.path.exists(name_path):
    file_ = open(name_path)
    i = 0
    for f in file_.readlines():
        output_names[i] = f.strip()
        i = i + 1


if not os.path.exists(engine_path):
    # os.remove(engine_path)

    myModule.load_onnx(onnx_path, engine_path, batch_size, input_height, input_width, fp16)

myModule.load_engine(engine_path, batch_size, input_height, input_width, output_size)

#Batch of images
all_batch_images = []
all_meta_info = []


# Ford
img1 = np.array(cv2.imread(os.path.join("/app/samples/color_data/3454690427_20211011133354.587.jpeg")))
details1 = (339, 4, 240, 211)
all_batch_images.append(img1)
all_meta_info.append(details1)


# Lexus 
img2 = np.array(cv2.imread(os.path.join("/app/samples/color_data/3454744081_20211011150322.274.jpeg")))
details2 = (281, 30, 248, 200)
all_batch_images.append(img2)
all_meta_info.append(details2)




# Ford
img3 = np.array(cv2.imread(os.path.join("/app/samples/color_data/3454690427_20211011133354.587.jpeg")))
details3 = (339, 4, 240, 211)
all_batch_images.append(img3)
all_meta_info.append(details3)


# Lexus 
img4 = np.array(cv2.imread(os.path.join("/app/samples/color_data/3454744081_20211011150322.274.jpeg")))
details4 = (281, 30, 248, 200)
all_batch_images.append(img4)
all_meta_info.append(details4)



# Ford
img5 = np.array(cv2.imread(os.path.join("/app/samples/color_data/3454690427_20211011133354.587.jpeg")))
details5 = (339, 4, 240, 211)
all_batch_images.append(img5)
all_meta_info.append(details5)


# Lexus 
img6 = np.array(cv2.imread(os.path.join("/app/samples/color_data/3454744081_20211011150322.274.jpeg")))
details6 = (281, 30, 248, 200)
all_batch_images.append(img6)
all_meta_info.append(details6)



# Ford
img7 = np.array(cv2.imread(os.path.join("/app/samples/color_data/3454690427_20211011133354.587.jpeg")))
details7 = (339, 4, 240, 211)
all_batch_images.append(img7)
all_meta_info.append(details7)


# Ford
img8 = np.array(cv2.imread(os.path.join("/app/samples/color_data/3454690427_20211011133354.587.jpeg")))
details8 = (339, 4, 240, 211)
all_batch_images.append(img8)
all_meta_info.append(details8)

# Ford
img9 = np.array(cv2.imread(os.path.join("/app/samples/color_data/3454690427_20211011133354.587.jpeg")))
details9 = (339, 4, 240, 211)
all_batch_images.append(img9)
all_meta_info.append(details9)


# Lexus 
img10 = np.array(cv2.imread(os.path.join("/app/samples/color_data/3454744081_20211011150322.274.jpeg")))
details10 = (281, 30, 248, 200)
all_batch_images.append(img10)
all_meta_info.append(details10)



myModule.add_capacity(batch_size)

total_images = len(all_batch_images)
totalPasses = int(total_images / batch_size)
leftOver = total_images % batch_size 

start_index = 0
for i in range(0, totalPasses):
    print("Pass Number: " + str(i))
    start_index = i * batch_size  
    end_index = start_index + batch_size
    print(str(start_index) + " -- " + str(end_index)) 
    batch_images = all_batch_images[start_index:end_index]
    crop_info = all_meta_info[start_index:end_index]
    print("Start")

    for i,single_image in enumerate(batch_images):
        single_crop = crop_info[i]
        dims = single_image.shape
        print("Python : " + str(single_image[ 180,320,0 ] ))
        print(dims)
        print("Python : " + str(single_crop))
        print(i)
        single_image = single_image.ravel()
        myModule.add_images(dims, single_image)
        myModule.add_crops(single_crop)
        print("--------------")

    trt_outputs = myModule.perform_inference(mean_values, scale_values)
    trt_outputs = np.array(trt_outputs).reshape(batch_size, output_size)
    indexs = np.argmax(trt_outputs, axis = 1)
    print(indexs)
    print(np.take_along_axis(trt_outputs, indexs[:, None], axis=-1))
    for i in indexs:
        print("Output: ",output_names[i])
    

    myModule.clear_images()
    myModule.clear_crops()

if leftOver > 0:
    start_index = totalPasses * batch_size  
    batch_images = all_batch_images[start_index:]
    print("Leftover from " + str(start_index)) 
    crop_info = all_meta_info[start_index:]
    print("Start")
    j = 0

    for i,single_image in enumerate(batch_images):
        j = j + 1
        single_crop = crop_info[i]
        dims = single_image.shape
        print(dims)
        print(single_crop)
        single_image = single_image.ravel()
        myModule.add_images(dims, single_image)
        myModule.add_crops(single_crop)

    trt_outputs = myModule.perform_inference(mean_values, scale_values)
    trt_outputs = np.array(trt_outputs).reshape(j, output_size)
    indexs = np.argmax(trt_outputs, axis = 1)
    print(indexs)
    print(np.take_along_axis(trt_outputs, indexs[:, None], axis=-1))
    for i in indexs:
        print("Output: ",output_names[i])
    

    myModule.clear_images()
    myModule.clear_crops()






myModule.clear_cache()

print("End")

# stream.synchronize()




