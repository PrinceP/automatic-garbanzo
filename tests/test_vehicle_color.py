import myModule
import os
from ctypes import py_object
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


batch_size = 5
fp16 = 0
onnx_path = '/app/models/vehicle_color/vehicle_color_march.onnx'
engine_path = '/app/models/vehicle_color/color.trt'+str(batch_size)
name_path = '/app/models/vehicle_color/color.names'
input_height = 224
input_width = 224
output_size = 10

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


# if os.path.exists(engine_path):
#     os.remove(engine_path)

myModule.load_onnx(onnx_path, engine_path, batch_size, input_height, input_width, fp16)
myModule.load_engine(engine_path, batch_size, input_height, input_width, output_size)

#Batch of images
all_batch_images = []
all_meta_info = []

img = np.array(cv2.imread(os.path.join("/app/samples/classifier_data/testpersonchar1.jpg")))
details = (6, 78, 114-6, 231-78)

cv2.imwrite("cropped1.jpg", img[details[1]: details[3]+details[1]  ,  details[0]: details[0]+details[2], :  ])

myModule.perform_preprocess(img, input_width, input_height, details[0], details[1], details[2], details[3] )

all_batch_images.append(img)
all_meta_info.append(details)

img = np.array(cv2.imread(os.path.join("/app/samples/classifier_data/search_1pdup.jpg")))
details = (89, 82, 190-89, 197-82)

cv2.imwrite("cropped2.jpg", img[details[1]: details[3]+details[1]  ,  details[0]: details[0]+details[2], :  ])
all_batch_images.append(img)
all_meta_info.append(details)

img = np.array(cv2.imread(os.path.join("/app/samples/classifier_data/testpersonchar1.jpg")))
details = (6, 78, 114-6, 231-78)
all_batch_images.append(img)
all_meta_info.append(details)

img = np.array(cv2.imread(os.path.join("/app/samples/classifier_data/search_1pdup.jpg")))
details = (89, 82, 190-89, 197-82)
all_batch_images.append(img)
all_meta_info.append(details)


img = np.array(cv2.imread(os.path.join("/app/samples/bcc_data/3461685072_20211019155136.065.jpeg")))
details = (404, 36, 55, 65)
all_batch_images.append(img)
all_meta_info.append(details)


img = np.array(cv2.imread(os.path.join("/app/samples/bcc_data/3461685072_20211019155136.065.jpeg")))
details = (414, 97, 46, 56)
all_batch_images.append(img)
all_meta_info.append(details)



img = np.array(cv2.imread(os.path.join("/app/samples/color_data/3454690427_20211011133354.587.jpeg")))
details = (339, 4, 240, 211)
all_batch_images.append(img)
all_meta_info.append(details)

# img = np.array(cv2.imread(os.path.join("/app/samples/color_data/3454690427_20211011133354.587.jpeg")))
# details = (459, 127, 71, 81)
# all_batch_images.append(img)
# all_meta_info.append(details)


img = np.array(cv2.imread(os.path.join("/app/samples/color_data/3454744081_20211011150322.274.jpeg")))
details = (281, 30, 248, 200)
all_batch_images.append(img)
all_meta_info.append(details)





myModule.add_capacity(batch_size)

total_images = len(all_batch_images)
totalPasses = int(total_images / batch_size)
leftOver = total_images % batch_size 


for i in range(0, totalPasses):
    start_index = i  
    end_index = start_index + batch_size
    batch_images = all_batch_images[start_index:end_index]
    crop_info = all_meta_info[start_index:end_index]
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
    trt_outputs = np.array(trt_outputs).reshape(batch_size, output_size)
    indexs = np.argmax(trt_outputs, axis = 1)
    for i in indexs:
        print("Output: ",output_names[i])
    

    myModule.clear_images()
    myModule.clear_crops()

if leftOver > 0:
    start_index = totalPasses * batch_size  
    batch_images = all_batch_images[start_index:]
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
    for i in indexs:
        print("Output: ",output_names[i])
    

    myModule.clear_images()
    myModule.clear_crops()






myModule.clear_cache()

print("End")

# stream.synchronize()




