import myModule
import os
from ctypes import py_object
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


batch_size = 5
fp16 = 0
onnx_path = '/app/models/bcc/bcc_march.onnx'
engine_path = '/app/models/bcc/bcc.trt'+str(batch_size)
name_path = '/app/models/bcc/bcc.names'
input_height = 64
input_width = 64
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

# img = np.array(cv2.imread(os.path.join("/app/samples/classifier_data/testpersonchar1.jpg")))
# details = (6, 78, 114-6, 231-78)
# cv2.imwrite("cropped1.jpg", img[details[1]: details[3]+details[1]  ,  details[0]: details[0]+details[2], :  ])
# myModule.perform_preprocess(img, input_width, input_height, details[0], details[1], details[2], details[3] )
# 
# all_batch_images.append(img)
# all_meta_info.append(details)
# 
# img = np.array(cv2.imread(os.path.join("/app/samples/classifier_data/search_1pdup.jpg")))
# details = (89, 82, 190-89, 197-82)
# cv2.imwrite("cropped2.jpg", img[details[1]: details[3]+details[1]  ,  details[0]: details[0]+details[2], :  ])
# all_batch_images.append(img)
# all_meta_info.append(details)
# 
# img = np.array(cv2.imread(os.path.join("/app/samples/classifier_data/testpersonchar1.jpg")))
# details = (6, 78, 114-6, 231-78)
# all_batch_images.append(img)
# all_meta_info.append(details)
# 
# img = np.array(cv2.imread(os.path.join("/app/samples/classifier_data/search_1pdup.jpg")))
# details = (89, 82, 190-89, 197-82)
# all_batch_images.append(img)
# all_meta_info.append(details)
# 
# 
# img = np.array(cv2.imread(os.path.join("/app/samples/bcc_data/3461685072_20211019155136.065.jpeg")))
# details = (404, 36, 55, 65)
# all_batch_images.append(img)
# all_meta_info.append(details)
# 
# 
# img = np.array(cv2.imread(os.path.join("/app/samples/bcc_data/3461685072_20211019155136.065.jpeg")))
# details = (414, 97, 46, 56)
# all_batch_images.append(img)
# all_meta_info.append(details)
# 
# 
# 
# img = np.array(cv2.imread(os.path.join("/app/samples/bcc_data/3461669924_20211019152631.027.jpeg")))
# details = (479, 84, 49, 53)
# all_batch_images.append(img)
# all_meta_info.append(details)
# 
# img = np.array(cv2.imread(os.path.join("/app/samples/bcc_data/3461669924_20211019152631.027.jpeg")))
# details = (459, 127, 71, 81)
# all_batch_images.append(img)
# all_meta_info.append(details)
# 

# ./brown/upper_body/c4.jpg JPEG 76x91 76x91+0+0 8-bit sRGB 3.56KB 0.000u 0:00.000
# ./brown/lower_body/c3.jpg JPEG 83x161 83x161+0+0 8-bit sRGB 5.54KB 0.000u 0:00.000
# ./red/upper_body/c1.jpg JPEG 72x106 72x106+0+0 8-bit sRGB 3.99KB 0.000u 0:00.000
# ./white/upper_body/c3.jpg JPEG 81x102 81x102+0+0 8-bit sRGB 4.73KB 0.000u 0:00.000
# ./white/lower_body/c1.jpg JPEG 92x108 92x108+0+0 8-bit sRGB 3.96KB 0.000u 0:00.000
# ./blue/upper_body/c2.jpg JPEG 72x125 72x125+0+0 8-bit sRGB 4.57KB 0.000u 0:00.000
# ./blue/upper_body/c7.jpg JPEG 87x156 87x156+0+0 8-bit sRGB 5.7KB 0.000u 0:00.000
# ./blue/lower_body/c7.jpg JPEG 53x81 53x81+0+0 8-bit sRGB 2.48KB 0.000u 0:00.000
# ./black/upper_body/c5.jpg JPEG 92x139 92x139+0+0 8-bit sRGB 5.19KB 0.000u 0:00.000
# ./black/upper_body/c8.jp JPEG 93x131 93x131+0+0 8-bit sRGB 5KB 0.000u 0:00.000
# ./black/upper_body/c6.jpg JPEG 82x134 82x134+0+0 8-bit sRGB 4.27KB 0.000u 0:00.000
# ./black/lower_body/c2.jpg JPEG 58x175 58x175+0+0 8-bit sRGB 4.41KB 0.000u 0:00.000
# ./black/lower_body/c5.jpg JPEG 69x129 69x129+0+0 8-bit sRGB 3.55KB 0.000u 0:00.000
# ./black/lower_body/c72.jpg JPEG 104x68 104x68+0+0 8-bit sRGB 2.91KB 0.000u 0:00.000
# ./black/lower_body/c8.jpg JPEG 104x181 104x181+0+0 8-bit sRGB 7.16KB 0.000u 0:00.000
# ./black/lower_body/c6.jpg JPEG 58x131 58x131+0+0 8-bit sRGB 3.46KB 0.000u 0:00.000
# ./black/lower_body/c4.jpg JPEG 67x180 67x180+0+0 8-bit sRGB 3.74KB 0.000u 0:00.000g
# 


# all_batch_images.append(np.array(cv2.imread(os.path.join("/app/samples/bcc_data/arvind/brown/upper_body/c4.jpg"))))
# all_batch_images.append(np.array(cv2.imread(os.path.join("/app/samples/bcc_data/arvind/brown/lower_body/c3.jpg"))))
# all_batch_images.append(np.array(cv2.imread(os.path.join("/app/samples/bcc_data/arvind/red/upper_body/c1.jpg"))))
# all_batch_images.append(np.array(cv2.imread(os.path.join("/app/samples/bcc_data/arvind/white/upper_body/c3.jpg"))))
# all_batch_images.append(np.array(cv2.imread(os.path.join("/app/samples/bcc_data/arvind/white/lower_body/c1.jpg"))))
# all_batch_images.append(np.array(cv2.imread(os.path.join("/app/samples/bcc_data/arvind/blue/upper_body/c2.jpg"))))
# all_batch_images.append(np.array(cv2.imread(os.path.join("/app/samples/bcc_data/arvind/blue/upper_body/c7.jpg"))))
# all_batch_images.append(np.array(cv2.imread(os.path.join("/app/samples/bcc_data/arvind/blue/lower_body/c7.jpg"))))
# all_batch_images.append(np.array(cv2.imread(os.path.join("/app/samples/bcc_data/arvind/black/upper_body/c5.jpg"))))
# all_batch_images.append(np.array(cv2.imread(os.path.join("/app/samples/bcc_data/arvind/black/upper_body/c8.jpg"))))
# all_batch_images.append(np.array(cv2.imread(os.path.join("/app/samples/bcc_data/arvind/black/upper_body/c6.jpg"))))
# all_batch_images.append(np.array(cv2.imread(os.path.join("/app/samples/bcc_data/arvind/black/lower_body/c2.jpg"))))
# all_batch_images.append(np.array(cv2.imread(os.path.join("/app/samples/bcc_data/arvind/black/lower_body/c5.jpg"))))
# all_batch_images.append(np.array(cv2.imread(os.path.join("/app/samples/bcc_data/arvind/black/lower_body/c72.jpg"))))
# all_batch_images.append(np.array(cv2.imread(os.path.join("/app/samples/bcc_data/arvind/black/lower_body/c8.jpg"))))
# all_batch_images.append(np.array(cv2.imread(os.path.join("/app/samples/bcc_data/arvind/black/lower_body/c6.jpg"))))
# all_batch_images.append(np.array(cv2.imread(os.path.join("/app/samples/bcc_data/arvind/black/lower_body/c4.jpg"))))
# all_batch_images.append(np.array(cv2.imread(os.path.join("/app/samples/bcc_data/arvind/custom.jpg"))))
# 
# 
# all_batch_images.append(np.array(cv2.imread(os.path.join("/app/samples/bcc_data/prince_crops/4_158_104_202_177.jpg"))))
# 

all_batch_images.append(np.array(cv2.imread(os.path.join("/app/samples/bcc_data/prince_crops/brownwhite_p3.jpg"))))



# all_meta_info.append((0,0,76,91)) 
# all_meta_info.append((0,0,83,161)) 
# all_meta_info.append((0,0,72,106))
# all_meta_info.append((0,0,81,102)) 
# all_meta_info.append((0,0,92,108)) 
# all_meta_info.append((0,0,72,125))
# all_meta_info.append((0,0,87,156))
# all_meta_info.append((0,0,53,81))
# all_meta_info.append((0,0,92,139)) 
# all_meta_info.append((0,0,93,131))
# all_meta_info.append((0,0,82,134))
# all_meta_info.append((0,0,58,175))
# all_meta_info.append((0,0,69,129))
# all_meta_info.append((0,0,104,68))
# all_meta_info.append((0,0,104,181))
# all_meta_info.append((0,0,58,131))
# all_meta_info.append((0,0,67,180)) 
# 
# 
# all_meta_info.append((0,0,46,143))
# all_meta_info.append((0,0,64,64))
# 
all_meta_info.append((0,0,44,73))



myModule.add_capacity(batch_size)

total_images = len(all_batch_images)
totalPasses = int(total_images / batch_size)
leftOver = total_images % batch_size 


for i in range(0, totalPasses):
    start_index = i * batch_size  
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




