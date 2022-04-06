import myModule
import os
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


batch_size = 10
fp16 = 0
onnx_path = '/app/models/bcc/bcc.onnx'
engine_path = '/app/models/bcc/bcc.trt'+str(batch_size)
input_height = 224
input_width = 224
output_size = 10

mean_values = (0.485, 0.456, 0.406)
scale_values = (0.229, 0.224, 0.225)

# if os.path.exists(engine_path):
#     os.remove(engine_path)
#Path to engine, batch size, input height, input width, output total classes of model
# myModule.load_onnx(onnx_path, engine_path, batch_size, input_height, input_width, fp16)
myModule.load_engine(engine_path, batch_size, input_height, input_width, output_size)

#Batch of images
all_batch_images = []
all_meta_info = []

img = np.array(cv2.imread(os.path.join("/app/samples/classifier_data/testpersonchar1.jpg")))
details = (6, 78, 114-6, 231-78)
all_batch_images.append(img)
all_meta_info.append(details)

img = np.array(cv2.imread(os.path.join("/app/samples/classifier_data/search_1pdup.jpg")))
details = (89, 82, 190-89, 197-82)
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

img = np.array(cv2.imread(os.path.join("/app/samples/classifier_data/testpersonchar1.jpg")))
details = (6, 78, 114-6, 231-78)
all_batch_images.append(img)
all_meta_info.append(details)

img = np.array(cv2.imread(os.path.join("/app/samples/classifier_data/search_1pdup.jpg")))
details = (89, 82, 190-89, 197-82)
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


img = np.array(cv2.imread(os.path.join("/app/samples/classifier_data/testpersonchar1.jpg")))
details = (6, 78, 114-6, 231-78)
all_batch_images.append(img)
all_meta_info.append(details)

img = np.array(cv2.imread(os.path.join("/app/samples/classifier_data/search_1pdup.jpg")))
details = (89, 82, 190-89, 197-82)
all_batch_images.append(img)
all_meta_info.append(details)






batch_images = all_batch_images[0:batch_size]
crop_info = all_meta_info[0:batch_size]
myModule.add_capacity(batch_size)
print("Start")

while True:

    for i,single_image in enumerate(batch_images):
        single_crop = crop_info[i]
        dims = single_image.shape
        # print(dims)
        # print(single_crop)
        single_image = single_image.ravel()
        myModule.add_images(dims, single_image)
        myModule.add_crops(single_crop)


    trt_outputs = myModule.perform_inference(mean_values, scale_values)
    myModule.clear_images()
    myModule.clear_crops()


# stream.synchronize()



