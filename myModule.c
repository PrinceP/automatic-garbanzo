#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>


#include <stdlib.h>

#include <iostream>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <cstdint>

#include "NvInfer.h"
#include "logging.h"
#include "cuda_utils.h"
#include "NvOnnxParser.h"
#include <fstream>
#include <chrono>
using namespace nvonnxparser;

static Logger gLogger;
using namespace nvinfer1;

#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000
/* #define INPUT_BLOB_NAME "batched_inputs.1" */
#define INPUT_BLOB_NAME "input0"
#define OUTPUT_BLOB_NAME "output0"
/* #define OUTPUT_BLOB_NAME "identity_out" */



extern void preprocess_kernel_img(uint8_t* src, int src_width, int src_height,
                           float* dst, int dst_width, int dst_height,
                           float mean_values[3], float scale_values[3],
                           cv::Rect crop,
                           int letterbox,
                           cudaStream_t stream);


static IRuntime* runtime;
static ICudaEngine* engine;
static IExecutionContext* context;
static cudaStream_t stream;

static std::vector<cv::Mat> list_of_images;
static std::vector<cv::Rect> list_of_crops;
static std::vector<int> list_of_crop_indx;
static float* buffers[2];
/* static float* output; */
static uint8_t* img_host = nullptr;
static uint8_t* img_device = nullptr;
static uint8_t** hostbuffers;
static int inputIndex;
static int outputIndex;
static int inputHeight;
static int inputWidth;
static int batchSize; 
static int outputSize;



static PyObject* perform_preprocess(PyObject* self, PyObject* args){
    PyArrayObject *input;
    int input_Height, input_Width;
    int output_Height, output_Width;
    
    int x, y, w, h;



    printf("Hello PreProcess 1\n");
    if (!PyArg_ParseTuple(args, "Oiiiiii",
          &input,
          &output_Width,  &output_Height,
          &x, &y, &w, &h
          ))
        return NULL;
    if (input->nd != 3) {
      fprintf(stderr, "invalid image\n");
      Py_RETURN_NONE;
    }
    if (input->dimensions[2] != 3) {
      fprintf(stderr, "invalid image\n");
      Py_RETURN_NONE;
    }
    input_Height = input->dimensions[0];
    input_Width = input->dimensions[1];
    printf("Got dimensions %d x %d to %d x %d \n", input_Width, input_Height, output_Width, output_Height);
    uint8_t* img_host = nullptr;
    uint8_t* img_device = nullptr;
    float* outArr_host = nullptr;
    float* outArr_device = nullptr;
    printf("Hello PreProcess 2\n");
    CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    CUDA_CHECK(cudaMalloc((float**)&outArr_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3 ));
    CUDA_CHECK(cudaMallocHost((float**)&outArr_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    printf("Hello PreProcess 3\n");
    size_t  size_image = input_Width * input_Height * 3;
    size_t  size_image_dst = output_Width * output_Height * 3 * sizeof(float); //Size of float
    //copy data to pinned memory
    printf("Hello PreProcess 3.1\n");
    memcpy(img_host,input->data,size_image);
    //copy data to device memory
    printf("Hello PreProcess 3.2\n");
    CUDA_CHECK(cudaMemcpy(img_device,img_host,size_image,cudaMemcpyHostToDevice));
    printf("Hello PreProcess 4\n");

    float mean_values[3] = { 0.485, 0.456, 0.406 };
    float scale_values[3] ={ 0.229, 0.224, 0.225 };
    cv::Rect my_crop(x, y, w, h);

    preprocess_kernel_img(img_device, input_Width, input_Height, 
         outArr_device,  output_Width, output_Height, mean_values, scale_values, my_crop, 0, stream); 
    printf("Hello PreProcess 5\n");
    CUDA_CHECK(cudaMemcpy(outArr_host, outArr_device, size_image_dst, cudaMemcpyDeviceToHost));
    cv::Mat float_R = cv::Mat(output_Height, output_Width,CV_32FC1, outArr_host);
    cv::Mat float_G = cv::Mat(output_Height, output_Width,CV_32FC1, outArr_host + output_Width * output_Height) ;
    cv::Mat float_B = cv::Mat(output_Height, output_Width,CV_32FC1, outArr_host + 2 * output_Width * output_Height);

    FILE* fp = fopen("finaldata.bin", "wb");
    fwrite(outArr_host, 1, sizeof(float) * output_Width * output_Height * 3, fp); 
    fclose(fp);


    float_R = float_R * scale_values[0] + mean_values[0];
    float_G = float_G * scale_values[1] + mean_values[1];
    float_B = float_B * scale_values[2] + mean_values[2];


    cv::Mat char_r, char_g, char_b;
    float_R.convertTo(char_r, CV_8UC1, 255.0);
    float_G.convertTo(char_g, CV_8UC1, 255.0);
    float_B.convertTo(char_b, CV_8UC1, 255.0);
    /* cv::imwrite("PreProcessed/R_channel.jpg", char_r); */
    /* cv::imwrite("PreProcessed/G_channel.jpg", char_g); */
    /* cv::imwrite("PreProcessed/B_channel.jpg", char_b); */
    cv::Mat channels[3] = {char_b, char_g, char_r};
    cv::Mat new_output;
    cv::merge(channels, 3, new_output);
    cv::imwrite("final.jpg", new_output);


  Py_RETURN_NONE;
}

static PyObject* add_capacity(PyObject* self, PyObject* args){
  int n;
  if (!PyArg_ParseTuple(args, "i", &n))
    return NULL;

  list_of_images.reserve(n);
  list_of_crops.reserve(n);
  Py_RETURN_NONE;
}


static PyObject* add_crop_indexes(PyObject* self,PyObject* args){
	int x;
	if(!PyArg_ParseTuple(args,"i",&x))
		return NULL;
	list_of_crop_indx.push_back(x);
	Py_RETURN_NONE;
}


static PyObject* add_crops(PyObject* self, PyObject* args){

  PyObject *size;

  if (!PyArg_ParseTuple(args, "O!", &PyTuple_Type, &size)) {
    return NULL;
  }
  int x = PyLong_AsLong(PyTuple_GetItem(size ,0));
  int y = PyLong_AsLong(PyTuple_GetItem(size ,1));
  int w = PyLong_AsLong(PyTuple_GetItem(size ,2));
  int h = PyLong_AsLong(PyTuple_GetItem(size ,3));


  cv::Rect my_crop(x, y, w, h);
  std::cout << "C++ : " << x << " " << y << " "<< w << " " << h << std::endl;
  list_of_crops.push_back(my_crop);

  Py_RETURN_NONE;
}


static PyObject* add_images(PyObject* self, PyObject* args){

  PyObject *size;
  PyArrayObject *image;

  if (!PyArg_ParseTuple(args, "O!O!", &PyTuple_Type, &size, &PyArray_Type, &image)) {
    return NULL;
  }
  int rows = PyLong_AsLong(PyTuple_GetItem(size ,0));
  int cols = PyLong_AsLong(PyTuple_GetItem(size ,1));
  int nchannels = PyLong_AsLong(PyTuple_GetItem(size ,2));

  /* char my_arr[rows * nchannels * cols]; */
  
  std::cout << rows << " "<< cols << " " << nchannels <<  std::endl;

  std::cout << "Size = " << list_of_images.size() << std::endl;

  unsigned char* p = hostbuffers[list_of_images.size()];
    
  std::cout << static_cast<void*>(p) << std::endl;
  std::cout << static_cast<void*>(hostbuffers) << std::endl;

  for(size_t length = 0; length<(rows * nchannels * cols); length++){

    /* my_arr[length] = (*(char *)PyArray_GETPTR1(image, length)); */
    p[length] =  (*(char *)PyArray_GETPTR1(image, length));
  
  }


  cv::Mat my_img = cv::Mat(cv::Size(cols, rows), CV_8UC3, p);

  /* unsigned char * p = my_img.ptr(180, 320);   */
  /* std::cout << "C++ : " << (float)p[0] << std::endl; */

  


  std::cout << "Size = " << list_of_images.size() << std::endl;


  list_of_images.push_back(my_img);

  Py_RETURN_NONE;
}


static PyObject* load_onnx(PyObject* self, PyObject* args){

  char *filename, *enginename;
  int BatchSize, InputH, InputW, isFP16;
  /* Parse arguments */
  if(!PyArg_ParseTuple(args, "ssiiii", &filename, &enginename, &BatchSize, &InputH, &InputW, &isFP16)) {
    return NULL;
  }
  std::ifstream file(filename, std::ios::binary);
  if (!file.good()) {
    std::cerr << "read " << filename << " error!" << std::endl;
    Py_RETURN_NONE;
  }
  IBuilder* builder = createInferBuilder(gLogger);
  builder->setMaxBatchSize(BatchSize + 20);
  uint32_t flag = 1U <<static_cast<uint32_t>
    (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 

  INetworkDefinition* network = builder->createNetworkV2(flag);
  IParser*  parser = createParser(*network, gLogger);
  parser->parseFromFile(filename, 3);
  for (int32_t i = 0; i < parser->getNbErrors(); ++i)
  {
    std::cout << parser->getError(i)->desc() << std::endl;
  }

  IOptimizationProfile* profile = builder->createOptimizationProfile();
  profile->setDimensions("input0", OptProfileSelector::kMIN, Dims4(BatchSize,3,InputH,InputW));
  profile->setDimensions("input0", OptProfileSelector::kOPT, Dims4(BatchSize,3,InputH,InputW));
  profile->setDimensions("input0", OptProfileSelector::kMAX, Dims4(BatchSize,3,InputH,InputW));
  
  
  /* ITensor* output_tensors = network->getOutput(0); */
  /* network->unmarkOutput(*output_tensors); */
  /* ITensor* identity_out_tensor = network->addIdentity( *output_tensors )->getOutput(0); */
  /* const char* name = "identity_out"; */
  /* identity_out_tensor->setName(name); */
  /* network->markOutput(*identity_out_tensor); */



  IBuilderConfig* config = builder->createBuilderConfig();
  config->setMaxWorkspaceSize(1U << 32);
  config->addOptimizationProfile(profile);
  std::cout<<"isFP16: "<<isFP16<<"\n";
  if(isFP16){
    config->setFlag(BuilderFlag::kFP16);
  }
  IHostMemory*  serializedModel = builder->buildSerializedNetwork(*network, *config);
  std::ofstream p(enginename, std::ios::binary);
  if (!p) {
    std::cerr << "could not open plan output file" << std::endl;
    Py_RETURN_NONE;
  }
  p.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());

  delete parser;
  delete network;
  delete config;
  delete builder;
  delete serializedModel;

  Py_RETURN_NONE;
}

static PyObject* load_engine(PyObject* self, PyObject* args){

  char *filename;
  int BatchSize, InputH, InputW, OutputSize;
  /* Parse arguments */
  if(!PyArg_ParseTuple(args, "siiii", &filename, &BatchSize, &InputH, &InputW, &OutputSize)) {
    return NULL;
  }
  inputWidth = InputW;
  inputHeight = InputH;
  batchSize = BatchSize;
  outputSize = OutputSize;

  std::ifstream file(filename, std::ios::binary);
  if (!file.good()) {
    std::cerr << "read " << filename << " error!" << std::endl;
    Py_RETURN_NONE;
  }
  char *trtModelStream = nullptr;
  size_t size = 0;
  file.seekg(0, file.end);
  size = file.tellg();
  file.seekg(0, file.beg);
  trtModelStream = new char[size];
  assert(trtModelStream);
  file.read(trtModelStream, size);
  file.close();

  runtime = createInferRuntime(gLogger);
  assert(runtime != nullptr);
  engine = runtime->deserializeCudaEngine(trtModelStream, size);
  assert(engine != nullptr);
  context = engine->createExecutionContext();
  assert(context != nullptr);
  delete[] trtModelStream;
  
  inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
  outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
  assert(inputIndex == 0);
  assert(outputIndex == 1);
  
  CUDA_CHECK(cudaStreamCreate(&stream));
  
  context->setBindingDimensions(inputIndex, Dims4(BatchSize, 3, InputH, InputW));
  /* context->setOptimizationProfileAsync(0, stream); */
  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BatchSize * 3 * InputH * InputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], BatchSize * OutputSize * sizeof(float)));
  /* output = new float[BatchSize * OutputSize]; */

  hostbuffers = (uint8_t**)malloc(sizeof(uint8_t*) * BatchSize);
  for(int i= 0 ; i < BatchSize; i++)
    CUDA_CHECK(cudaMallocHost(  &hostbuffers[i]   , MAX_IMAGE_INPUT_SIZE_THRESH * 3 ));

  // prepare input data cache in pinned memory 
  CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
  // prepare input data cache in device memory
  CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
  Py_RETURN_NONE; 
}

static PyObject* perform_inference(PyObject* self, PyObject* args){
  
//int length = list_of_images.size();
  int no_of_images=list_of_images.size();
  printf("Total Images taken: %d \n",no_of_images);
  int length = list_of_crops.size();
  printf("Total no of crops take:%d\n",length);
  PyObject *mean_values_input;
  PyObject *scale_values_input;

  if (!PyArg_ParseTuple(args, "O!O!", &PyTuple_Type, &mean_values_input, &PyTuple_Type, &scale_values_input)) {
    return NULL;
  }

  float x1 = PyFloat_AsDouble(PyTuple_GetItem(mean_values_input ,0));
  float y1 = PyFloat_AsDouble(PyTuple_GetItem(mean_values_input ,1));
  float z1 = PyFloat_AsDouble(PyTuple_GetItem(mean_values_input ,2));


  float x2 = PyFloat_AsDouble(PyTuple_GetItem(scale_values_input ,0));
  float y2 = PyFloat_AsDouble(PyTuple_GetItem(scale_values_input ,1));
  float z2 = PyFloat_AsDouble(PyTuple_GetItem(scale_values_input ,2));


  float mean_values[3] = {x1, y1, z1};
  float scale_values[3] ={x2, y2, z2};
   
  float* buffer_idx = (float*)buffers[inputIndex];
  float* output = new float[length * outputSize];

  for(int b=0; b < length; b++){
    
    std::cout << b << std::endl;

    cv::Mat img = list_of_images[list_of_crop_indx[b]];
    cv::Rect crop_of_img  = list_of_crops[b];
  
    /* unsigned char * p = img.ptr(180, 320); */
    /* std::cout << "Inference : " << (float)p[0] << std::endl; */
    /* std::cout << "Inference : " << crop_of_img.x << " " << crop_of_img.y << " " << crop_of_img.width << " " << crop_of_img.height << std::endl; */
  
    size_t  size_image = img.cols * img.rows * 3;
    size_t  size_image_dst = inputHeight * inputWidth * 3;

    std::cout << size_image << std::endl;
    std::cout << size_image_dst << std::endl;
    //copy data to pinned memory
    
     float my_data[3 * inputWidth * inputHeight]; 

     FILE* fp = fopen("/app/samples/bcc_data/prince_crops/5_395_119_455_187.bin", "rb"); 
     fread(my_data, 1, sizeof(float) *  inputWidth  * inputHeight * 3, fp); 
     for(int i=0; i< 50 ; i++){
	     std::cout<<my_data[i]<<"  ";
     }
     fclose(fp); 

    /* size_image = inputWidth * inputHeight * 3 * sizeof(float); */
    /* memcpy(img_host ,img.data,size_image); */
    /* memcpy(img_host ,my_data,size_image); */

    //copy data to device memory
    /* CUDA_CHECK(cudaMemcpyAsync(buffer_idx,img.data,size_image,cudaMemcpyHostToDevice,stream)); */

    CUDA_CHECK(cudaMemcpyAsync(img_device,img.data,size_image,cudaMemcpyHostToDevice,stream));
    preprocess_kernel_img(img_device, img.cols, img.rows, 
        buffer_idx, inputWidth, inputHeight, 
        mean_values, scale_values,
        crop_of_img,
        1,          
        stream);       

    buffer_idx += size_image_dst;
  }
  
  auto start = std::chrono::system_clock::now();
  context->enqueue(length, (void**)buffers, stream, nullptr);
  /* context->enqueueV2((void**)buffers, stream, nullptr); */
  auto end = std::chrono::system_clock::now();
  std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

  try{
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[outputIndex], length * outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
  }catch(std::runtime_error error){
    std::cout << "Cuda Exception caught" << error.what() <<std::endl;    
  }
  
  /* std::cout << length * outputSize * sizeof(float) << std::endl; */
  cudaStreamSynchronize(stream);


    //Dump input 

    
  
   
    size_t size_image_dst = inputWidth * inputHeight * 3;
    float* our_input = new float[batchSize * size_image_dst];  

    CUDA_CHECK(cudaMemcpy(our_input , buffers[inputIndex], size_image_dst * batchSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    static int filcount = 0;
    float *outArr_host;
    for(int i =0; i < length; i++){
      outArr_host = our_input + i * size_image_dst;
      int output_Width = inputWidth;
      int output_Height = inputHeight;
      cv::Mat float_R = cv::Mat(output_Height, output_Width,CV_32FC1, outArr_host);
      cv::Mat float_G = cv::Mat(output_Height, output_Width,CV_32FC1, outArr_host + output_Width * output_Height) ;
      cv::Mat float_B = cv::Mat(output_Height, output_Width,CV_32FC1, outArr_host + 2 * output_Width * output_Height);

      FILE* fp = fopen("testing/a.bin", "wb");
      fwrite(outArr_host, 1, sizeof(float) * output_Width * output_Height * 3, fp); 
      fclose(fp);
      float_R = float_R * scale_values[0] + mean_values[0];
      float_G = float_G * scale_values[1] + mean_values[1];
      float_B = float_B * scale_values[2] + mean_values[2];
      cv::Mat char_r, char_g, char_b;
      float_R.convertTo(char_r, CV_8UC1, 255.0);
      float_G.convertTo(char_g, CV_8UC1, 255.0);
      float_B.convertTo(char_b, CV_8UC1, 255.0);
      cv::Mat channels[3] = {char_b, char_g, char_r};
      cv::Mat new_output;
      cv::merge(channels, 3, new_output);
      char filename[50];
      sprintf(filename, "testing/a.jpg", filcount);
      cv::imwrite(filename, new_output);
      filcount++;
    }
 


  double outData[length][outputSize];
  std::cout << "Output size = " << outputSize << std::endl;

  for (int b = 0; b < length; b++) {
    int offset = b * outputSize;
    std::cout << "Offset: "<< offset << std::endl;
    for (unsigned int i = 0; i < outputSize; i++)
    {
      std::cout << output[offset + i] << ", ";
      outData[b][i] = output[offset + i];
    }
    std::cout << std::endl;
  }

  PyObject *result = PyTuple_New(length);
  for (Py_ssize_t i = 0; i < length; i++) {
    Py_ssize_t len = outputSize;
    PyObject *item = PyTuple_New(len);

    for (Py_ssize_t j = 0; j < len; j++)
      PyTuple_SET_ITEM(item, j, PyFloat_FromDouble(outData[i][j]));
    PyTuple_SET_ITEM(result, i, item);
  }

  /* delete[] output; */

  return result;
}

static PyObject* clear_images(PyObject* self, PyObject* args){
  list_of_images.clear();
  list_of_images.shrink_to_fit();
  Py_RETURN_NONE; 
}


static PyObject* clear_crops(PyObject* self, PyObject* args){
  list_of_crops.clear();
  list_of_crops.shrink_to_fit();
  Py_RETURN_NONE; 
}

static PyObject* clear_cache(PyObject* self, PyObject* args){
  //Release stream
  cudaStreamDestroy(stream);
  //Destroy Bufferes
  CUDA_CHECK(cudaFree(img_device));
  CUDA_CHECK(cudaFreeHost(img_host));
  CUDA_CHECK(cudaFree(buffers[inputIndex]));
  CUDA_CHECK(cudaFree(buffers[outputIndex]));
  // Destroy the engine
  context->destroy();
  engine->destroy();
  runtime->destroy();
  Py_RETURN_NONE; 
}


static PyMethodDef myMethods[] = {
    {"perform_preprocess", perform_preprocess, METH_VARARGS, "do preprocess with CUDA"},
    {"perform_inference", perform_inference, METH_VARARGS, "do inference with CUDA"},
    {"add_capacity", add_capacity, METH_VARARGS, "initialize capacity"},
    {"load_engine", load_engine, METH_VARARGS, "initialize trt engine"},
    {"load_onnx", load_onnx, METH_VARARGS, "initialize trt engine"},
    {"add_images", add_images, METH_VARARGS, "add images"},
    {"add_crops", add_crops, METH_VARARGS, "add images"},
    {"clear_images", clear_images, METH_VARARGS, "clear images"},
    {"clear_crops", clear_crops, METH_VARARGS, "clear images"},
    {"clear_cache", clear_cache, METH_VARARGS, "clear images"},
    {"add_crop_indexes",add_crop_indexes,METH_VARARGS,"add crop indexes"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef myModule = {
    PyModuleDef_HEAD_INIT, "myModule",
    "myModule", -1, myMethods
};

PyMODINIT_FUNC PyInit_myModule(void) {
    import_array(); 
    return PyModule_Create(&myModule);
}

