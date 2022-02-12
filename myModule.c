#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>

#include <iostream>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <cstdint>

#include "NvInfer.h"
#include "logging.h"
#include "cuda_utils.h"
#include <fstream>
#include <chrono>

static Logger gLogger;
using namespace nvinfer1;

#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000
#define INPUT_BLOB_NAME "input0"
#define OUTPUT_BLOB_NAME "output0"



extern void preprocess_kernel_img(uint8_t* src, int src_width, int src_height,
                           float* dst, int dst_width, int dst_height,
                           float mean_values[3], float scale_values[3],
                           cv::Rect crop);
                           /* cudaStream_t stream); */


static IRuntime* runtime;
static ICudaEngine* engine;
static IExecutionContext* context;
static cudaStream_t stream;

static std::vector<cv::Mat> list_of_images;
static std::vector<cv::Rect> list_of_crops;
static float* buffers[2];
static float* output;
static uint8_t* img_host = nullptr;
static uint8_t* img_device = nullptr;
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
    printf("Hello PreProcess 1\n");
    if (!PyArg_ParseTuple(args, "Oii",
          &input,
          &output_Width,  &output_Height))
        return NULL;
    if (input->nd != 3) {
      fprintf(stderr, "invalid image\n");
      return Py_None;
    }
    if (input->dimensions[2] != 3) {
      fprintf(stderr, "invalid image\n");
      return Py_None;
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
    /* preprocess_kernel_img(img_device, input_Width, input_Height, */
    /*     outArr_device,  output_Width, output_Height); */
    printf("Hello PreProcess 5\n");
    CUDA_CHECK(cudaMemcpy(outArr_host, outArr_device, size_image_dst, cudaMemcpyDeviceToHost));
    cv::Mat float_R = cv::Mat(output_Height, output_Width,CV_32FC1, outArr_host);
    cv::Mat float_G = cv::Mat(output_Height, output_Width,CV_32FC1, outArr_host + output_Width * output_Height) ;
    cv::Mat float_B = cv::Mat(output_Height, output_Width,CV_32FC1, outArr_host + 2 * output_Width * output_Height);
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
    npy_intp dims[] = { output_Height, output_Width, 3 };
    PyObject *image = PyArray_SimpleNew(3, dims, NPY_FLOAT);
    if (!image) {
      fprintf(stderr, "failed to allocate array\n");
      return Py_None;
    }
    PyArrayObject *dstarray = (PyArrayObject *)image;
    for (int y = 0; y < output_Height; y++) {
      uint8_t *src = (uint8_t *)outArr_host + y * output_Width * 3;
      uint8_t *dst = (uint8_t *)dstarray->data + y * output_Width * 3;
      for (int x = 0; x < output_Width; x++) {
        uint8_t r = src[x * 3 + 0];
        uint8_t g = src[x * 3 + 1];
        uint8_t b = src[x * 3 + 2];
        dst[x * 3 + 0] = r;
        dst[x * 3 + 1] = g;
        dst[x * 3 + 2] = b;
      }
    }
    CUDA_CHECK(cudaFree(img_device));
    CUDA_CHECK(cudaFreeHost(img_host));
    return image;
}

static PyObject* add_capacity(PyObject* self, PyObject* args){
  int n;
  if (!PyArg_ParseTuple(args, "i", &n))
    return NULL;

  list_of_images.reserve(n);
  return Py_None;
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
  list_of_crops.push_back(my_crop);

  return Py_None;
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
  char my_arr[rows * nchannels * cols];

  for(size_t length = 0; length<(rows * nchannels * cols); length++){
    my_arr[length] = (*(char *)PyArray_GETPTR1(image, length));
  }

  cv::Mat my_img = cv::Mat(cv::Size(cols, rows), CV_8UC3, &my_arr);

  list_of_images.push_back(my_img);

  return Py_None;
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
    return Py_None;
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

  // Create GPU buffers on device
  CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BatchSize * 3 * InputH * InputW * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], BatchSize * OutputSize * sizeof(float)));
  output = new float[BatchSize * OutputSize];

  // prepare input data cache in pinned memory 
  CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
  // prepare input data cache in device memory
  CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
  return Py_None; 
}

static PyObject* perform_inference(PyObject* self, PyObject* args){
  
  int length = list_of_images.size();
  printf("Total Images taken: %d \n", length);

  float mean_values[3] = {0.485, 0.456, 0.406};
  float scale_values[3] = {0.229, 0.224, 0.225};

  
  float* buffer_idx = (float*)buffers[inputIndex];
  for(int b=0; b < length; b++){
    cv::Mat img = list_of_images[b];
    cv::Rect crop_of_img  = list_of_crops[b];

    size_t  size_image = img.cols * img.rows * 3;
    size_t  size_image_dst = inputHeight * inputWidth * 3;
    //copy data to pinned memory
    memcpy(img_host,img.data,size_image);
    //copy data to device memory
    CUDA_CHECK(cudaMemcpyAsync(img_device,img_host,size_image,cudaMemcpyHostToDevice,stream));
    preprocess_kernel_img(img_device, img.cols, img.rows, 
        buffer_idx, inputWidth, inputHeight, 
        mean_values, scale_values,
        crop_of_img);
        /* stream);        */
    buffer_idx += size_image_dst; 
  }
  
  auto start = std::chrono::system_clock::now();
  context->enqueue(batchSize, (void**)buffers, stream, nullptr);
  auto end = std::chrono::system_clock::now();
  std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

  try{
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[outputIndex], length * outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
  }catch(std::runtime_error error){
    std::cout << "Cuda Exception caught" << error.what() <<std::endl;    
  }

  cudaStreamSynchronize(stream);
  for (int b = 0; b < length; b++) {
    for (unsigned int i = 0; i < outputSize; i++)
    {
      std::cout << output[b + i] << ", ";
    }
    std::cout << std::endl;
  }

  return Py_None;
}

static PyObject* clear_images(PyObject* self, PyObject* args){
  list_of_images.clear();
  return Py_None; 
}


static PyObject* clear_crops(PyObject* self, PyObject* args){
  list_of_crops.clear();
  return Py_None; 
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
  return Py_None; 
}


static PyMethodDef myMethods[] = {
    {"perform_inference", perform_inference, METH_VARARGS, "do inference with CUDA"},
    {"add_capacity", add_capacity, METH_VARARGS, "initialize capacity"},
    {"load_engine", load_engine, METH_VARARGS, "initialize trt engine"},
    {"add_images", add_images, METH_VARARGS, "add images"},
    {"add_crops", add_crops, METH_VARARGS, "add images"},
    {"clear_images", clear_images, METH_VARARGS, "clear images"},
    {"clear_crops", clear_crops, METH_VARARGS, "clear images"},
    {"clear_cache", clear_cache, METH_VARARGS, "clear images"},
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

