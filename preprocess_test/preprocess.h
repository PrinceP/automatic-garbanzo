#ifndef __PREPROCESS_H
#define __PREPROCESS_H

#include <cuda_runtime.h>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

struct AffineMatrix{
    float value[6];
    float mean_values[3];
    float scale_values[3];
};

//extern "C" { 
  void preprocess_kernel_img(uint8_t* src, int src_width, int src_height,
                           float* dst, int dst_width, int dst_height,
                           cv::Rect crop,
                           float Imean_values[3],
                           float Iscale_values[3],
                           cudaStream_t stream);
  /* void preprocess_kernel_img(uint8_t* src, int src_width, int src_height, */
  /*                          float* dst, int dst_width, int dst_height); */

//}


#endif  // __PREPROCESS_H
