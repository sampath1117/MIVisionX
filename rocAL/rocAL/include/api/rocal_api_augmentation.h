/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef MIVISIONX_ROCAL_API_AUGMENTATION_H
#define MIVISIONX_ROCAL_API_AUGMENTATION_H
#include "rocal_api_types.h"

/// Accepts U8 and RGB24 input.
// Rearranges the order of the frames in the sequences with respect to new_order.
// new_order can have values in the range [0, sequence_length).
// Frames can be repeated or dropped in the new_order.
/// \param context
/// \param input
/// \param new_order
/// \param is_output
/// \return
extern "C"  RocalTensor  ROCAL_API_CALL rocalSequenceRearrange(RocalContext p_context, RocalTensor p_input,
                                                              std::vector<unsigned int>& new_order,
                                                              bool is_output);

/// Accepts U8 and RGB24 input.
/// \param context
/// \param input
/// \param dest_width
/// \param dest_height
/// \param is_output
/// \param scaling_mode The resize scaling_mode to resize the image.
/// \param max_size Limits the size of the resized image.
/// \param resize_shorter The length of the shorter dimension of the image.
/// \param resize_longer The length of the larger dimension of the image.
/// \param interpolation_type The type of interpolation to be used for resize.
/// \return
extern "C"  RocalTensor  ROCAL_API_CALL rocalResize(RocalContext context, RocalTensor input,
                                                   unsigned dest_width, unsigned dest_height,
                                                   bool is_output,
                                                   RocalResizeScalingMode scaling_mode = ROCAL_SCALING_MODE_STRETCH,
                                                   std::vector<unsigned> max_size = {},
                                                   unsigned resize_shorter = 0,
                                                   unsigned resize_longer = 0,
                                                   RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                                   RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                   RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// Accepts U8 and RGB24 input.
/// \param context Rocal context
/// \param input Input Rocal Image
/// \param dest_width The output width
/// \param dest_height The output height
/// \param mean The channel mean values
/// \param std_dev The channel standard deviation values
/// \param is_output True: the output image is needed by user and will be copied to output buffers using the data
/// transfer API calls. False: the output image is just an intermediate image, user is not interested in
/// using it directly. This option allows certain optimizations to be achieved.
/// \param p_mirror Parameter to enable horizontal flip for output image.
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalResizeMirrorNormalize(RocalContext p_context,
                                            RocalTensor p_input,
                                            unsigned dest_width, unsigned dest_height,
                                            std::vector<float> &mean,
                                            std::vector<float> &std_dev,
                                            bool is_output,
                                            RocalResizeScalingMode scaling_mode = ROCAL_SCALING_MODE_STRETCH,
                                            std::vector<unsigned> max_size = {},
                                            unsigned resize_shorter = 0,
                                            unsigned resize_longer = 0,
                                            RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                            RocalIntParam mirror = NULL,
                                            RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                            RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);


/// Accepts U8 and RGB24 input.
/// \param context
/// \param input
/// \param dest_width
/// \param dest_height
/// \param is_output
/// \param area
/// \param x_center_drift
/// \param y_center_drift
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalCropResize(RocalContext context, RocalTensor input,
                                                   unsigned dest_width, unsigned dest_height,
                                                   bool is_output,
                                                   RocalFloatParam area = NULL,
                                                    RocalFloatParam aspect_ratio = NULL,
                                                    RocalFloatParam x_center_drift = NULL,
                                                    RocalFloatParam y_center_drift = NULL,
                                                    RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                    RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// Accepts U8 and RGB24 input. Crops the input image to a new area and same aspect ratio.
/// \param context
/// \param input
/// \param dest_width
/// \param dest_height
/// \param is_output
/// \param area
/// \param x_center_drift
/// \param y_center_drift
/// \return
extern "C"  RocalTensor  ROCAL_API_CALL rocalCropResizeFixed(RocalContext context, RocalTensor input, unsigned dest_width,
                                                            unsigned dest_height, bool is_output, float area, float aspect_ratio,
                                                            float x_center_drift, float y_center_drift,
                                                            RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                            RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// Accepts U8 and RGB24 input. The output image dimension can be set to new values allowing the rotated image to fit,
/// otherwise; the image is cropped to fit the result.
/// \param context Rocal context
/// \param input Input Rocal Image
/// \param is_output True: the output image is needed by user and will be copied to output buffers using the data
/// transfer API calls. False: the output image is just an intermediate image, user is not interested in
/// using it directly. This option allows certain optimizations to be achieved.
/// \param angle Rocal parameter defining the rotation angle value in degrees.
/// \param dest_width The output width
/// \param dest_height The output height
/// \return Returns a new image that keeps the result.
extern "C"  RocalTensor  ROCAL_API_CALL rocalRotate(RocalContext context, RocalTensor input, bool is_output,
                                                RocalFloatParam angle = NULL,  unsigned dest_width = 0,
                                                unsigned dest_height = 0,
                                                RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                                RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// Accepts U8 and RGB24 input. The output image dimension can be set to new values allowing the rotated image to fit,
/// otherwise; the image is cropped to fit the result.
/// \param context Rocal context
/// \param input Input Rocal Image
/// \param dest_width The output width
/// \param dest_height The output height
/// \param is_output Is the output image part of the graph output
/// \param angle The rotation angle value in degrees.
/// \return Returns a new image that keeps the result.
extern "C"  RocalTensor  ROCAL_API_CALL rocalRotateFixed(RocalContext context, RocalTensor input, float angle,
                                                    bool is_output, unsigned dest_width = 0, unsigned dest_height = 0,
                                                    RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                                    RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                    RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// Accepts U8 and RGB24 inputs
/// \param context Rocal context
/// \param input Input Rocal tensor
/// \param is_output is the output tensor part of the graph output
/// \param alpha controls contrast of the image
/// \param beta controls brightness of the image
/// \param tensor_output_layout the layout of the output tensor
/// \param tensor_output_datatype the data type of the output tensor
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalBrightness(RocalContext context, RocalTensor input, bool is_output,
                                                      RocalFloatParam alpha = NULL, RocalFloatParam beta = NULL,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param shift
/// \param is_output
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalBrightnessFixed(RocalContext context, RocalTensor input,
                                                            bool is_output,
                                                            float alpha, float beta,
                                                            RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                            RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \param alpha
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalGamma(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      RocalFloatParam alpha = NULL,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param alpha
/// \param is_output
/// \return

extern "C" RocalTensor ROCAL_API_CALL rocalGammaFixed(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      float alpha,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// Accepts U8 and RGB24 inputs.
/// \param context
/// \param input
/// \param is_output
/// \param min
/// \param max
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalContrast(RocalContext context, RocalTensor input,
                                                    bool is_output,
                                                    RocalFloatParam min = NULL, RocalFloatParam max = NULL,
                                                    RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                    RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);


/// Accepts U8 and RGB24 inputs.
/// \param context
/// \param input
/// \param min
/// \param max
/// \param is_output
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalContrastFixed(RocalContext context, RocalTensor input,
                                                         bool is_output,
                                                         float min = NULL, float max = NULL,
                                                         RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                         RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);



///
/// \param context
/// \param input
/// \param axis
/// \param is_output
/// \return
extern "C"  RocalTensor  ROCAL_API_CALL rocalFlip(RocalContext context, RocalTensor input, bool is_output,
                                                RocalIntParam horizonal_flag = NULL, RocalIntParam vertical_flag = NULL,
                                                RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

///
/// \param context
/// \param input
/// \param axis
/// \param is_output
/// \return
extern "C"  RocalTensor  ROCAL_API_CALL rocalFlipFixed(RocalContext context, RocalTensor input,
                                                       int horizonal_flag, int vertical_flag, bool is_output,
                                                       RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                       RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \param sdev
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalBlur(RocalContext context, RocalTensor input,
                                                bool is_output,
                                                RocalIntParam sdev = NULL,
                                                RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

///
/// \param context
/// \param input
/// \param sdev
/// \param is_output
/// \return

extern "C" RocalTensor ROCAL_API_CALL rocalBlurFixed(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      int sdev,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// Blends two input images given the ratio: output = input1*ratio + input2*(1-ratio)
/// \param context
/// \param input1
/// \param input2
/// \param is_output
/// \param ratio Rocal parameter defining the blending ratio, should be between 0.0 and 1.0.
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalBlend(RocalContext context, RocalTensor input1, RocalTensor input2,
                                                bool is_output,
                                                RocalFloatParam ratio = NULL,
                                                RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// Blends two input images given the ratio: output = input1*ratio + input2*(1-ratio)
/// \param context
/// \param input1
/// \param input2
/// \param ratio Float value defining the blending ratio, should be between 0.0 and 1.0.
/// \param is_output
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalBlendFixed(RocalContext context, RocalTensor input1, RocalTensor input2,
                                                      bool is_output,
                                                      float ratio,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// \param context
/// \param input
/// \param is_output
/// \param x0
/// \param x1
/// \param y0
/// \param y1
/// \param o0
/// \param o1
/// \param dest_height
/// \param dest_width
/// \return
extern "C"  RocalTensor  ROCAL_API_CALL rocalWarpAffine(RocalContext context, RocalTensor input, bool is_output,
                                                     unsigned dest_height = 0, unsigned dest_width = 0,
                                                     RocalFloatParam x0 = NULL, RocalFloatParam x1 = NULL,
                                                     RocalFloatParam y0= NULL, RocalFloatParam y1 = NULL,
                                                     RocalFloatParam o0 = NULL, RocalFloatParam o1 = NULL,
                                                     RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                                     RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                     RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

///
/// \param context
/// \param input
/// \param x0
/// \param x1
/// \param y0
/// \param y1
/// \param o0
/// \param o1
/// \param is_output
/// \param dest_height
/// \param dest_width
/// \return
extern "C"  RocalTensor  ROCAL_API_CALL rocalWarpAffineFixed(RocalContext context, RocalTensor input, float x0, float x1,
                                                         float y0, float y1, float o0, float o1, bool is_output,
                                                         unsigned int dest_height = 0, unsigned int dest_width = 0,
                                                         RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                                         RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                         RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// \param context
/// \param input
/// \param is_output
/// \return
extern "C"  RocalTensor  ROCAL_API_CALL rocalFishEye(RocalContext context, RocalTensor input, bool is_output,
                                                     RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                     RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

///
/// \param context
/// \param input
/// \param is_output
/// \param sdev
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalVignette(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      RocalFloatParam sdev = NULL,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);


/// \param context
/// \param input
/// \param sdev
/// \param is_output
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalVignetteFixed(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      float sdev,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);


///
/// \param context
/// \param input
/// \param is_output
/// \param min
/// \param max
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalJitter(RocalContext context, RocalTensor input,
                                                bool is_output,
                                                RocalIntParam kernel_size = NULL,
                                                int seed=0,
                                                RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

///
/// \param context
/// \param input
/// \param min
/// \param max
/// \param is_output
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalJitterFixed(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      int kernel_size,
                                                      int seed=0,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

///
/// \param context
/// \param input
/// \param is_output
/// \param sdev
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalSnPNoise(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      RocalFloatParam noise_prob = NULL, RocalFloatParam salt_prob = NULL,
                                                      RocalFloatParam noise_val = NULL, RocalFloatParam salt_val = NULL,
                                                      int seed=11110,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

///
/// \param context
/// \param input
/// \param sdev
/// \param is_output
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalSnPNoiseFixed(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      float noise_prob, float salt_prob,
                                                      float noise_val, float salt_val,
                                                      int seed=11110,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

///
/// \param context
/// \param input
/// \param is_output
/// \param sdev
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalSnow(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      RocalFloatParam shift = NULL,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// \param context
/// \param input
/// \param sdev
/// \param is_output
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalSnowFixed(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      float shift,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

///
/// \param context
/// \param input
/// \param is_output
/// \param rain_value
/// \param rain_width
/// \param rain_heigth
/// \param rain_transparency
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalRain(RocalContext context, RocalTensor input,
                                                bool is_output,
                                                RocalFloatParam rain_value = NULL,
                                                RocalIntParam rain_width = NULL,
                                                RocalIntParam rain_height = NULL,
                                                RocalFloatParam rain_transparency = NULL,
                                                RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);


/// \param context
/// \param input
/// \param is_output
/// \param rain_value
/// \param rain_width
/// \param rain_heigth
/// \param rain_transparency
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalRainFixed(RocalContext context, RocalTensor input,
                                                    bool is_output,
                                                    float rain_value,
                                                    int rain_width,
                                                    int rain_height,
                                                    float rain_transparency,
                                                    RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                    RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// \param context
/// \param input
/// \param is_output
/// \param adjustment
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalColorTemp(RocalContext context, RocalTensor input,
                                                     bool is_output,
                                                     RocalIntParam adjustment = NULL,
                                                     RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                     RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// \param context
/// \param input
/// \param adjustment
/// \param is_output
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalColorTempFixed(RocalContext context, RocalTensor input,
                                                          bool is_output,
                                                          int adjustment,
                                                          RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                          RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// \param context
/// \param input
/// \param is_output
/// \param fog_value
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalFog(RocalContext context, RocalTensor input,
                                                bool is_output,
                                                RocalFloatParam fog_value = NULL,
                                                RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// \param context
/// \param input
/// \param fog_value
/// \param is_output
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalFogFixed(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      float fog_value,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// \param context
/// \param input
/// \param is_output
/// \param strength
/// \param zoom
/// \return
extern "C"  RocalTensor  ROCAL_API_CALL rocalLensCorrection(RocalContext context, RocalTensor input, bool is_output,
                                                        RocalFloatParam strength = NULL,
                                                        RocalFloatParam zoom = NULL,
                                                        RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                        RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// \param context
/// \param input
/// \param strength
/// \param zoom
/// \param is_output
/// \return
extern "C"  RocalTensor  ROCAL_API_CALL rocalLensCorrectionFixed(RocalContext context, RocalTensor input,
                                                                float strength, float zoom, bool is_output,
                                                                RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                                RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

///
/// \param context
/// \param input
/// \param is_output
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalPixelate(RocalContext context, RocalTensor input,
                                                    bool is_output,
                                                    RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                    RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);


///
/// \param context
/// \param input
/// \param is_output
/// \param shift
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalExposure(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      RocalFloatParam shift = NULL,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// \param context
/// \param input
/// \param is_output
/// \param shift
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalExposureFixed(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      float shift,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

///
/// \param context
/// \param input
/// \param is_output
/// \return

extern "C" RocalTensor ROCAL_API_CALL rocalHue(RocalContext context, RocalTensor input,
                                                bool is_output,
                                                RocalFloatParam p_hue = NULL,
                                                RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);


///
/// \param context
/// \param input
/// \param is_output
/// \param hue
/// \return

extern "C" RocalTensor ROCAL_API_CALL rocalHueFixed(RocalContext context, RocalTensor input,
                                                    bool is_output,
                                                    float hue,
                                                    RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                    RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);


/// Accepts U8 and RGB24 inputs.
/// \param context
/// \param input
/// \param is_output
/// \param min
/// \param max
/// \return

extern "C" RocalTensor ROCAL_API_CALL rocalSaturation(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      RocalFloatParam p_sat = NULL,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

extern "C" RocalTensor ROCAL_API_CALL rocalSaturationFixed(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      float sat,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// Accepts U8 and RGB24 inputs.
/// \param context
/// \param input
/// \param is_output
/// \param min
/// \param max
/// \return

extern "C"  RocalTensor  ROCAL_API_CALL rocalCopy(RocalContext context, RocalTensor input, bool is_output);

///
/// \param context
/// \param input
/// \param is_output
/// \return
extern "C"  RocalTensor  ROCAL_API_CALL rocalNop(RocalContext context, RocalTensor input, bool is_output);


/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalColorTwist(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      RocalFloatParam alpha = NULL,
                                                      RocalFloatParam beta = NULL,
                                                      RocalFloatParam hue = NULL,
                                                      RocalFloatParam sat = NULL,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);


/// Accepts U8 and RGB24 inputs
/// \param context
/// \param input
/// \param is_output
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalColorTwistFixed(RocalContext context, RocalTensor input,
                                                      bool is_output,
                                                      float alpha,
                                                      float beta,
                                                      float hue,
                                                      float sat,
                                                      RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                      RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);


/// Accepts U8 and RGB24 inputs
/// \param context Rocal context
/// \param input Input Rocal tensor
/// \param crop_height crop width of the image
/// \param crop_width crop height of the image
/// \param start_x x-coordinate, start of the input image to be cropped
/// \param start_y y-coordinate, start of the input image to be cropped
/// \param mean mean value (specified for each channel) for image normalization
/// \param std_dev standard deviation value (specified for each channel) for image normalization
/// \param is_output is the output tensor part of the graph output
/// \param mirror controls horizontal flip of the image
/// \param tensor_output_layout the layout of the output tensor
/// \param tensor_output_type the data type of the output tensor
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalCropMirrorNormalize(RocalContext context, RocalTensor input,
                                                               unsigned crop_height,
                                                               unsigned crop_width,
                                                               float start_x,
                                                               float start_y,
                                                               std::vector<float> &mean,
                                                               std::vector<float> &std_dev,
                                                               bool is_output,
                                                               RocalIntParam mirror = NULL,
                                                               RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                               RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

extern "C" RocalTensor  ROCAL_API_CALL rocalCrop(RocalContext context, RocalTensor input, bool is_output,
                                                 RocalFloatParam crop_width = NULL,
                                                 RocalFloatParam crop_height = NULL,
                                                 RocalFloatParam crop_depth = NULL,
                                                 RocalFloatParam crop_pox_x = NULL,
                                                 RocalFloatParam crop_pos_y = NULL,
                                                 RocalFloatParam crop_pos_z = NULL,
                                                 RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                 RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

extern "C"  RocalTensor  ROCAL_API_CALL rocalCropFixed(RocalContext context, RocalTensor  input,
                                                       unsigned crop_width,
                                                       unsigned crop_height,
                                                       unsigned crop_depth,
                                                       bool is_output,
                                                       float crop_pox_x,
                                                       float crop_pos_y,
                                                       float crop_pos_z,
                                                       RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                       RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);
// //// \param crop_width


extern "C" RocalTensor  ROCAL_API_CALL rocalCropCenterFixed(RocalContext context, RocalTensor input,
                                                            unsigned crop_width,
                                                            unsigned crop_height,
                                                            unsigned crop_depth,
                                                            bool output,
                                                            RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                            RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

extern "C"  RocalTensor  ROCAL_API_CALL rocalResizeCropMirrorFixed(RocalContext context, RocalTensor input,
                                                                    unsigned dest_width, unsigned dest_height,
                                                                    bool is_output,
                                                                    unsigned crop_h,
                                                                    unsigned crop_w,
                                                                    RocalIntParam mirror,
                                                                    RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                                    RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

extern "C"  RocalTensor  ROCAL_API_CALL rocalResizeCropMirror(RocalContext context, RocalTensor input,
                                                              unsigned dest_width, unsigned dest_height,
                                                              bool is_output, RocalFloatParam crop_height = NULL,
                                                              RocalFloatParam crop_width = NULL, RocalIntParam mirror = NULL,
                                                              RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                              RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// Accepts U8 and RGB24 inputs and Ouptus Cropped Images, valid bounding boxes and labels
/// \param context
/// \param input
/// \param num_of_attmpts
/// \return
extern "C" RocalTensor ROCAL_API_CALL rocalRandomCrop(RocalContext context, RocalTensor input,
                                                    bool is_output,
                                                    RocalFloatParam crop_area_factor  = NULL,
                                                    RocalFloatParam crop_aspect_ratio = NULL,
                                                    RocalFloatParam crop_pos_x = NULL,
                                                    RocalFloatParam crop_pos_y = NULL,
                                                    int num_of_attempts = 20,
                                                    RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                    RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// Accepts U8 and RGB24 inputs and Ouptus Cropped Images, valid bounding boxes and labels
/// \param context
/// \param input
/// \param IOU_threshold
/// \param num_of_attmpts
/// \return

extern "C" RocalTensor ROCAL_API_CALL rocalSSDRandomCrop(  RocalContext context, RocalTensor input,
                                                    bool is_output,
                                                    RocalFloatParam threshold = NULL,
                                                    RocalFloatParam crop_area_factor  = NULL,
                                                    RocalFloatParam crop_aspect_ratio = NULL,
                                                    RocalFloatParam crop_pos_x = NULL,
                                                    RocalFloatParam crop_pos_y = NULL,
                                                    int num_of_attempts = 20,
                                                    RocalTensorLayout rocal_tensor_output_layout = ROCAL_NHWC,
                                                    RocalTensorOutputType rocal_tensor_output_type = ROCAL_UINT8);

/// Accepts F32 audio buffers
/// \param context
/// \param input

extern "C" RocalTensor ROCAL_API_CALL rocalPreEmphasisFilter(RocalContext p_context,
                                                             RocalTensor p_input,
                                                             RocalTensorOutputType rocal_tensor_output_type,
                                                             bool is_output,
                                                             RocalFloatParam p_preemph_coeff = NULL,
                                                             RocalAudioBorderType preemph_border_type = RocalAudioBorderType::CLAMP);

/// Accepts F32 audio buffers and returns start index and length of non silent region in audio buffer
/// \param context
/// \param input

extern "C" std::pair<RocalTensor, RocalTensor> ROCAL_API_CALL rocalNonSilentRegion(RocalContext context,
                                                                                   RocalTensor input,
                                                                                   bool is_output,
                                                                                   float cutoff_db,
                                                                                   float reference_power,
                                                                                   int reset_interval,
                                                                                   int window_length);

/// Accepts F32 audio buffers
/// \param context
/// \param input

extern "C" RocalTensor ROCAL_API_CALL rocalSlice(RocalContext p_context,
                                                 RocalTensor p_input,
                                                 bool is_output,
                                                 RocalTensor anchor_tensor,
                                                 RocalTensor shape_tensor,
                                                 std::vector<float> fill_values,
                                                 std::vector<unsigned> axes,
                                                 bool normalized_anchor,
                                                 bool normalized_shape,
                                                 RocalOutOfBoundsPolicy policy,
                                                 RocalTensorOutputType rocal_tensor_output_type);

/// Accepts F32 audio buffers
/// \param context
/// \param input

extern "C" RocalTensor ROCAL_API_CALL rocalSpectrogram(RocalContext p_context,
                                                       RocalTensor p_input,
                                                       bool is_output,
                                                       std::vector<float> &window_fn,
                                                       bool center_windows,
                                                       bool reflect_padding,
                                                       RocalSpectrogramLayout spectrogram_layout,
                                                       int power,
                                                       int nfft,
                                                       int window_length,
                                                       int window_step,
                                                       RocalTensorOutputType rocal_tensor_output_type);

/// Accepts F32 audio buffers
/// \param context
/// \param input

extern "C"  RocalTensor ROCAL_API_CALL rocalMelFilterBank(RocalContext p_context,
                                                          RocalTensor p_input,
                                                          bool is_output,
                                                          float freq_high,
                                                          float freq_low,
                                                          RocalMelScaleFormula mel_formula,
                                                          int nfilter,
                                                          bool normalize,
                                                          float sample_rate,
                                                          RocalTensorOutputType rocal_tensor_output_datatype);

/// Accepts F32 audio buffers
/// \param context
/// \param input

extern "C" RocalTensor ROCAL_API_CALL rocalToDecibels(RocalContext p_context,
                                                      RocalTensor p_input,
                                                      bool is_output,
                                                      float cutoff_db,
                                                      float multiplier,
                                                      float reference_magnitude,
                                                      RocalTensorOutputType rocal_tensor_output_type);

/// Accepts F32 audio buffers
/// \param context
/// \param input

extern "C" RocalTensor ROCAL_API_CALL rocalNormalize(RocalContext p_context,
                                                     RocalTensor p_input,
                                                     bool is_output,
                                                     bool batch,
                                                     std::vector<int> axes,
                                                     float mean, float std_dev,
                                                     float scale, float shift,
                                                     int ddof, float epsilon,
                                                     RocalTensorOutputType rocal_tensor_output_type);

#endif //MIVISIONX_ROCAL_API_AUGMENTATION_H
