/*
MIT License

Copyright (c) 2018 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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


#include <iostream>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <unistd.h>
#include <vector>

#include "rocal_api.h"

#include "opencv2/opencv.hpp"
using namespace cv;
#if USE_OPENCV_4
#define CV_LOAD_IMAGE_COLOR IMREAD_COLOR
#define CV_BGR2GRAY COLOR_BGR2GRAY
#define CV_GRAY2RGB COLOR_GRAY2RGB
#define CV_RGB2BGR COLOR_RGB2BGR
#define CV_FONT_HERSHEY_SIMPLEX FONT_HERSHEY_SIMPLEX
#define CV_FILLED FILLED
#define CV_WINDOW_AUTOSIZE WINDOW_AUTOSIZE
#define cvDestroyWindow destroyWindow
#endif

#define DISPLAY
using namespace std::chrono;


int test(int test_case, const char* path, int rgb, int processing_device, int width, int height, int batch_size, int shards, int shuffle);
int main(int argc, const char ** argv)
{
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    printf( "Usage: rocal_performance_tests <image-dataset-folder> <width> <height> <test_case> <batch_size> <gpu=1/cpu=0> <rgb=1/grayscale=0> <shard_count>  <shuffle=1>\n" );
    if(argc < MIN_ARG_COUNT)
        return -1;

    int argIdx = 0;
    const char * path = argv[++argIdx];
    int width = atoi(argv[++argIdx]);
    int height = atoi(argv[++argIdx]);

    int rgb = 1;// process color images
    bool processing_device = 1;
    int test_case = 0;
    int batch_size = 10;
    int shards = 4;
    int shuffle = 0;

    if (argc >= argIdx + MIN_ARG_COUNT)
        test_case = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        batch_size = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        processing_device = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        rgb = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
	shards = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
	shuffle = atoi(argv[++argIdx]);

    test(test_case, path, rgb, processing_device, width, height, batch_size, shards, shuffle);

    return 0;
}

int test(int test_case, const char* path, int rgb, int processing_device, int width, int height, int batch_size, int shards, int shuffle)
{
    size_t num_threads = shards;
    int inputBatchSize = batch_size;
    int decode_max_width = 0;
    int decode_max_height = 0;
    std::cout << ">>> test case " << test_case << std::endl;
    std::cout << ">>> Running on " << (processing_device ? "GPU" : "CPU") << " , "<< (rgb ? " Color ":" Grayscale ")<< std::endl;
    printf(">>> Batch size = %d -- shard count = %lu\n", inputBatchSize, num_threads);

    RocalImageColor color_format = (rgb != 0) ? RocalImageColor::ROCAL_COLOR_RGB24 : RocalImageColor::ROCAL_COLOR_U8;
    RocalTensorLayout tensorLayout = RocalTensorLayout::ROCAL_NHWC;
    RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_UINT8;
    auto handle = rocalCreate(inputBatchSize, processing_device ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU, 0, 1);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not create the Rocal context\n";
        return -1;
    }

    /*>>>>>>>>>>>>>>>> Creating Rocal parameters  <<<<<<<<<<<<<<<<*/

    rocalSetSeed(0);

    // Creating uniformly distributed random objects to override some of the default augmentation parameters
    RocalFloatParam rand_crop_area = rocalCreateFloatUniformRand(0.3, 0.5);
    RocalIntParam color_temp_adj = rocalCreateIntParameter(-50);

    // Creating a custom random object to set a limited number of values to randomize the rotation angle
    const size_t num_values = 3;
    float values[num_values] = {0, 10, 135};
    double frequencies[num_values] = {1, 5, 5};
    RocalFloatParam rand_angle = rocalCreateFloatRand(values, frequencies,
                                                    sizeof(values) / sizeof(values[0]));


    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/
    RocalTensor image1,image2, input1;
    RocalTensor image0_b;

    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
    if (decode_max_height <= 0 || decode_max_width <= 0)
        input1 = rocalJpegFileSource(handle, path, color_format, num_threads, false, shuffle, false);
    else
        input1 = rocalJpegFileSource(handle, path, color_format, num_threads, false, shuffle, false,
                                    ROCAL_USE_USER_GIVEN_SIZE, decode_max_width, decode_max_height);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "JPEG source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }

    RocalFloatParam alpha = rocalCreateFloatParameter(1.0);
    RocalFloatParam beta = rocalCreateFloatParameter(12.5);
    RocalFloatParam gamma = rocalCreateFloatParameter(5.0);
    RocalIntParam contrast_min = rocalCreateIntParameter(15);
    RocalIntParam contrast_max = rocalCreateIntParameter(75);
    RocalIntParam flip_h = rocalCreateIntParameter(1);
    RocalIntParam flip_v = rocalCreateIntParameter(0);
    RocalFloatParam exposure_val = rocalCreateFloatParameter(0.50);
    RocalIntParam blend_pt = rocalCreateIntParameter(0.5);
    RocalFloatParam color_twist_alpha = rocalCreateFloatParameter(1.0);
    RocalFloatParam color_twist_beta = rocalCreateFloatParameter(15.2);
    RocalFloatParam color_twist_hue = rocalCreateFloatParameter(150);
    RocalFloatParam color_twist_saturation = rocalCreateFloatParameter(0.3);
    RocalFloatParam crop_width = rocalCreateFloatParameter(100);
    RocalFloatParam crop_height = rocalCreateFloatParameter(100);
    RocalFloatParam crop_depth = rocalCreateFloatParameter(0);
    RocalFloatParam crop_x = rocalCreateFloatParameter(0);
    RocalFloatParam crop_y = rocalCreateFloatParameter(0);
    RocalFloatParam crop_z = rocalCreateFloatParameter(0);
    RocalFloatParam noise_val = rocalCreateFloatParameter(0.5);
    RocalFloatParam salt_prob = rocalCreateFloatParameter(0.1);
    RocalFloatParam salt_val = rocalCreateFloatParameter(1.0);
    RocalFloatParam pepper_val = rocalCreateFloatParameter(0.0);

    
    int resize_w = width, resize_h = height;

    switch (test_case)
    {
    case 0:
    {
         std::cout << ">>>>>>> Running "
                  << "rocalResize" << std::endl;
        image1 = rocalResize(handle, input1, tensorLayout, tensorOutputType,resize_w , resize_h, true);
    }
    break;
    case 1:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalColorCast" << std::endl;
        // image1 = rocalColorCast(handle, input1, tensorLayout, tensorOutputType, true);
    break;
    }
    case 2:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalRotate" << std::endl;
        // image1 = rocalRotate(handle, input1, tensorLayout, tensorOutputType, true,300, 300,0);
    }
    break;
    
    case 3:
    {
        std::cout << ">>>>>>> Running "
                  << "Brightness" << std::endl;
        image1 = rocalBrightness(handle, input1, true,alpha,beta);
    }
    break;
    case 4:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalGamma" << std::endl;
        image1 = rocalGamma(handle, input1, tensorLayout, tensorOutputType, true, gamma);
    }
    break;
    
    
    case 5:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalContrast" << std::endl;
        image1 = rocalContrast(handle, input1, tensorLayout, tensorOutputType, true,contrast_min,contrast_max);
    break;
    }
    
    case 6:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalFlip" << std::endl;
        image1 = rocalFlip(handle, input1, tensorLayout, tensorOutputType, true,flip_h,flip_v);
    }
    break;
    
    case 7:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalBlur" << std::endl;
        // image1 = rocalBlur(handle, input1, tensorLayout, tensorOutputType, true);
    break;
    }
    case 8:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalBlend" << std::endl;
        image2 = rocalFlip(handle, input1, tensorLayout, tensorOutputType, true,flip_h,flip_v);
        image1 = rocalBlend(handle, input1,image2, tensorLayout, tensorOutputType, true,blend_pt);

    }
    break;
    case 9:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalwarp_affine" << std::endl;
        // image1 = rocalWarpAffine(handle, input1, tensorLayout, tensorOutputType, true);

    }
break;
    case 10:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalFisheye" << std::endl;
        // image1 = rocalFisheye(handle, input1, tensorLayout, tensorOutputType, true);
    }
    break;
    
    case 11:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalVignette" << std::endl;
        // image1 = rocalVignette(handle, input1, tensorLayout, tensorOutputType, true);
    }
    break;
    case 12:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalJitter" << std::endl;
        // image1 = rocalJitter(handle, input1, tensorLayout, tensorOutputType, true);
    }
    break;
    case 13:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalNoise" << std::endl;
        image1 = rocalNoise(handle, input1, tensorLayout, tensorOutputType, true,noise_val,salt_prob,salt_val,pepper_val);
    }
    break;
    case 14:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalSnow" << std::endl;
        // image1 = rocalSnow(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;
case 15:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalRain" << std::endl;
        // image1 = rocalRain(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;
case 16:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalColorTemperature" << std::endl;
        // image1 = rocalColorTemperature(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;
case 17:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalfog" << std::endl;
        // image1 = rocalFog(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;
case 18:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalLensCorrection" << std::endl;
        // image1 = rocalLensCorrection(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;
case 19:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalColorPixelate" << std::endl;
        // image1 = rocalPixelate(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;
case 20:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalExposure" << std::endl;
        image1 = rocalExposure(handle, input1, tensorLayout, tensorOutputType, true,exposure_val);
    break;
    }
case 21:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalHue" << std::endl;
        // image1 = rocalHue(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;
    
case 22:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalSaturation" << std::endl;
        // image1 = rocalSaturation(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;
case 23:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalspatter" << std::endl;
        // image1 = rocalSpatter(handle, input1, tensorLayout, tensorOutputType, true);
    break;
    }
case 24:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalColorTwist" << std::endl;
        image1 = rocalColorTwist(handle, input1, tensorLayout, tensorOutputType, true,color_twist_alpha,color_twist_beta,color_twist_hue,color_twist_saturation);
    }
    break;
    case 25:
    {
        std::vector<float> mean{0, 0, 0};
        std::vector<float> sdev{1, 1, 1};
        resize_h=200;
        resize_w=200;
        std::cout << ">>>>>>> Running "
                  << " CropMirrorNormalize " << std::endl;
        image1 = rocalCropMirrorNormalize(handle, input1, tensorLayout, tensorOutputType, 3, resize_h, resize_w, 0, 0, 0, mean, sdev, true);
        break;
    }
    case 26:
    {
         std::cout << ">>>>>>> Running "
                  << "rocalcrop" << std::endl;
        image1 = rocalCropCenterFixed(handle, input1, tensorLayout, tensorOutputType,100,100,3,true);

    }
    break;
    case 27:
    {
        std::vector<float> mean{0, 0, 0};
        std::vector<float> sdev{1, 1, 1};
        std::cout << ">>>>>>> Running "
                  << " Resize Mirror Normalize " << std::endl;
        // image1 = rocalResizeMirrorNormalize(handle, input1, tensorLayout, tensorOutputType, 3,resize_w , resize_h, 0, mean, sdev,true);
        break;
    }
    case 28:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalColorJitter" << std::endl;
        // image1 = rocalColorJitter(handle, input1, tensorLayout, tensorOutputType, true);
    }
break;

    case 29:
    {
        std::cout << ">>>>>>> Running "
                  << "rocalGridmask" << std::endl;
        // image1 = rocalGridmask(handle, input1, tensorLayout, tensorOutputType, true);

    }
    break;
   
    
    




    




	default:
            std::cout << "Not a valid option! Exiting!\n";
            return -1;
    }

    // Calling the API to verify and build the augmentation graph
    rocalVerify(handle);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not verify the augmentation graph " << rocalGetErrorMessage(handle);
        return -1;
    }



    // printf("Augmented copies count %lu\n", rocalGetAugmentationBranchCount(handle));



    printf("Going to process images\n");
//    printf("Remaining images %d \n", rocalGetRemainingImages(handle));
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    int i = 0;
    for(int epoch = 0; epoch < 1; epoch++)
    {
        while (rocalGetRemainingImages(handle) >= inputBatchSize) {
            if (rocalRun(handle) != 0) {
                break;
            }
        }
        // rocalResetLoaders(handle);
    }
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(t2 - t1).count();
    auto rocal_timing = rocalGetTimingInfo(handle);
    std::cout << "Load     time " << rocal_timing.load_time << std::endl;
    std::cout << "Decode   time " << rocal_timing.decode_time << std::endl;
    std::cout << "Process  time " << rocal_timing.process_time << std::endl;
    std::cout << "Transfer time " << rocal_timing.transfer_time << std::endl;
    std::cout << "Total time " << dur << std::endl;
    std::cout << ">>>>> Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us " << std::endl;

    rocalRelease(handle);


    return 0;
}
