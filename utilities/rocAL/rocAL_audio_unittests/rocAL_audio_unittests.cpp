/*
MIT License

Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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
#endif

#define DISPLAY 1
#define METADATA 0 // Switch the meta-data part once the meta-data reader (file list reader) is introduced
using namespace std::chrono;

int test(int test_case, const char *path, float sample_rate, int downmix, unsigned max_frames, unsigned max_channels, int gpu);
int main(int argc, const char **argv)
{
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    printf("Usage: rocAL_audio_unittests <audio-dataset-folder> <test_case> <sample-rate> <downmix> <max_frames> <max_channels> gpu=1/cpu=0 \n");
    if (argc < MIN_ARG_COUNT)
        return -1;

    int argIdx = 0;
    const char *path = argv[++argIdx];
    unsigned test_case = 0;
    float sample_rate = 0.0;
    bool downmix = false;
    unsigned max_frames = 1;
    unsigned max_channels = 1;
    bool gpu = 0;

    if (argc >= argIdx + MIN_ARG_COUNT)
        test_case = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        sample_rate = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        downmix = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        max_frames = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        max_channels = atoi(argv[++argIdx]);

    if (argc >= argIdx + MIN_ARG_COUNT)
        gpu = atoi(argv[++argIdx]);

    int return_val = test(test_case, path, sample_rate, downmix, max_frames, max_channels, gpu);
    return return_val;
}

int test(int test_case, const char *path, float sample_rate, int downmix, unsigned max_frames, unsigned max_channels, int gpu)
{
    int inputBatchSize = 3;
    std::cout << ">>> test case " << test_case << std::endl;
    std::cout << ">>> Running on " << (gpu ? "GPU" : "CPU") << std::endl;

    auto handle = rocalCreate(inputBatchSize,
                              gpu ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU, 0,
                              1);

    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Could not create the Rocal contex\n";
        return -1;
    }

    /*>>>>>>>>>>>>>>>> Creating Rocal parameters  <<<<<<<<<<<<<<<<*/

    RocalMetaData metadata_output;

    //Decoder
    RocalTensor input1; // Uncomment when augmentations are enabled
    input1 = rocalAudioFileSourceSingleShard(handle, path, 0, 1, true, false, false, false, max_frames, max_channels, 0);
    if (rocalGetStatus(handle) != ROCAL_OK) {
        std::cout << "Audio source could not initialize : " << rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }
    RocalTensor output;

    switch (test_case)
    {
        case 1:
        {
            std::cout<< "\n Augmentation - rocalPreEmphasisFilter ";
            RocalTensorLayout tensorLayout = RocalTensorLayout::ROCAL_NONE;
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            output = rocalPreEmphasisFilter(handle, input1, tensorOutputType, true);
            break;
        }
        case 3:
        {
            auto non_silent_region = rocalNonSilentRegion(handle, input1, true, -60, 1, -1, 3);
            break;
        }
        case 5:
        {
            std::cerr << "running slice" << std::endl;
            RocalTensorLayout tensorLayout = RocalTensorLayout::ROCAL_NONE;
            RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_FP32;
            const size_t num_values = 3;
            std::pair <RocalTensor, RocalTensor>  non_silent_region_output;
            non_silent_region_output = rocalNonSilentRegion(handle, input1, false, -60, 0.0, -1, 3);

            int crop_shape[] = {2, 2};
            rocalROIRandomCrop(handle, input1, crop_shape);

            // int *temp = static_cast<int *>(non_silent_region_output.second->buffer());
            // for(int k = 0; k < inputBatchSize; k++)
            //     temp[k] = 2;

            RocalTensor crop_begin = rocalGetROIRandomCropValues(handle);
            output = rocalSlice(handle, input1, true, crop_begin, non_silent_region_output.second, {0.3f}, {0}, false, false, RocalOutOfBoundsPolicy::PAD, tensorOutputType);
            break;
        }
        default:
        {
            std::cout << "Not a valid pipeline type ! Exiting!\n";
            return -1;
        }

    }

    rocalVerify(handle);
    if (rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Could not verify the augmentation graph " << rocalGetErrorMessage(handle);
        return -1;
    }

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/
    cv::Mat mat_output, mat_input, mat_color;
    int iteration = 0;
    RocalTensorList output_tensor_list;

    while (rocalGetRemainingImages(handle) >= static_cast<size_t>(inputBatchSize))
    {
        std::cout<<"\n rocalGetRemainingImages:: "<<rocalGetRemainingImages(handle)<<"\t inputBatchsize:: "<< inputBatchSize;
        std::cout<<"\n iteration:: "<<iteration;
        iteration++;
        if (rocalRun(handle) != 0) {
            break;
        }

        // RocalTensor crop_output = rocalGetROIRandomCropValues(handle);
        // int *buf = static_cast<int *>(crop_output->buffer());
        // std::cerr << "starting values: " << std::endl;
        // for(int k = 0; k < inputBatchSize * 1; k++)
        //     std::cerr << buf[k] << " ";
        // std::cerr<<std::endl;

        std::vector<float> audio_op;
        output_tensor_list = rocalGetOutputTensors(handle);
        std::cout << "\n *****************************Audio output**********************************\n";
        std::cout << "\n **************Printing the first 5 values of the Audio buffer**************\n";
        // for(uint idx = 0; idx < output_tensor_list->size(); idx++) {
        //     float * buffer = (float *)output_tensor_list->at(idx)->buffer();
        //     for(int n = 0; n < 5; n++)
        //         std::cout << buffer[n] << "\n";
        // }

        if (METADATA) {
            RocalTensorList labels = rocalGetImageLabels(handle);
            for(uint i = 0; i < labels->size(); i++) {
                int * labels_buffer = (int *)(labels->at(i)->buffer());
                std::cout << ">>>>> LABELS : " << labels_buffer[0] << "\t";
            }
        }
        std::cout<<"******************************************************************************\n";
    }
    rocalRelease(handle);
    return 0;
}
