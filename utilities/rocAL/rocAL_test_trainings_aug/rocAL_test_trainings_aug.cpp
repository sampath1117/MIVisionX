/*
MIT License

Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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
#include<string>

#include "rocal_api.h"
#define CLASSIFICATION_TRAIN 1
// #define CLASSIFICATION_VAL 1
// #define SSD 1
// #define MASK 1
// #define RANDOMBBOXCROP

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
#define DISPLAY 1

using namespace std::chrono;

void convert_float_to_uchar_buffer(float * input_float_buffer, unsigned char * output_uchar_buffer, size_t data_size)
{
    for(size_t i = 0; i < data_size; i++)
    {
        output_uchar_buffer[i] = (unsigned char)(*(input_float_buffer + i) * 255);
    }
}

void convert_nchw_to_nhwc(unsigned char * input_chw, unsigned char * output_hwc, int n, int h, int w, int c)
{
    int image_stride = h * w * c;
    int channel_stride = h * w;
    for(size_t idx = 0; idx < n; idx++)
    {
        unsigned char * input_image = input_chw + idx * image_stride;
        unsigned char * plane_R = input_image;
        unsigned char * plane_G = input_image + channel_stride;
        unsigned char * plane_B = input_image + channel_stride;

        unsigned char * output_image = output_hwc;
        for(size_t i = 0; i < h; i++)
        {
            for(size_t j = 0; j < w; j++)
            {
                *output_image++ = *plane_R;
                *output_image++ = *plane_G;
                *output_image++ = *plane_B;
                plane_R++;
                plane_G++;
                plane_B++;
            }
        }
        output_hwc += image_stride;
    }
}

int main(int argc, const char ** argv)
{
    // check command-line usage
    const int MIN_ARG_COUNT = 2;
    if(argc < MIN_ARG_COUNT) {
        std::cout <<  "Usage: rocAL_basic_test <image_dataset_folder> <test_case:0/1> <processing_device=1/cpu=0>  decode_width decode_height <gray_scale:0/rgb:1> decode_shard_counts \n";
        return -1;
    }
    int argIdx = 0;
    const char * folderPath1 = argv[++argIdx];
    int rgb = 1;// process color images
    int decode_width = 0;
    int decode_height = 0;
    int test_case = 0;
    bool processing_device = 0;
    size_t decode_shard_counts = 1;

    if(argc >= argIdx+MIN_ARG_COUNT)
        test_case = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        processing_device = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        decode_width = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        decode_height = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        rgb = atoi(argv[++argIdx]);

    if(argc >= argIdx+MIN_ARG_COUNT)
        decode_shard_counts = atoi(argv[++argIdx]);


    int inputBatchSize = 1;

    std::cout << ">>> Running on " << (processing_device?"GPU":"CPU") << std::endl;

    RocalImageColor color_format = (rgb != 0) ? RocalImageColor::ROCAL_COLOR_RGB24 : RocalImageColor::ROCAL_COLOR_U8;

    auto handle = rocalCreate(inputBatchSize, processing_device?RocalProcessMode::ROCAL_PROCESS_GPU:RocalProcessMode::ROCAL_PROCESS_CPU, 0,1);

    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Could not create the Rocal contex\n";
        return -1;
    }
    rocalSetSeed(0);

    /*>>>>>>>>>>>>>>>>>>> Graph description <<<<<<<<<<<<<<<<<<<*/
    RocalTensor input1;
    RocalMetaData metadata_output;
    RocalTensor image1, image2;

    RocalTensorLayout tensorLayout = RocalTensorLayout::ROCAL_NHWC;
    RocalTensorOutputType tensorOutputType = RocalTensorOutputType::ROCAL_UINT8;

#if CLASSIFICATION_TRAIN
    std::cout << ">>>>>>> Running CLASSIFICATION TRAIN" << std::endl;
    metadata_output = rocalCreateLabelReader(handle, folderPath1);
    input1 = rocalFusedJpegCropSingleShard(handle, folderPath1,  color_format, 0, 1, false, false, false,
                                ROCAL_USE_USER_GIVEN_SIZE, 2000, 2000); 
#elif CLASSIFICATION_VAL
    std::cout << ">>>>>>> Running CLASSIFICATION VAL" << std::endl;
    metadata_output = rocalCreateLabelReader(handle, folderPath1);
    input1 = rocalJpegFileSourceSingleShard(handle, folderPath1,  color_format, 0, 1, false, false, false);
#elif SSD
    char const *json_path = "/media/coco_10_img/coco2017/annotations/instances_train2017.json";
#if defined RANDOMBBOXCROP
    bool all_boxes_overlap = true;
    bool no_crop = false;
#endif
    if (strcmp(json_path, "") == 0)
    {
        std::cout << "\n json_path has to be set in rocal_unit test manually";
        exit(0);
    }
    rocalCreateCOCOReader(handle, json_path, true, false);
#if defined RANDOMBBOXCROP
    RocalFloatParam aspect_ratio = rocalCreateFloatUniformRand(0.5, 2.0);
    RocalFloatParam scaling = rocalCreateFloatUniformRand(0.3, 1.0);

    // rocalRandomBBoxCrop(handle, all_boxes_overlap, no_crop);
    rocalRandomBBoxCrop(handle, all_boxes_overlap, no_crop, aspect_ratio, false, 0, 0, 50, scaling);
#endif
    input1 = rocalJpegCOCOFileSourcePartialSingleShard(handle, folderPath1, json_path, color_format, 0, 1, false, false, false);
#elif MASK
    char const *json_path = "/media/coco_10_img/coco2017/annotations/instances_train2017.json";
    if (strcmp(json_path, "") == 0)
    {
        std::cout << "\n json_path has to be set in rocal_unit test manually";
        exit(0);
    }
    rocalCreateCOCOReader(handle, json_path, true, true);
    input1 = rocalJpegCOCOFileSourcePartialSingleShard(handle, folderPath1, json_path, color_format, 0, 1, false, false, false);
#else
    std::cout << ">>>>>>> Running IMAGE READER" << std::endl;
    metadata_output = rocalCreateLabelReader(handle, folderPath1);
    // The jpeg file loader can automatically select the best size to decode all images to that size
    // User can alternatively set the size or change the policy that is used to automatically find the size
    if(decode_height <= 0 || decode_width <= 0)
        input1 = rocalJpegFileSource(handle, folderPath1,  color_format, decode_shard_counts, false, false);
    else
        input1 = rocalJpegFileSource(handle, folderPath1,  color_format, decode_shard_counts, false, false, false,
                                    ROCAL_USE_USER_GIVEN_SIZE, decode_width, decode_height);
#endif
    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "JPEG source could not initialize : "<<rocalGetErrorMessage(handle) << std::endl;
        return -1;
    }
#if CLASSIFICATION_TRAIN
    std::vector<float> mean{0.485 * 255, 0.456 * 255, 0.406 * 255};
    std::vector<float> sdev{0.229 * 255, 0.224 * 255, 0.225 * 255};
    std::vector<int> values = {0, 1};
    std::vector<double> frequencies = {0.5, 0.5};
    RocalIntParam mirror = rocalCreateIntRand(values.data(), frequencies.data(), values.size());

    image1 = rocalResize(handle, input1, tensorLayout, tensorOutputType, 3, 224 , 224, 0, false);
    image2 = rocalCropMirrorNormalize(handle, image1, tensorLayout, RocalTensorOutputType::ROCAL_FP32, 3, 224, 224, 0, 0, 0, mean, sdev, true, mirror);
#elif CLASSIFICATION_VAL
    std::vector<float> mean{0.485 * 255, 0.456 * 255, 0.406 * 255};
    std::vector<float> sdev{0.229 * 255, 0.224 * 255, 0.225 * 255};
    image1 = rocalResizeShorter(handle, input1, tensorLayout, tensorOutputType, 256, true);
    image2 = rocalCropCenterFixed(handle, image1, tensorLayout, tensorOutputType, 224, 224, 3, false);
    image1 = rocalCropMirrorNormalize(handle, image2, tensorLayout, RocalTensorOutputType::ROCAL_FP32, 3, 224, 224, 0, 0, 0, mean, sdev, true);
#elif SSD
    std::vector<float> mean{0, 0, 0};
    std::vector<float> sdev{1, 1, 1};
    RocalFloatParam saturation = rocalCreateFloatUniformRand(0.5, 1.5);
    RocalFloatParam contrast = rocalCreateFloatUniformRand(0.5, 1.5);
    RocalFloatParam brightness = rocalCreateFloatUniformRand(0.875, 1.125);
    RocalFloatParam hue = rocalCreateFloatUniformRand(0.5, -0.5);
    std::vector<int> values = {0, 1};
    std::vector<double> frequencies = {0.5, 0.5};
    RocalIntParam mirror = rocalCreateIntRand(values.data(), frequencies.data(), values.size());
    
    image1 = rocalResize(handle, input1, tensorLayout, tensorOutputType, 3, 224 , 224, 0, false);
    image2 = rocalColorTwist(handle, image1, tensorLayout, tensorOutputType, false, brightness, contrast, saturation, hue);
    auto image3 = rocalCropMirrorNormalize(handle, image2, tensorLayout, RocalTensorOutputType::ROCAL_FP32, 3, 224, 224, 0, 0, 0, mean, sdev, true, mirror);
#elif MASK
    std::vector<float> mean{102.9801, 115.9465, 122.7717};
    std::vector<float> sdev{1.0f, 1.0f, 1.0f};
    std::vector<int> values = {0, 1};
    std::vector<double> frequencies = {0.5, 0.5};
    RocalIntParam mirror = rocalCreateIntRand(values.data(), frequencies.data(), values.size());
    image1 = rocalResizeMirrorNormalize(handle, input1, tensorLayout, RocalTensorOutputType::ROCAL_UINT8, 3, 1344, 1344, 1, mean, sdev, true, mirror);
#else
    image1 = rocalBrightness(handle, input1, true);
#endif

    rocalVerify(handle);
    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Error while adding the augmentation nodes " << std::endl;
        auto err_msg = rocalGetErrorMessage(handle);
        std::cout << err_msg << std::endl;
    }

    /*>>>>>>>>>>>>>>>>>>> Diplay using OpenCV <<<<<<<<<<<<<<<<<*/

    cv::Mat mat_color;
    int col_counter = 0;
    char * outName = "out_";
    //cv::namedWindow("output", CV_WINDOW_AUTOSIZE);
    std::cerr << "Going to process images\n";
    std::cerr << "Remaining images" << rocalGetRemainingImages(handle) << "\n";
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    int index = 0;

    RocalTensorList output_tensor_list;
    auto cv_color_format = ((color_format == RocalImageColor::ROCAL_COLOR_RGB24) ?  CV_8UC3 : CV_8UC1);

    while (rocalGetRemainingImages(handle) >= inputBatchSize)
    {
        index++;
        if (rocalRun(handle) != 0)
            break;
#if CLASSIFICATION_TRAIN || CLASSIFICATION_VAL
        RocalTensorList labels = rocalGetImageLabels(handle);

        for(int i = 0; i < labels->size(); i++)
        {
            int * labels_buffer = (int *)(labels->at(i)->buffer());
            std::cerr << ">>>>> LABELS : " << labels_buffer[0] << "\t";
        }
#elif SSD
        RocalTensorList bbox_labels = rocalGetBoundingBoxLabel(handle);
        RocalTensorList bbox_coords = rocalGetBoundingBoxCords(handle);
        for(int i = 0; i < bbox_labels->size(); i++)
        {
            int * labels_buffer = (int *)(bbox_labels->at(i)->buffer());
            float *bbox_buffer = (float *)(bbox_coords->at(i)->buffer());
            std::cerr << "\n>>>>> BBOX LABELS : ";
            for(int j = 0; j < bbox_labels->at(i)->info().dims().at(0); j++)
                std::cerr << labels_buffer[j] << " ";
            std::cerr << "\n>>>>> BBOXX : " <<bbox_coords->at(i)->info().dims().at(0) << " : \n";
            for(int j = 0, j4 = 0; j < bbox_coords->at(i)->info().dims().at(0); j++, j4 = j * 4)
                std::cerr << bbox_buffer[j4] << " " << bbox_buffer[j4 + 1] << " " << bbox_buffer[j4 + 2] << " " << bbox_buffer[j4 + 3] << "\n";

        }
#elif MASK
        RocalTensorList bbox_labels = rocalGetBoundingBoxLabel(handle);
        RocalTensorList bbox_coords = rocalGetBoundingBoxCords(handle);
        std::vector<int> mask_count;
        std::vector<int> polygon_size;
        unsigned total_number_of_objects_per_batch = rocalGetBoundingBoxCount(handle);
        mask_count.resize(total_number_of_objects_per_batch);
        int mask_size = rocalGetMaskCount(handle, mask_count.data());
        polygon_size.resize(mask_size);
        RocalTensorList mask_data = rocalGetMaskCoordinates(handle, polygon_size.data());
        
        for(int i = 0; i < bbox_labels->size(); i++)
        {
            int * labels_buffer = (int *)(bbox_labels->at(i)->buffer());
            float *bbox_buffer = (float *)(bbox_coords->at(i)->buffer());
            float *mask_buffer = (float *)(mask_data->at(i)->buffer());
            std::cerr << "\n>>>>> BBOX LABELS : ";
            for(int j = 0; j < bbox_labels->at(i)->info().dims().at(0); j++)
                std::cerr << labels_buffer[j] << " ";
            std::cerr << "\n>>>>> BBOXX : " <<bbox_coords->at(i)->info().dims().at(0) << " : \n";
            for(int j = 0, j4 = 0; j < bbox_coords->at(i)->info().dims().at(0); j++, j4 = j * 4)
                std::cerr << bbox_buffer[j4] << " " << bbox_buffer[j4 + 1] << " " << bbox_buffer[j4 + 2] << " " << bbox_buffer[j4 + 3] << "\n";
            std::cerr << "\n>>>>>>> MASK COORDS : ";
            int poly_cnt = 0;
            for(unsigned j = 0; j < total_number_of_objects_per_batch; j++)
            {
                std::cerr << "Mask idx : " << j << "Polygons : " <<  mask_count[j] << "[" ;
                for(int k = 0; k < mask_count[j]; k++)
                {
                    std::cerr << "[";
                    for(int l = 0; l < polygon_size[poly_cnt]; l++)
                        std::cerr << mask_buffer[l] << ", ";
                    std::cerr << "]";
                    mask_buffer += polygon_size[poly_cnt++];
                }
                std::cerr << "]\n";
            }
        }
#else
        RocalTensorList labels = rocalGetImageLabels(handle);

        for(int i = 0; i < labels->size(); i++)
        {
            int * labels_buffer = (int *)(labels->at(i)->buffer());
            std::cerr << ">>>>> LABELS : " << labels_buffer[0] << "\t";
        }
#endif

        output_tensor_list = rocalGetOutputTensors(handle);
        std::vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);  
#if DISPLAY
        cv::Mat mat_input;
        cv::Mat mat_output;
        for(int idx = 0; idx < output_tensor_list->size(); idx++)
        {
            int h = output_tensor_list->at(idx)->info().max_shape().at(1) * output_tensor_list->at(idx)->info().dims().at(0);
            int w = output_tensor_list->at(idx)->info().max_shape().at(0);
            mat_input = cv::Mat(h, w, cv_color_format);
            mat_output = cv::Mat(h, w, cv_color_format);

            unsigned char *out_buffer;
            if(output_tensor_list->at(idx)->info().data_type() == RocalTensorDataType::FP32)
            {
                float * out_f_buffer;
                if(output_tensor_list->at(idx)->info().mem_type() == RocalMemType::HIP)
                {
                    out_f_buffer = (float *)malloc(output_tensor_list->at(idx)->info().data_size());
                    output_tensor_list->at(idx)->copy_data(out_f_buffer, false);
                }
                else if(output_tensor_list->at(idx)->info().mem_type() == RocalMemType::HOST)
                    out_f_buffer = (float *)output_tensor_list->at(idx)->buffer();

                out_buffer = (unsigned char *)malloc(output_tensor_list->at(idx)->info().data_size() / 4);
                convert_float_to_uchar_buffer(out_f_buffer, out_buffer, output_tensor_list->at(idx)->info().data_size() / 4);
            }
            else
            {
                if(output_tensor_list->at(idx)->info().mem_type() == RocalMemType::HIP)
                {
                    out_buffer = (unsigned char *)malloc(output_tensor_list->at(idx)->info().data_size());
                    output_tensor_list->at(idx)->copy_data(out_buffer, false);
                }
                else if(output_tensor_list->at(idx)->info().mem_type() == RocalMemType::HOST)
                    out_buffer = (unsigned char *)(output_tensor_list->at(idx)->buffer());
            }

            if(output_tensor_list->at(idx)->info().layout() == RocalTensorlayout::NCHW)
            {
                // cv::Mat mat_input_nchw = cv::Mat(cv_color_format, h, w);
                // mat_input_nchw = (unsigned char *)out_buffer;
                // cv::transposeND(mat_input_nchw, {0, 3, 1, 2}, mat_input); // Can be enabled only with OpenCV 4.6.0
                convert_nchw_to_nhwc(out_buffer, mat_input.data, output_tensor_list->at(idx)->info().dims().at(0), output_tensor_list->at(idx)->info().dims().at(2),
                                     output_tensor_list->at(idx)->info().dims().at(3), output_tensor_list->at(idx)->info().dims().at(1));            
            }
            else
                mat_input.data = (unsigned char *)out_buffer;

            mat_input.copyTo(mat_output(cv::Rect(0, 0, w, h)));

            std::string out_filename = std::string(outName) + ".png";   // in case the user specifies non png filename
            out_filename = std::string(outName) + std::to_string(index) + std::to_string(idx) + ".png";   // in case the user specifies non png filename

            if (color_format == RocalImageColor::ROCAL_COLOR_RGB24)
            {
                cv::cvtColor(mat_output, mat_color, CV_RGB2BGR);
                cv::imwrite(out_filename, mat_color, compression_params);
            }
            else
            {
                cv::imwrite(out_filename, mat_output, compression_params);
            }
        }
        mat_input.release();
        mat_output.release();
#endif
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto dur = duration_cast<microseconds>(t2 - t1).count();
    auto rocal_timing = rocalGetTimingInfo(handle);
    std::cout << "\nLoad     time " << rocal_timing.load_time << std::endl;
    std::cout << "Decode   time " << rocal_timing.decode_time << std::endl;
    std::cout << "Process  time " << rocal_timing.process_time << std::endl;
    std::cout << "Transfer time " << rocal_timing.transfer_time << std::endl;
    std::cout << ">>>>> Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us " << std::endl;
    rocalRelease(handle);
    return 0;
}
