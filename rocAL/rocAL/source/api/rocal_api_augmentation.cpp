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


#include "node_gamma.h"
#include "node_brightness.h"
#include "node_resize.h"
#include "node_resize_shorter.h"
#include "node_color_twist.h"
#include "node_crop_mirror_normalize.h"
#include "node_crop.h"
#include "node_copy.h"
#include "node_nop.h"
#include "node_sequence_rearrange.h"
#include "meta_node_crop_mirror_normalize.h"
#include "node_resize.h"
#include "node_crop.h"
#include "node_to_decibles.h"
#include "node_preemphasis_filter.h"
#include "node_spectrogram.h"
#include "node_non_silent_region.h"
#include "node_mel_filter_bank.h"
#include "node_tensor_mul_scalar.h"
#include "node_tensor_add_tensor.h"
#include "node_slice.h"
#include "node_pad.h"
#include "node_normalize.h"
#include "node_normal_distribution.h"
#include "node_uniform_distribution.h"
#include "node_resample.h"

#include "meta_node_resize.h"
#include "meta_node_crop.h"
#include "meta_node_ssd_random_crop.h"
#include "node_resize_mirror_normalize.h"


#include "commons.h"
#include "context.h"
#include "rocal_api.h"
#define MAX_ASPECT_RATIO 3.0f

void get_rocal_tensor_layout(RocalTensorLayout &tensor_layout, RocalTensorlayout &op_tensor_layout, int &layout)
{
    switch(tensor_layout)
    {
        case 0:
            op_tensor_layout = RocalTensorlayout::NHWC;
            layout = 0;
            return;
        case 1:
            op_tensor_layout = RocalTensorlayout::NCHW;
            layout = 1;
            return;
        default:
            THROW("Unsupported Tensor layout" + TOSTR(tensor_layout))
    }
}

void get_rocal_tensor_data_type(RocalTensorOutputType &tensor_output_type, RocalTensorDataType &tensor_data_type)
{
    switch(tensor_output_type)
    {
        case ROCAL_FP32:
            tensor_data_type = RocalTensorDataType::FP32;
            return;
        case ROCAL_FP16:
            tensor_data_type = RocalTensorDataType::FP16;
            return;
        case ROCAL_UINT8:
            tensor_data_type = RocalTensorDataType::UINT8;
            return;
        default:
            THROW("Unsupported Tensor output type" + TOSTR(tensor_output_type))
    }
}


RocalTensor  ROCAL_API_CALL
rocalSequenceRearrange(
            RocalContext p_context,
            RocalTensor input,
            unsigned int* new_order,
            unsigned int  new_sequence_length,
            unsigned int sequence_length,
            bool is_output )
{
    rocalTensor* output = nullptr;

    if ((p_context == nullptr) || (input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    try
    {

        if(sequence_length == 0)
            THROW("sequence_length passed should be bigger than 0")

        rocalTensorInfo output_info = input->info();
        std::vector<size_t> new_dims;
        new_dims = output_info.dims();
        new_dims[1] = new_sequence_length;
        output_info.set_dims(new_dims);

        output = context->master_graph->create_tensor(output_info, is_output);
        std::shared_ptr<SequenceRearrangeNode> sequence_rearrange_node =  context->master_graph->add_node<SequenceRearrangeNode>({input}, {output});
        sequence_rearrange_node->init(new_order, new_sequence_length, sequence_length, context->user_batch_size());
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalBrightness(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalFloatParam p_alpha,
        RocalFloatParam p_beta)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    auto beta = static_cast<FloatParam*>(p_beta);
    try
    {
        output = context->master_graph->create_tensor(input->info(), is_output);
        context->master_graph->add_node<BrightnessNode>({input}, {output})->init(alpha, beta);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor  ROCAL_API_CALL
rocalCopyTensor(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    try
    {
        output = context->master_graph->create_tensor(input->info(), is_output);
        context->master_graph->add_node<CopyNode>({input}, {output});
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

// Commented for Now
// RocalTensor  ROCAL_API_CALL
// rocalNopTensor(
//         RocalContext p_context,
//         RocalTensor p_input,
//         bool is_output)
// {
//     if(!p_context || !p_input)
//         THROW("Null values passed as input")
//     rocalTensor* output = nullptr;
//     auto context = static_cast<Context*>(p_context);
//     auto input = static_cast<rocalTensor*>(p_input);
//     try
//     {
//         output = context->master_graph->create_tensor(input->info(), is_output);
//         context->master_graph->add_tensor_node<NopNode>({input}, {output});
//     }
//     catch(const std::exception& e)
//     {
//         context->capture_error(e.what());
//         ERR(e.what())
//     }
//     return output;
// }

RocalTensor ROCAL_API_CALL
rocalCrop(RocalContext p_context,
          RocalTensor p_input,
          RocalTensorLayout rocal_tensor_layout,
          RocalTensorOutputType rocal_tensor_output_type,
          bool is_output,
          RocalFloatParam p_crop_width,
          RocalFloatParam p_crop_height,
          RocalFloatParam p_crop_depth,
          RocalFloatParam p_crop_pox_x,
          RocalFloatParam p_crop_pos_y,
          RocalFloatParam p_crop_pos_z)
{
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    auto crop_h = static_cast<FloatParam*>(p_crop_height);
    auto crop_w = static_cast<FloatParam*>(p_crop_width);
    auto x_drift = static_cast<FloatParam*>(p_crop_pox_x);
    auto y_drift = static_cast<FloatParam*>(p_crop_pos_y);

    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        if(!input || !context)
            THROW("Null values passed as input")
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocalTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        // std::vector<size_t> out_dims = output_info.dims();
        // if(op_tensorLayout == RocalTensorlayout::NHWC)
        // {
        //     out_dims[1] = crop_h;
        //     out_dims[2] = crop_w;
        // }
        // else if(op_tensorLayout == RocalTensorlayout::NCHW)
        // {
        //     out_dims[2] = crop_h;
        //     out_dims[3] = crop_w;
        // }
        // output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::shared_ptr<CropNode> crop_node = context->master_graph->add_node<CropNode>({input}, {output});
        crop_node->init(crop_h, crop_w, x_drift, y_drift, layout);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropMetaNode,CropNode>(crop_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }
    return output; // Changed to input----------------IMPORTANT
}

//resizemirrornormalize
RocalTensor
ROCAL_API_CALL rocalResizeMirrorNormalize(RocalContext p_context,
                                          RocalTensor p_input,
                                          RocalTensorLayout rocal_tensor_layout,
                                          RocalTensorOutputType rocal_tensor_output_type,
                                          unsigned resize_depth,
                                          unsigned resize_height,
                                          unsigned resize_width,
                                          int interpolation_type,
                                          std::vector<float> &mean,
                                          std::vector<float> &std_dev,
                                          bool is_output,
                                          RocalIntParam p_mirror)
{
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    auto mirror = static_cast<IntParam *>(p_mirror);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        if(!input || !context || resize_width == 0 || resize_height == 0)
            THROW("Null values passed as input")
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocalTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensorLayout == RocalTensorlayout::NHWC)
        {
            out_dims[1] = resize_height;
            out_dims[2] = resize_width;
            out_dims[3] = 3;

        }
        else if(op_tensorLayout == RocalTensorlayout::NCHW)
        {
            out_dims[1] = 3;
            out_dims[2] = resize_height;
            out_dims[3] = resize_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<ResizeMirrorNormalizeNode>({input}, {output})->init( interpolation_type, mean,std_dev , mirror, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }

    return output; // Changed to input----------------IMPORTANT
}


RocalTensor  ROCAL_API_CALL
rocalCropFixed(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        unsigned crop_width,
        unsigned crop_height,
        unsigned crop_depth,
        bool is_output,
        float crop_pos_x,
        float crop_pos_y,
        float crop_pos_z)
{
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        if(crop_width == 0 || crop_height == 0 || crop_depth == 0)
            THROW("Crop node needs tp receive non-zero destination dimensions")
        // For the crop node, user can create an image with a different width and height
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocalTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensorLayout == RocalTensorlayout::NHWC)
        {
            out_dims[1] = crop_height;
            out_dims[2] = crop_width;
        }
        else if(op_tensorLayout == RocalTensorlayout::NCHW)
        {
            out_dims[2] = crop_height;
            out_dims[3] = crop_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::shared_ptr<CropNode> crop_node =  context->master_graph->add_node<CropNode>({input}, {output});
        crop_node->init(crop_height, crop_width, crop_pos_x, crop_pos_y, layout);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropMetaNode,CropNode>(crop_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor  ROCAL_API_CALL
rocalCropCenterFixed(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        unsigned crop_width,
        unsigned crop_height,
        unsigned crop_depth,
        bool is_output)
{
    rocalTensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        if(crop_width == 0 || crop_height == 0 || crop_depth == 0)
            THROW("Crop node needs tp receive non-zero destination dimensions")
        // For the crop node, user can create an image with a different width and height
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocalTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensorLayout == RocalTensorlayout::NHWC)
        {
            out_dims[1] = crop_height;
            out_dims[2] = crop_width;
        }
        else if(op_tensorLayout == RocalTensorlayout::NCHW)
        {
            out_dims[2] = crop_height;
            out_dims[3] = crop_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::shared_ptr<CropNode> crop_node = context->master_graph->add_node<CropNode>({input}, {output});
        crop_node->init(crop_height, crop_width, layout);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropMetaNode,CropNode>(crop_node);
    }

    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}



RocalTensor
ROCAL_API_CALL rocalCropMirrorNormalize(RocalContext p_context, RocalTensor p_input, RocalTensorLayout rocal_tensor_layout,
                                    RocalTensorOutputType rocal_tensor_output_type, unsigned crop_depth, unsigned crop_height,
                                    unsigned crop_width, float start_x, float start_y, float start_z, std::vector<float> &mean,
                                    std::vector<float> &std_dev, bool is_output, RocalIntParam p_mirror)
{
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    auto mirror = static_cast<IntParam *>(p_mirror);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        if(!input || !context || crop_width == 0 || crop_height == 0)
            THROW("Null values passed as input")
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocalTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensorLayout == RocalTensorlayout::NHWC)
        {
            out_dims[1] = crop_height;
            out_dims[2] = crop_width;
            out_dims[3] = 3;
        }
        else if(op_tensorLayout == RocalTensorlayout::NCHW)
        {
            out_dims[1] = 3;
            out_dims[2] = crop_height;
            out_dims[3] = crop_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::shared_ptr<CropMirrorNormalizeNode> cmn_node = context->master_graph->add_node<CropMirrorNormalizeNode>({input}, {output});
        cmn_node->init(crop_height, crop_width, start_x, start_y, mean, std_dev , mirror,layout );
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropMirrorNormalizeMetaNode,CropMirrorNormalizeNode>(cmn_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }
    return output; // Changed to input----------------IMPORTANT
}


RocalTensor ROCAL_API_CALL
rocalResize(RocalContext p_context,
            RocalTensor p_input,
            RocalTensorLayout rocal_tensor_layout,
            RocalTensorOutputType rocal_tensor_output_type,
            unsigned dest_width,
            unsigned dest_height,
            bool is_output,
            RocalResizeScalingMode scaling_mode,
            std::vector<unsigned> max_size,
            unsigned resize_shorter,
            unsigned resize_longer,
            RocalResizeInterpolationType interpolation_type)
{
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        if((dest_width | dest_height | resize_longer | resize_shorter) == 0)
            THROW("Atleast one size 'dest_width' or 'dest_height' or 'resize_shorter' or 'resize_longer' must be specified")
        if((dest_width | dest_height) && (resize_longer | resize_shorter))
            THROW("Only one method of specifying size can be used \ndest_width and/or dest_height\nresize_shorter\nresize_longer")
        if(resize_longer && resize_shorter)
            THROW("'resize_longer' and 'resize_shorter' cannot be passed together. They are mutually exclusive.")

        unsigned out_width, out_height;
        RocalResizeScalingMode resize_scaling_mode;

        // Change the scaling mode if resize_shorter or resize_longer is specified
        if(resize_shorter) {
            resize_scaling_mode = RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_SMALLER;
            out_width = out_height = resize_shorter;
        } else if(resize_longer) {
            resize_scaling_mode = RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_LARGER;
            out_width = out_height = resize_longer;
        } else {
            resize_scaling_mode = scaling_mode;
            out_width = dest_width;
            out_height = dest_height;
        }

        std::vector<unsigned> maximum_size;
        if (max_size.size()) {
            if(max_size.size() == 1) {
                maximum_size = {max_size[0], max_size[0]};
            } else if(max_size.size() == 2) {
                maximum_size = {max_size[0], max_size[1]}; // {width, height}
            } else {
                THROW("The length of max_size vector exceeds the image dimension.")
            }
        }

        // Determine the max width and height to be set to the output info
        unsigned max_out_width, max_out_height;
        if (maximum_size.size() && maximum_size[0] != 0 && maximum_size[1] != 0) {
            // If max_size is passed by the user, the resized images cannot exceed the max size,
            max_out_width = maximum_size[0];
            max_out_height = maximum_size[1];
        } else {
            // compute the output info width and height wrt the scaling modes and roi passed
            if(resize_scaling_mode == ROCAL_SCALING_MODE_STRETCH) {
                max_out_width = out_width ? out_width : input->info().max_shape()[0];
                max_out_height = out_height ? out_height : input->info().max_shape()[1];
            } else if(resize_scaling_mode == ROCAL_SCALING_MODE_NOT_SMALLER) {
                max_out_width = (out_width ? out_width : out_height) * MAX_ASPECT_RATIO;
                max_out_height = (out_height ? out_height : out_width) * MAX_ASPECT_RATIO;
            } else {
                max_out_width = out_width ? out_width : out_height * MAX_ASPECT_RATIO;
                max_out_height = out_height ? out_height : out_width * MAX_ASPECT_RATIO;
            }
            if(maximum_size.size() == 2) {
                max_out_width = maximum_size[0] ? maximum_size[0] : max_out_width;
                max_out_height = maximum_size[1] ? maximum_size[1] : max_out_height;
            }
        }

        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocalTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensorLayout == RocalTensorlayout::NHWC)
        {
            out_dims[1] = max_out_height;
            out_dims[2] = max_out_width;
        }
        else if(op_tensorLayout == RocalTensorlayout::NCHW)
        {
            out_dims[2] = max_out_height;
            out_dims[3] = max_out_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::shared_ptr<ResizeNode> resize_node =  context->master_graph->add_node<ResizeNode>({input}, {output});
        resize_node->init(out_width, out_height, resize_scaling_mode, maximum_size, interpolation_type, layout);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<ResizeMetaNode,ResizeNode>(resize_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }
    return output; // Changed to input----------------IMPORTANT
}

RocalTensor  ROCAL_API_CALL
rocalResizeShorter(
        RocalContext p_context,
        RocalTensor p_input,
        RocalTensorLayout rocal_tensor_layout,
        RocalTensorOutputType rocal_tensor_output_type,
        unsigned size,
        bool is_output)
{
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        if(!input || !context)
            THROW("Null values passed as input")
        // For the resize node, user can create an image with a different width and height
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocalTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        if (size == 0) size = input->info().max_shape()[0];
        if (size == 0) size = input->info().max_shape()[1];
        std::vector<size_t> out_dims = output_info.dims();
        int size_dim = size * 10;
        if(op_tensorLayout == RocalTensorlayout::NHWC)
        {
            out_dims[1] = size_dim;
            out_dims[2] = size_dim;
        }
        else if(op_tensorLayout == RocalTensorlayout::NCHW)
        {
            out_dims[2] = size_dim;
            out_dims[3] = size_dim;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::shared_ptr<ResizeShorterNode> resize_node =  context->master_graph->add_node<ResizeShorterNode>({input}, {output});
        resize_node->init(size);
        // if (context->master_graph->meta_data_graph())
        //     context->master_graph->meta_add_node<ResizeMetaNode,ResizeShorterNode>(resize_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


RocalTensor ROCAL_API_CALL
rocalColorTwist(RocalContext p_context,
                RocalTensor p_input,
                RocalTensorLayout rocal_tensor_layout,
                RocalTensorOutputType rocal_tensor_output_type,
                bool is_output,
                RocalFloatParam p_alpha,
                RocalFloatParam p_beta,
                RocalFloatParam p_hue,
                RocalFloatParam p_sat)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    auto beta = static_cast<FloatParam*>(p_beta);
    auto hue = static_cast<FloatParam*>(p_hue);
    auto sat = static_cast<FloatParam*>(p_sat);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocalTensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ColorTwistNode>({input}, {output})->init(alpha, beta, hue ,sat, layout);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalNormalize(RocalContext p_context,
               RocalTensor p_input,
               RocalTensorOutputType rocal_tensor_output_type,
               bool is_output, bool batch,
               std::vector<int>axes,
               float mean, float std_dev,
               float scale, float shift,
               int ddof, float epsilon) {
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    try {
        if(mean > 0.0f && std_dev > 0.0f && axes.size())
            THROW("Axes must not be passed when both mean and standard deviation are specified")

        rocalTensorInfo output_info = input->info();
        RocalTensorDataType op_tensorDataType;
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        output_info.set_data_type(op_tensorDataType);
        

        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<NormalizeNode>({input}, {output})->init(mean, std_dev, axes, batch, scale, shift, ddof, epsilon);
    }
    catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalPad(RocalContext p_context,
        RocalTensor p_input,
        RocalTensorOutputType rocal_tensor_output_type,
        bool is_output,
        float fill_value,
        std::vector<int>axes,
        std::vector<int>align) {

    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    try {
        rocalTensorInfo output_info = input->info();
        RocalTensorDataType op_tensorDataType;
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<PadNode>({input}, {output})->init(fill_value);

    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalToDecibels(RocalContext p_context,
                RocalTensor p_input,
                RocalTensorLayout rocal_tensor_layout,
                RocalTensorOutputType rocal_tensor_output_type,
                bool is_output,
                float cut_off_DB,
                float multiplier,
                float magnitude_reference)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocalTensorInfo output_info = input->info();
        output_info.set_tensor_layout(RocalTensorlayout::NONE);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<ToDeciblesNode>({input}, {output})->init(cut_off_DB, multiplier, magnitude_reference);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalPreEmphasisFilter(RocalContext p_context,
                RocalTensor p_input,
                RocalTensorOutputType rocal_tensor_output_type,
                bool is_output,
                RocalFloatParam p_preemph_coeff,
                RocalAudioBorderType preemph_border_type)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    auto preemph_coeff = static_cast<FloatParam*>(p_preemph_coeff);
    RocalTensorDataType op_tensorDataType;
    try
    {
        int layout=0;
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocalTensorInfo output_info = input->info();
        output_info.set_tensor_layout(RocalTensorlayout::NONE);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<PreemphasisFilterNode>({input}, {output})->init(preemph_coeff, preemph_border_type);

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalSpectrogram(RocalContext p_context,
                RocalTensor p_input,
                RocalTensorOutputType rocal_tensor_output_type,
                bool is_output,
                std::vector<float> & window_fn,
                bool center_windows,
                bool reflect_padding,
                RocalSpectrogramLayout spec_layout,
                int power,
                int nfft_size,
                int window_length,
                int window_step) {
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    RocalTensorDataType op_tensorDataType;
    try {
        int layout=0;
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocalTensorInfo output_info = input->info();
        std::vector<size_t> max_dims = output_info.max_shape();
        int window_offset = 0;
        if(!center_windows)
            window_offset = window_length;
        int max_frame = (((max_dims[0] - window_offset) / window_step) + 1);
        max_frame = std::max(0, max_frame);
        int bins = std::max(0, (nfft_size / 2) + 1);
        std::vector<size_t> dims = output_info.dims();
        if(spec_layout == RocalSpectrogramLayout::FT) {
            dims[1] = max_frame;
            dims[2] = bins;
        } else {
            dims[1] = bins;
            dims[2] = max_frame;
        }
        output_info.set_dims(dims);
        output_info.set_tensor_layout(RocalTensorlayout::NONE);
        output_info.set_data_type(op_tensorDataType);

        if(power != 1 || power != 2) {
            WRN("rocalSpectrogram Power value can be 1 or 2 setting it to default 2")
            power = 2;
        }

        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<SpectrogramNode>({input}, {output})->init(center_windows, reflect_padding, spec_layout,
                                                                                  power, nfft_size, window_length,
                                                                                  window_step, window_fn);

    }
    catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

std::pair<RocalTensor,RocalTensor> ROCAL_API_CALL
rocalNonSilentRegion(RocalContext p_context,
                     RocalTensor p_input,
                     bool is_output,
                     float cut_off_db,
                     float reference_power,
                     int reset_interval,
                     int window_length) {
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocalTensor* output1 = nullptr;
    rocalTensor* output2 = nullptr;
    rocalTensorList output_tensors;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    try {
        RocalTensorDataType tensor_data_type = RocalTensorDataType::INT32;
        unsigned num_of_dims = 4;
        std::vector<size_t> dims;
        dims.resize(num_of_dims);
        dims.at(0) = context->user_batch_size();
        dims.at(1) = 1;
        dims.at(2) = 1;
        dims.at(3) = 1;
        auto info  = rocalTensorInfo(std::vector<size_t>(std::move(dims)),
                                context->master_graph->mem_type(),
                                tensor_data_type);
        info.set_tensor_layout(RocalTensorlayout::NONE);
        std::vector<size_t> dims1;
        dims1.resize(num_of_dims);
        dims1.at(0) = context->user_batch_size();
        dims1.at(1) = 1;
        dims1.at(2) = 1;
        dims1.at(3) = 1;
        auto info1  = rocalTensorInfo(std::vector<size_t>(std::move(dims1)),
                            context->master_graph->mem_type(),
                            tensor_data_type);
        info1.set_tensor_layout(RocalTensorlayout::NONE);
        output1 = context->master_graph->create_tensor(info, is_output);
        output2 = context->master_graph->create_tensor(info1, is_output);
        output_tensors.push_back(output1);
        output_tensors.push_back(output2);
        context->master_graph->add_node<NonSilentRegionNode>({input}, {output1, output2})->init(cut_off_db, reference_power, window_length, reset_interval);
    }
    catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return std::make_pair(output_tensors.at(0), output_tensors.at(1));
}

RocalTensor rocalMelFilterBank(RocalContext p_context,
                                RocalTensor p_input,
                                bool is_output,
                                float freq_high,
                                float freq_low,
                                RocalMelScaleFormula mel_formula,
                                int nfilter,
                                bool normalize,
                                float sample_rate) {
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    try {
        rocalTensorInfo output_info = input->info();
        std::vector<size_t> max_dims = output_info.max_shape();
        int max_frame = max_dims[0];
        max_frame = std::max(0, max_frame);
        std::vector<size_t> dims = output_info.dims();
        dims[1] = max_frame;
        dims[2] = nfilter;
        output_info.set_dims(dims);
        output_info.set_tensor_layout(RocalTensorlayout::NONE);
        output_info.set_data_type(RocalTensorDataType::FP32);

        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<MelFilterBankNode>({input}, {output})->init(freq_high, freq_low, mel_formula,
                                                                                  nfilter, normalize, sample_rate);
    }
    catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor rocalSlice(RocalContext p_context,
                        RocalTensor p_input,
                        RocalTensorOutputType rocal_tensor_output_type,
                        bool is_output,
                        RocalTensor anchor_tensor,
                        RocalTensor shape_tensor,
                        std::vector<float> fill_values,
                        std::vector<unsigned> axes,
                        bool normalized_anchor,
                        bool normalized_shape,
                        RocalOutOfBoundsPolicy policy) {

    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);

    RocalTensorDataType op_tensorDataType;
    try {

        rocalTensorInfo output_info = input->info();
        output_info.set_tensor_layout(RocalTensorlayout::NONE);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        output->reset_audio_sample_rate();
        context->master_graph->add_node<SliceNode>({input}, {output})->init(anchor_tensor, shape_tensor, fill_values,
                                                                            axes, normalized_anchor, normalized_shape, policy);
    }
    catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor rocalTensorMulScalar(RocalContext p_context,
                                RocalTensor p_input,
                                bool is_output,
                                RocalTensorLayout rocal_tensor_layout, // TODO : Swetha - not required for audio data - Check on this
                                RocalTensorOutputType rocal_tensor_output_type,
                                float scalar) {
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try {
        int layout=0;
        std::cerr << "Here in Op overl";
        get_rocal_tensor_layout(rocal_tensor_layout, op_tensorLayout, layout);
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocalTensorInfo output_info = input->info();
        output_info.set_tensor_layout(RocalTensorlayout::NONE);
        output = context->master_graph->create_tensor(output_info, is_output); // Create a rocALTensor object dynamically on heap 
        context->master_graph->add_node<TensorMulScalarNode>({input}, {output})->init(scalar); // Change this line of code
    }
    catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor rocalNormalDistribution(RocalContext p_context, 
                                RocalTensor p_input,
                                bool is_output,
                                float mean,
                                float stddev) {
    if(!p_context )
        THROW("Null values passed as input")
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    try {
        RocalTensorDataType tensor_data_type = RocalTensorDataType::FP32;
        unsigned num_of_dims = 3;
        std::vector<size_t> dims;
        dims.resize(num_of_dims);
        dims.at(0) = context->user_batch_size();
        dims.at(1) = 1;
        dims.at(2) = 1;
        auto info  = rocalTensorInfo(std::vector<size_t>(std::move(dims)),
                                context->master_graph->mem_type(),
                                tensor_data_type);
        info.set_tensor_layout(RocalTensorlayout::NONE);
        output = context->master_graph->create_tensor(info, is_output);
        output->create_from_handle(context->master_graph->get_vx_context());
        output->reset_tensor_roi();
        context->master_graph->add_node<NormalDistributionNode>({input}, {output})->init(mean, stddev); // Change this line of code - Check with Shobana

    }
    catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor rocalTensorAddTensor(RocalContext p_context,
                                RocalTensor p_input1,
                                RocalTensor p_input2,
                                bool is_output,
                                RocalTensorLayout rocal_tensor_layout, // TODO : Swetha - not required for audio data - Check on this
                                RocalTensorOutputType rocal_tensor_output_type) {
    if(!p_context || !p_input1 || !p_input2)
        THROW("Null values passed as input")
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input1 = static_cast<rocalTensor*>(p_input1);
    auto input2 = static_cast<rocalTensor*>(p_input2);
    RocalTensorlayout op_tensorLayout;
    RocalTensorDataType op_tensorDataType;
    try {
        int layout=0;
        std::cerr << "Here in Op over for addition!";
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        rocalTensorInfo output_info = input1->info();
        output_info.set_data_type(op_tensorDataType);
        output_info.set_tensor_layout(RocalTensorlayout::NONE);
        output = context->master_graph->create_tensor(output_info, is_output); // Assuming the bigger tensor is the first input
        context->master_graph->add_node<TensorAddTensorNode>({input1,input2}, {output})->init(); // Change this line of code
    }
    catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor rocalUniformDistribution(RocalContext p_context, 
                                RocalTensor p_input,
                                bool is_output,
                                std::vector<float> &range) {
    if(!p_context )
        THROW("Null values passed as input")
    rocalTensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    try {
        std::cerr << "Here in Uniform Dist";
        RocalTensorDataType tensor_data_type = RocalTensorDataType::FP32;
        unsigned num_of_dims = 3;
        std::vector<size_t> dims;
        dims.resize(num_of_dims);
        dims.at(0) = context->user_batch_size();
        dims.at(1) = 1;
        dims.at(2) = 1;
        auto info  = rocalTensorInfo(std::vector<size_t>(std::move(dims)),
                                context->master_graph->mem_type(),
                                tensor_data_type);
        info.set_tensor_layout(RocalTensorlayout::NONE); // Change for generic data
        output = context->master_graph->create_tensor(info, is_output);
        output->create_from_handle(context->master_graph->get_vx_context());
        context->master_graph->add_node<UniformDistributionNode>({input}, {output})->init(range); // Change this line of code - Check with Shobana

    }
    catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalResample(RocalContext p_context,
                RocalTensor p_input,
                RocalTensor p_input_resample_rate,
                RocalTensorOutputType rocal_tensor_output_type,
                bool is_output,
                float sample_hint)
{
    if(!p_context || !p_input_resample_rate || !p_input)
        THROW("Null values passed as input")
    rocalTensor* resampled_output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<rocalTensor*>(p_input);
    auto input_resample_rate = static_cast<rocalTensor*>(p_input_resample_rate);
    RocalTensorDataType op_tensorDataType;

    try
    {
        std::cerr << "In Resample node :";
        int layout=0;
        rocalTensorInfo output_info = input->info();
        get_rocal_tensor_data_type(rocal_tensor_output_type, op_tensorDataType);
        output_info.set_tensor_layout(RocalTensorlayout::NONE);
        output_info.set_data_type(op_tensorDataType);
        if(sample_hint > 0) {
            std::vector<size_t> max_dims = output_info.max_shape();
            std::vector<size_t> dims = output_info.dims();
            dims[1] = std::ceil(sample_hint);
            dims[2] = max_dims[1];
            output_info.set_dims(dims);
        }
        resampled_output = context->master_graph->create_tensor(output_info, is_output);
        // resampled_output->reset_tensor_roi();
        context->master_graph->add_node<ResampleNode>({input}, {resampled_output})->init(input_resample_rate, 50.0);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return resampled_output;
}