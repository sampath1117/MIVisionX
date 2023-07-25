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



#include "augmentations_nodes.h"
#include "augmentations_meta_nodes.h"
#include "commons.h"
#include "context.h"
#include "rocal_api.h"
#include "image_source_evaluator.h"

RocalTensor  ROCAL_API_CALL
rocalSequenceRearrange(RocalContext p_context,
                       RocalTensor p_input,
                       std::vector<unsigned int>& new_order,
                       bool is_output) {
    Tensor* output = nullptr;
    auto input = static_cast<Tensor*>(p_input);
    if ((p_context == nullptr) || (input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    try {

        if(new_order.size() == 0)
            THROW("The new order for the sequence passed should be greater than 0")
        TensorInfo output_info = input->info();
        std::vector<size_t> new_dims;
        new_dims = output_info.dims();
        new_dims[1] = new_order.size();
        output_info.set_dims(new_dims);

        output = context->master_graph->create_tensor(output_info, is_output);
        std::shared_ptr<SequenceRearrangeNode> sequence_rearrange_node =  context->master_graph->add_node<SequenceRearrangeNode>({input}, {output});
        sequence_rearrange_node->init(new_order);
    }
    catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor  ROCAL_API_CALL
rocalRotate(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalFloatParam p_angle,
        unsigned dest_width,
        unsigned dest_height,
        RocalResizeInterpolationType interpolation_type,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {

    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto angle = static_cast<FloatParam*>(p_angle);
    try {
        if(dest_width == 0 || dest_height == 0) {
            dest_width = input->info().max_shape()[0];
            dest_height = input->info().max_shape()[1];
        }
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(rocal_tensor_output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(rocal_tensor_output_datatype);

        // For the rotate node, user can create an image with a different width and height
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensor_layout == RocalTensorlayout::NHWC) {
            out_dims[1] = dest_height;
            out_dims[2] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NCHW) {
            out_dims[2] = dest_height;
            out_dims[3] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFHWC) {
            out_dims[2] = dest_height;
            out_dims[3] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFCHW) {
            out_dims[3] = dest_height;
            out_dims[4] = dest_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::shared_ptr<RotateNode> rotate_node =  context->master_graph->add_node<RotateNode>({input}, {output});
        rotate_node->init(angle, interpolation_type);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<RotateMetaNode,RotateNode>(rotate_node);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor  ROCAL_API_CALL
rocalRotateFixed(
        RocalContext p_context,
        RocalTensor p_input,
        float angle,
        bool is_output,
        unsigned dest_width,
        unsigned dest_height,
        RocalResizeInterpolationType interpolation_type,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);

    try {
        if(dest_width == 0 || dest_height == 0) {
            dest_width = input->info().max_shape()[0];
            dest_height = input->info().max_shape()[1];
        }
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(rocal_tensor_output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(rocal_tensor_output_datatype);

        // For the rotate node, user can create an image with a different width and height
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensor_layout == RocalTensorlayout::NHWC) {
            out_dims[1] = dest_height;
            out_dims[2] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NCHW) {
            out_dims[2] = dest_height;
            out_dims[3] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFHWC) {
            out_dims[2] = dest_height;
            out_dims[3] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFCHW) {
            out_dims[3] = dest_height;
            out_dims[4] = dest_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();

        std::shared_ptr<RotateNode> rotate_node = context->master_graph->add_node<RotateNode>({input}, {output});
        rotate_node->init(angle, interpolation_type);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<RotateMetaNode,RotateNode>(rotate_node);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}
/* TO be removed - Flip
RocalImage ROCAL_API_CALL
rocalFlip(
        RocalContext p_context,
        RocalImage p_input,
        RocalFlipAxis axis,
        bool is_output)
{
    Image* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {
        output = context->master_graph->create_image(input->info(), is_output);
        std::shared_ptr<FlipNode> flip_node =  context->master_graph->add_node<FlipNode>({input}, {output});
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<FlipMetaNode,FlipNode>(flip_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}
*/

RocalTensor ROCAL_API_CALL
rocalGamma(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalFloatParam p_alpha,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<GammaNode>({input}, {output})->init(alpha);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalGammaFixed(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        float alpha,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<GammaNode>({input}, {output})->init(alpha);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalHue(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalFloatParam p_hue,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto hue = static_cast<FloatParam*>(p_hue);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<HueNode>({input}, {output})->init(hue);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalHueFixed(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        float hue,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<HueNode>({input}, {output})->init(hue);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalSaturation(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalFloatParam p_sat,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto sat = static_cast<FloatParam*>(p_sat);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<SaturationNode>({input}, {output})->init(sat);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalSaturationFixed(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        float sat,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<SaturationNode>({input}, {output})->init(sat);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor  ROCAL_API_CALL
rocalCropResize(
        RocalContext p_context,
        RocalTensor p_input,
        unsigned dest_width, unsigned dest_height,
        bool is_output,
        RocalFloatParam p_area,
        RocalFloatParam p_aspect_ratio,
        RocalFloatParam p_x_center_drift,
        RocalFloatParam p_y_center_drift,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto area = static_cast<FloatParam*>(p_area);
    auto aspect_ratio = static_cast<FloatParam*>(p_aspect_ratio);
    auto x_center_drift = static_cast<FloatParam*>(p_x_center_drift);
    auto y_center_drift = static_cast<FloatParam*>(p_y_center_drift);
    try {
        if(dest_width == 0 || dest_height == 0)
            THROW("CropResize node needs tp receive non-zero destination dimensions")

        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(rocal_tensor_output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(rocal_tensor_output_datatype);
        // For the crop resize node, user can create an image with a different width and height
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensor_layout == RocalTensorlayout::NHWC) {
            out_dims[1] = dest_height;
            out_dims[2] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NCHW) {
            out_dims[2] = dest_height;
            out_dims[3] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFHWC) {
            out_dims[2] = dest_height;
            out_dims[3] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFCHW) {
            out_dims[3] = dest_height;
            out_dims[4] = dest_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);

        // For the nodes that user provides the output size the dimension of all the images after this node will be fixed and equal to that size
        output->reset_tensor_roi();

        std::shared_ptr<CropResizeNode> crop_resize_node =  context->master_graph->add_node<CropResizeNode>({input}, {output});
        crop_resize_node->init(area, aspect_ratio, x_center_drift, y_center_drift);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropResizeMetaNode,CropResizeNode>(crop_resize_node);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what());
    }
    return output; // Changed to input----------------IMPORTANT
}


RocalTensor  ROCAL_API_CALL
rocalCropResizeFixed(
        RocalContext p_context,
        RocalTensor p_input,
        unsigned dest_width, unsigned dest_height,
        bool is_output,
        float area,
        float aspect_ratio,
        float x_center_drift,
        float y_center_drift,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        if(dest_width == 0 || dest_height == 0)
            THROW("CropResize node needs tp receive non-zero destination dimensions")

        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(rocal_tensor_output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(rocal_tensor_output_datatype);

        // For the crop resize node, user can create an image with a different width and height
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensor_layout == RocalTensorlayout::NHWC) {
            out_dims[1] = dest_height;
            out_dims[2] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NCHW) {
            out_dims[2] = dest_height;
            out_dims[3] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFHWC) {
            out_dims[2] = dest_height;
            out_dims[3] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFCHW) {
            out_dims[3] = dest_height;
            out_dims[4] = dest_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);

        // user provides the output size and the dimension of all the images after this node will be fixed and equal to that size
        output->reset_tensor_roi();
        std::cerr<<"check in api_augmentations.cpp\n ";
        std::shared_ptr<CropResizeNode> crop_resize_node =  context->master_graph->add_node<CropResizeNode>({input}, {output});
        crop_resize_node->init(area, aspect_ratio, x_center_drift, y_center_drift);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropResizeMetaNode,CropResizeNode>(crop_resize_node);

    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what());
    }
    return output; // Changed to input----------------IMPORTANT
}

RocalTensor  ROCAL_API_CALL
rocalResize(
        RocalContext p_context,
        RocalTensor p_input,
        unsigned dest_width,
        unsigned dest_height,
        bool is_output,
        RocalResizeScalingMode scaling_mode,
        std::vector<unsigned> max_size,
        unsigned resize_shorter,
        unsigned resize_longer,
        RocalResizeInterpolationType interpolation_type,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {

    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
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

        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(rocal_tensor_output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(rocal_tensor_output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensor_layout == RocalTensorlayout::NHWC) {
            out_dims[1] = max_out_height;
            out_dims[2] = max_out_width;
        } else if(op_tensor_layout == RocalTensorlayout::NCHW) {
            out_dims[2] = max_out_height;
            out_dims[3] = max_out_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFHWC) {
            out_dims[2] = max_out_height;
            out_dims[3] = max_out_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFCHW) {
            out_dims[3] = max_out_height;
            out_dims[4] = max_out_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();

        std::shared_ptr<ResizeNode> resize_node = context->master_graph->add_node<ResizeNode>({input}, {output});
        resize_node->init(out_width, out_height, resize_scaling_mode, maximum_size, interpolation_type);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor  ROCAL_API_CALL
ROCAL_API_CALL rocalResizeMirrorNormalize(RocalContext p_context,
                                          RocalTensor p_input,
                                          unsigned dest_width,
                                          unsigned dest_height,
                                          std::vector<float> &mean,
                                          std::vector<float> &std_dev,
                                          bool is_output,
                                          RocalResizeScalingMode scaling_mode,
                                          std::vector<unsigned> max_size,
                                          unsigned resize_shorter,
                                          unsigned resize_longer,
                                          RocalResizeInterpolationType interpolation_type,
                                          RocalIntParam p_mirror,
                                          RocalTensorLayout rocal_tensor_output_layout,
                                          RocalTensorOutputType rocal_tensor_output_datatype) {
    if(!p_context || !p_input || dest_width == 0 || dest_height == 0 )
        THROW("Null values passed as input")
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto mirror = static_cast<IntParam *>(p_mirror);

    try {
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

        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(rocal_tensor_output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(rocal_tensor_output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensor_layout == RocalTensorlayout::NHWC) {
            out_dims[1] = max_out_height;
            out_dims[2] = max_out_width;
        } else if(op_tensor_layout == RocalTensorlayout::NCHW) {
            out_dims[2] = max_out_height;
            out_dims[3] = max_out_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFHWC) {
            out_dims[2] = max_out_height;
            out_dims[3] = max_out_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFCHW) {
            out_dims[3] = max_out_height;
            out_dims[4] = max_out_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::shared_ptr<ResizeMirrorNormalizeNode> rmn_node = context->master_graph->add_node<ResizeMirrorNormalizeNode>({input}, {output});
        rmn_node->init(out_width, out_height, resize_scaling_mode, maximum_size, interpolation_type, mean, std_dev, mirror);
        // if (context->master_graph->meta_data_graph())
        //     context->master_graph->meta_add_node<ResizeMirrorNormalizeMetaNode,ResizeMirrorNormalizeNode>(rmn_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what());
    }

    return output; // Changed to input----------------IMPORTANT
}

RocalTensor ROCAL_API_CALL
rocalBrightness(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalFloatParam p_alpha,
        RocalFloatParam p_beta,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    auto beta = static_cast<FloatParam*>(p_beta);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<BrightnessNode>({input}, {output})->init(alpha, beta);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalBrightnessFixed(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        float alpha,
        float beta,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<BrightnessNode>({input}, {output})->init(alpha, beta);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalBlur(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalIntParam p_sdev,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }
    std::cerr<<"\n Blur in rocal_api_augmentation";
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto sdev = static_cast<IntParam*>(p_sdev);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<BlurNode>({input}, {output})->init(sdev);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalBlurFixed(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        int sdev,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }
    std::cerr<<"\n Blur in rocal_api_augmentation";
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<BlurNode>({input}, {output})->init(sdev);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalBlend(
        RocalContext p_context,
        RocalTensor p_input1,
        RocalTensor p_input2,
        bool is_output,
        RocalFloatParam p_ratio,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input1 == nullptr) || (p_input2 == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input1 = static_cast<Tensor*>(p_input1);
    auto input2 = static_cast<Tensor*>(p_input2);
    auto ratio = static_cast<FloatParam*>(p_ratio);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input1->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<BlendNode>({input1, input2}, {output})->init(ratio);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalBlendFixed(
        RocalContext p_context,
        RocalTensor p_input1,
        RocalTensor p_input2,
        bool is_output,
        float ratio,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input1 == nullptr) || (p_input2 == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input1 = static_cast<Tensor*>(p_input1);
    auto input2 = static_cast<Tensor*>(p_input2);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input1->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<BlendNode>({input1, input2}, {output})->init(ratio);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor  ROCAL_API_CALL
rocalWarpAffine(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        unsigned dest_height, unsigned dest_width,
        RocalFloatParam p_x0, RocalFloatParam p_x1,
        RocalFloatParam p_y0, RocalFloatParam p_y1,
        RocalFloatParam p_o0, RocalFloatParam p_o1,
        RocalResizeInterpolationType interpolation_type,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto x0 = static_cast<FloatParam*>(p_x0);
    auto x1 = static_cast<FloatParam*>(p_x1);
    auto y0 = static_cast<FloatParam*>(p_y0);
    auto y1 = static_cast<FloatParam*>(p_y1);
    auto o0 = static_cast<FloatParam*>(p_o0);
    auto o1 = static_cast<FloatParam*>(p_o1);
    try {
        if(dest_width == 0 || dest_height == 0) {
            dest_width = input->info().max_shape()[0];
            dest_height = input->info().max_shape()[1];
        }
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(rocal_tensor_output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(rocal_tensor_output_datatype);

        // For the warp affine node, user can create an image with a different width and height
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensor_layout == RocalTensorlayout::NHWC) {
            out_dims[1] = dest_height;
            out_dims[2] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NCHW) {
            out_dims[2] = dest_height;
            out_dims[3] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFHWC) {
            out_dims[2] = dest_height;
            out_dims[3] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFCHW) {
            out_dims[3] = dest_height;
            out_dims[4] = dest_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<WarpAffineNode>({input}, {output})->init(x0, x1, y0, y1, o0, o1, static_cast<int>(interpolation_type));
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor  ROCAL_API_CALL
rocalWarpAffineFixed(
        RocalContext p_context,
        RocalTensor p_input,
        float x0, float x1,
        float y0, float y1,
        float o0, float o1,
        bool is_output,
        unsigned int dest_height,
        unsigned int dest_width,
        RocalResizeInterpolationType interpolation_type,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        if(dest_width == 0 || dest_height == 0) {
            dest_width = input->info().max_shape()[0];
            dest_height = input->info().max_shape()[1];
        }
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(rocal_tensor_output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(rocal_tensor_output_datatype);

        // For the warp affine node, user can create an image with a different width and height
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensor_layout == RocalTensorlayout::NHWC) {
            out_dims[1] = dest_height;
            out_dims[2] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NCHW) {
            out_dims[2] = dest_height;
            out_dims[3] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFHWC) {
            out_dims[2] = dest_height;
            out_dims[3] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFCHW) {
            out_dims[3] = dest_height;
            out_dims[4] = dest_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        context->master_graph->add_node<WarpAffineNode>({input}, {output})->init(x0, x1, y0, y1, o0, o1, static_cast<int>(interpolation_type));
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor  ROCAL_API_CALL
rocalFishEye(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(rocal_tensor_output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(rocal_tensor_output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<FisheyeNode>({input}, {output});
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalVignette(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalFloatParam p_sdev,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto sdev = static_cast<FloatParam*>(p_sdev);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<VignetteNode>({input}, {output})->init(sdev);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalVignetteFixed(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        float sdev,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<VignetteNode>({input}, {output})->init(sdev);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalJitter(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalIntParam p_kernel_size,
        int seed,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto kernel_size = static_cast<IntParam*>(p_kernel_size);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<JitterNode>({input}, {output})->init(kernel_size, seed);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalJitterFixed(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        int kernel_size,
        int seed,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<JitterNode>({input}, {output})->init(kernel_size, seed);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalSnPNoise(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalFloatParam noise_prob,
        RocalFloatParam salt_prob,
        RocalFloatParam noise_val,
        RocalFloatParam salt_val,
        int seed,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
        Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto noise_probability = static_cast<FloatParam*>(noise_prob);
    auto salt_probability = static_cast<FloatParam*>(salt_prob);
    auto noise_value = static_cast<FloatParam*>(noise_val);
    auto salt_value = static_cast<FloatParam*>(salt_val);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<SnPNoiseNode>({input}, {output})->init(noise_probability, salt_probability, noise_value ,salt_value,seed);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalSnPNoiseFixed(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        float noise_prob,
        float salt_prob,
        float noise_val,
        float salt_val,
        int seed,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<SnPNoiseNode>({input}, {output})->init(noise_prob, salt_prob, noise_val ,salt_val,seed);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalFlip(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalIntParam p_horizontal_flag,
        RocalIntParam p_vertical_flag,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto horizontal_flag = static_cast<IntParam*>(p_horizontal_flag);
    auto vertical_flag = static_cast<IntParam*>(p_vertical_flag);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(rocal_tensor_output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(rocal_tensor_output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<FlipNode>({input}, {output})->init(horizontal_flag, vertical_flag);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalFlipFixed(
        RocalContext p_context,
        RocalTensor p_input,
        int horizontal_flag,
        int vertical_flag,
        bool is_output,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(rocal_tensor_output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(rocal_tensor_output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<FlipNode>({input}, {output})->init(horizontal_flag, vertical_flag);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalContrast(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalFloatParam p_min,
        RocalFloatParam p_max,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto min = static_cast<FloatParam*>(p_min);
    auto max = static_cast<FloatParam*>(p_max);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ContrastNode>({input}, {output})->init(min, max);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalContrastFixed(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        float min,
        float max,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ContrastNode>({input}, {output})->init(min, max);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


RocalTensor ROCAL_API_CALL
rocalSnow(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalFloatParam p_shift,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
        Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto shift = static_cast<FloatParam*>(p_shift);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<SnowNode>({input}, {output})->init(shift);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalSnowFixed(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        float shift,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
        Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<SnowNode>({input}, {output})->init(shift);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalRain(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalFloatParam p_rain_value,
        RocalIntParam p_rain_width,
        RocalIntParam p_rain_height,
        RocalFloatParam p_rain_transparency,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
        Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto rain_width = static_cast<IntParam*>(p_rain_width);
    auto rain_height = static_cast<IntParam*>(p_rain_height);
    auto rain_transparency = static_cast<FloatParam*>(p_rain_transparency);
    auto rain_value = static_cast<FloatParam*>(p_rain_value);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<RainNode>({input}, {output})->init(rain_value, rain_width, rain_height, rain_transparency);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalRainFixed(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        float rain_value,
        int rain_width,
        int rain_height,
        float rain_transparency,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
        Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<RainNode>({input}, {output})->init(rain_value, rain_width, rain_height, rain_transparency);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalColorTemp(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalIntParam p_adj_value_param,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto adj_value_param = static_cast<IntParam*>(p_adj_value_param);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ColorTemperatureNode>({input}, {output})->init(adj_value_param);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalColorTempFixed(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        int adj_value_param,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ColorTemperatureNode>({input}, {output})->init(adj_value_param);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalFog(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalFloatParam p_fog_param,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto fog_param = static_cast<FloatParam*>(p_fog_param);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<FogNode>({input}, {output})->init(fog_param);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalFogFixed(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        float fog_param,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<FogNode>({input}, {output})->init(fog_param);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalPixelate(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<PixelateNode>({input}, {output});
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalLensCorrection(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalFloatParam p_strength,
        RocalFloatParam p_zoom,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto strength = static_cast<FloatParam*>(p_strength);
    auto zoom = static_cast<FloatParam*>(p_zoom);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(rocal_tensor_output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(rocal_tensor_output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<LensCorrectionNode>({input}, {output})->init(strength, zoom);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalLensCorrectionFixed(
        RocalContext p_context,
        RocalTensor p_input,
        float strength,
        float zoom,
        bool is_output,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(rocal_tensor_output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(rocal_tensor_output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<LensCorrectionNode>({input}, {output})->init(strength, zoom);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalExposure(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalFloatParam p_shift,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto shift = static_cast<FloatParam*>(p_shift);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ExposureNode>({input}, {output})->init(shift);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalExposureFixed(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        float shift,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);
        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ExposureNode>({input}, {output})->init(shift);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalColorTwist(RocalContext p_context,
                RocalTensor p_input,
                bool is_output,
                RocalFloatParam p_alpha,
                RocalFloatParam p_beta,
                RocalFloatParam p_hue,
                RocalFloatParam p_sat,
                RocalTensorLayout rocal_tensor_output_layout,
                RocalTensorOutputType rocal_tensor_output_datatype)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    auto beta = static_cast<FloatParam*>(p_beta);
    auto hue = static_cast<FloatParam*>(p_hue);
    auto sat = static_cast<FloatParam*>(p_sat);

    try
    {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ColorTwistNode>({input}, {output})->init(alpha, beta, hue, sat);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalColorTwistFixed(RocalContext p_context,
                RocalTensor p_input,
                bool is_output,
                float alpha,
                float beta,
                float hue,
                float sat,
                RocalTensorLayout rocal_tensor_output_layout,
                RocalTensorOutputType rocal_tensor_output_datatype)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try
    {
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        output = context->master_graph->create_tensor(output_info, is_output);
        context->master_graph->add_node<ColorTwistNode>({input}, {output})->init(alpha, beta, hue, sat);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalCropMirrorNormalize(RocalContext p_context, RocalTensor p_input, unsigned crop_height,
                         unsigned crop_width,float start_x, float start_y, std::vector<float> &mean,
                         std::vector<float> &std_dev, bool is_output, RocalIntParam p_mirror,
                         RocalTensorLayout rocal_tensor_output_layout,
                         RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto mirror = static_cast<IntParam *>(p_mirror);
    try {
        if( crop_width == 0 || crop_height == 0)
            THROW("Null values passed as input")
        RocalTensorlayout op_tensorLayout = (RocalTensorlayout)rocal_tensor_output_layout;
        RocalTensorDataType op_tensorDataType = (RocalTensorDataType)rocal_tensor_output_datatype;
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensorLayout);
        output_info.set_data_type(op_tensorDataType);

        // For the crop mirror normalize resize node, user can create an image with a different width and height
        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensorLayout == RocalTensorlayout::NHWC) {
            out_dims[1] = crop_height;
            out_dims[2] = crop_width;
        } else if(op_tensorLayout == RocalTensorlayout::NCHW) {
            out_dims[2] = crop_height;
            out_dims[3] = crop_width;
        } else if(op_tensorLayout == RocalTensorlayout::NFHWC) {
            out_dims[2] = crop_height;
            out_dims[3] = crop_width;
        } else if(op_tensorLayout == RocalTensorlayout::NFCHW) {
            out_dims[3] = crop_height;
            out_dims[4] = crop_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        std::shared_ptr<CropMirrorNormalizeNode> cmn_node =  context->master_graph->add_node<CropMirrorNormalizeNode>({input}, {output});
        cmn_node->init(crop_height, crop_width, start_x, start_y, mean, std_dev, mirror);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropMirrorNormalizeMetaNode,CropMirrorNormalizeNode>(cmn_node);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalCrop(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalFloatParam p_crop_width,
        RocalFloatParam p_crop_height,
        RocalFloatParam p_crop_depth,
        RocalFloatParam p_crop_pox_x,
        RocalFloatParam p_crop_pos_y,
        RocalFloatParam p_crop_pos_z,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto crop_h = static_cast<FloatParam*>(p_crop_height);
    auto crop_w = static_cast<FloatParam*>(p_crop_width);
    auto x_drift = static_cast<FloatParam*>(p_crop_pox_x);
    auto y_drift = static_cast<FloatParam*>(p_crop_pos_y);

    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(rocal_tensor_output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(rocal_tensor_output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::shared_ptr<CropNode> crop_node =  context->master_graph->add_node<CropNode>({input}, {output});
        crop_node->init(crop_h, crop_w, x_drift, y_drift);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropMetaNode,CropNode>(crop_node);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor  ROCAL_API_CALL
rocalCropFixed(
        RocalContext p_context,
        RocalTensor p_input,
        unsigned crop_width,
        unsigned crop_height,
        unsigned crop_depth,
        bool is_output,
        float crop_pos_x,
        float crop_pos_y,
        float crop_pos_z,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        if(crop_width == 0 || crop_height == 0 || crop_depth == 0)
            THROW("Crop node needs to receive non-zero destination dimensions")
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(rocal_tensor_output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(rocal_tensor_output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);

        // For the crop node, user can create an tensor with a different width and height
        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensor_layout == RocalTensorlayout::NHWC) {
            out_dims[1] = crop_height;
            out_dims[2] = crop_width;
        } else if(op_tensor_layout == RocalTensorlayout::NCHW) {
            out_dims[2] = crop_height;
            out_dims[3] = crop_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFHWC) {
            out_dims[2] = crop_height;
            out_dims[3] = crop_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFCHW) {
            out_dims[3] = crop_height;
            out_dims[4] = crop_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::shared_ptr<CropNode> crop_node =  context->master_graph->add_node<CropNode>({input}, {output});
        crop_node->init(crop_height, crop_width, crop_pos_x, crop_pos_y);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropMetaNode,CropNode>(crop_node);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor  ROCAL_API_CALL
rocalCropCenterFixed(
        RocalContext p_context,
        RocalTensor p_input,
        unsigned crop_width,
        unsigned crop_height,
        unsigned crop_depth,
        bool is_output,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        if(crop_width == 0 || crop_height == 0 || crop_depth == 0)
            THROW("Crop node needs to receive non-zero destination dimensions")

        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(rocal_tensor_output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(rocal_tensor_output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);

        // For the crop node, user can create an tensor with a different width and height
        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensor_layout == RocalTensorlayout::NHWC) {
            out_dims[1] = crop_height;
            out_dims[2] = crop_width;
        } else if(op_tensor_layout == RocalTensorlayout::NCHW) {
            out_dims[2] = crop_height;
            out_dims[3] = crop_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFHWC) {
            out_dims[2] = crop_height;
            out_dims[3] = crop_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFCHW) {
            out_dims[3] = crop_height;
            out_dims[4] = crop_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::shared_ptr<CropNode> crop_node =  context->master_graph->add_node<CropNode>({input}, {output});
        crop_node->init(crop_height, crop_width);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropMetaNode,CropNode>(crop_node);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor  ROCAL_API_CALL
rocalResizeCropMirrorFixed(
        RocalContext p_context,
        RocalTensor p_input,
        unsigned dest_width,
        unsigned dest_height,
        bool is_output,
        unsigned crop_h,
        unsigned crop_w,
        RocalIntParam p_mirror,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype)
{
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }
    auto mirror = static_cast<IntParam *>(p_mirror);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try
    {
        if(dest_width == 0 || dest_height == 0)
            THROW("Crop Mirror node needs tp receive non-zero destination dimensions")

        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(rocal_tensor_output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(rocal_tensor_output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensor_layout == RocalTensorlayout::NHWC) {
            out_dims[1] = dest_height;
            out_dims[2] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NCHW) {
            out_dims[2] = dest_height;
            out_dims[3] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFHWC) {
            out_dims[2] = dest_height;
            out_dims[3] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFCHW) {
            out_dims[3] = dest_height;
            out_dims[4] = dest_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::shared_ptr<ResizeCropMirrorNode> rcm_node =  context->master_graph->add_node<ResizeCropMirrorNode>({input}, {output});
        rcm_node->init(crop_h, crop_w, mirror);
        if (context->master_graph->meta_data_graph())
        context->master_graph->meta_add_node<ResizeCropMirrorMetaNode,ResizeCropMirrorNode>(rcm_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

extern "C"  RocalTensor  ROCAL_API_CALL rocalResizeCropMirror( RocalContext p_context, RocalTensor p_input,
                                                           unsigned dest_width, unsigned dest_height,
                                                            bool is_output, RocalFloatParam p_crop_height,
                                                            RocalFloatParam p_crop_width, RocalIntParam p_mirror,
                                                            RocalTensorLayout rocal_tensor_output_layout,
                                                            RocalTensorOutputType rocal_tensor_output_datatype)
{
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto crop_h = static_cast<FloatParam*>(p_crop_height);
    auto crop_w = static_cast<FloatParam*>(p_crop_width);
    auto mirror  = static_cast<IntParam*>(p_mirror);
    try
    {
        if(dest_width == 0 || dest_height == 0)
            THROW("Crop Mirror node needs tp receive non-zero destination dimensions")
        // For the resize node, user can create an image with a different width and height
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(rocal_tensor_output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(rocal_tensor_output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        std::vector<size_t> out_dims = output_info.dims();
        if(op_tensor_layout == RocalTensorlayout::NHWC) {
            out_dims[1] = dest_height;
            out_dims[2] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NCHW) {
            out_dims[2] = dest_height;
            out_dims[3] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFHWC) {
            out_dims[2] = dest_height;
            out_dims[3] = dest_width;
        } else if(op_tensor_layout == RocalTensorlayout::NFCHW) {
            out_dims[3] = dest_height;
            out_dims[4] = dest_width;
        }
        output_info.set_dims(out_dims);
        output = context->master_graph->create_tensor(output_info, is_output);
        std::shared_ptr<ResizeCropMirrorNode> rcm_node =  context->master_graph->add_node<ResizeCropMirrorNode>({input}, {output});
        rcm_node->init(crop_h, crop_w, mirror);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<ResizeCropMirrorMetaNode,ResizeCropMirrorNode>(rcm_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

extern "C" RocalTensor ROCAL_API_CALL
rocalRandomCrop(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalFloatParam p_crop_area_factor,
        RocalFloatParam p_crop_aspect_ratio,
        RocalFloatParam p_crop_pox_x,
        RocalFloatParam p_crop_pos_y,
        int num_of_attempts,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype)
{
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto crop_area_factor  = static_cast<FloatParam*>(p_crop_area_factor);
    auto crop_aspect_ratio = static_cast<FloatParam*>(p_crop_aspect_ratio);
    auto x_drift = static_cast<FloatParam*>(p_crop_pox_x);
    auto y_drift = static_cast<FloatParam*>(p_crop_pos_y);

    try
    {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(rocal_tensor_output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(rocal_tensor_output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::shared_ptr<RandomCropNode> crop_node =  context->master_graph->add_node<RandomCropNode>({input}, {output});
        crop_node->init(crop_area_factor, crop_aspect_ratio, x_drift, y_drift, num_of_attempts);
        // if (context->master_graph->meta_data_graph())
        //     context->master_graph->meta_add_node<SSDRandomCropMetaNode,RandomCropNode>(crop_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalSSDRandomCrop(
        RocalContext p_context,
        RocalTensor p_input,
        bool is_output,
        RocalFloatParam p_threshold,
        RocalFloatParam p_crop_area_factor,
        RocalFloatParam p_crop_aspect_ratio,
        RocalFloatParam p_crop_pox_x,
        RocalFloatParam p_crop_pos_y,
        int num_of_attempts,
        RocalTensorLayout rocal_tensor_output_layout,
        RocalTensorOutputType rocal_tensor_output_datatype) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    auto crop_area_factor  = static_cast<FloatParam*>(p_crop_area_factor);
    auto crop_aspect_ratio = static_cast<FloatParam*>(p_crop_aspect_ratio);
    auto x_drift = static_cast<FloatParam*>(p_crop_pox_x);
    auto y_drift = static_cast<FloatParam*>(p_crop_pos_y);

    try {
        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(rocal_tensor_output_layout);
        RocalTensorDataType op_tensor_datatype = static_cast<RocalTensorDataType>(rocal_tensor_output_datatype);
        TensorInfo output_info = input->info();
        output_info.set_tensor_layout(op_tensor_layout);
        output_info.set_data_type(op_tensor_datatype);
        output = context->master_graph->create_tensor(output_info, is_output);
        output->reset_tensor_roi();
        std::shared_ptr<SSDRandomCropNode> crop_node =  context->master_graph->add_node<SSDRandomCropNode>({input}, {output});
        crop_node->init(crop_area_factor, crop_aspect_ratio, x_drift, y_drift, num_of_attempts);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<SSDRandomCropMetaNode,SSDRandomCropNode>(crop_node);
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalCopy(RocalContext p_context,
          RocalTensor p_input,
          bool is_output) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        output = context->master_graph->create_tensor(input->info(), is_output);
        context->master_graph->add_node<CopyNode>({input}, {output});
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RocalTensor  ROCAL_API_CALL
rocalNop(RocalContext p_context,
         RocalTensor p_input,
         bool is_output) {
    Tensor* output = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr)) {
        ERR("Invalid ROCAL context or invalid input tensor")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        output = context->master_graph->create_tensor(input->info(), is_output);
        context->master_graph->add_node<NopNode>({input}, {output});
    } catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

std::pair<RocalTensor, RocalTensor> ROCAL_API_CALL
rocalNonSilentRegion(RocalContext p_context,
                     RocalTensor p_input,
                     bool is_output,
                     float cutoff_db,
                     float reference_power,
                     int reset_interval,
                     int window_length) {
    Tensor* output1 = nullptr;
    Tensor* output2 = nullptr;
    if ((p_context == nullptr) || (p_input == nullptr))
        ERR("Invalid ROCAL context or invalid input tensor")

    TensorList output_tensors;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Tensor*>(p_input);
    try {
        RocalTensorDataType tensor_data_type = RocalTensorDataType::INT32;
        unsigned number_of_dims = 4;
        std::vector<size_t> dims(number_of_dims, 1);
        dims.at(0) = context->user_batch_size();
        auto info = TensorInfo(dims,
                               context->master_graph->mem_type(),
                               tensor_data_type);
        info.set_tensor_layout(RocalTensorlayout::NONE);
        output1 = context->master_graph->create_tensor(info, is_output);
        output2 = context->master_graph->create_tensor(info, is_output);
        output_tensors.push_back(output1);
        output_tensors.push_back(output2);
        context->master_graph->add_node<NonSilentRegionNode>({input}, {output1, output2})->init(cutoff_db, reference_power, window_length, reset_interval);
    }
    catch(const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
    }

    return std::make_pair(output_tensors.at(0), output_tensors.at(1));
}