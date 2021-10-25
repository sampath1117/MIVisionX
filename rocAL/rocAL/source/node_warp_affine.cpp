/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

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

#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include "node_warp_affine.h"
#include "exception.h"

// void get_dir(float point[], float dir[], float r);
// void get_3rd_point(float mat[][3]);
// void get_inverse(float m[][3],float inv_m[3][3]);
// void matrix_mult(float src[2][3] , float dst[3][3] ,float affine[2][3]);

WarpAffineNode::WarpAffineNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) : 
        Node(inputs, outputs),
        _scale_factor(SCALE_RANGE[0], SCALE_RANGE[1]),
        _rotation_factor(ROTATION_RANGE[0], ROTATION_RANGE[1]),
        _x0(COEFFICIENT_RANGE_1[0], COEFFICIENT_RANGE_1[1]),
        _x1(COEFFICIENT_RANGE_0[0], COEFFICIENT_RANGE_0[1]),
        _y0(COEFFICIENT_RANGE_0[0], COEFFICIENT_RANGE_0[1]),
        _y1(COEFFICIENT_RANGE_1[0], COEFFICIENT_RANGE_1[1]),
        _o0(COEFFICIENT_RANGE_OFFSET[0], COEFFICIENT_RANGE_OFFSET[1]),
        _o1(COEFFICIENT_RANGE_OFFSET[0], COEFFICIENT_RANGE_OFFSET[1])
        //_rotate_probability(ROTATION_PROBABILITY_RANGE[0],ROTATION_PROBABILITY_RANGE[1])
{
    _is_set_meta_data = true;
}

void WarpAffineNode::create_node()
{
    if (_node)
        return;

    vx_status width_status, height_status;
    _affine.resize(6 * _batch_size);

    uint batch_size = _batch_size;
    for (uint i = 0; i < batch_size; i++)
    {
        _affine[i * 6 + 0] = _x0.renew();
        _affine[i * 6 + 1] = _y0.renew();
        _affine[i * 6 + 2] = _x1.renew();
        _affine[i * 6 + 3] = _y1.renew();
        _affine[i * 6 + 4] = _o0.renew();
        _affine[i * 6 + 5] = _o1.renew();
    }
    _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    std::vector<uint32_t> dst_roi_width(_batch_size, _outputs[0]->info().width());
    std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().height_single());
    width_status = vxAddArrayItems(_dst_roi_width, _batch_size, dst_roi_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_dst_roi_height, _batch_size, dst_roi_height.data(), sizeof(vx_uint32));
    if (width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the rotate (vxExtrppNode_WarpAffinePD) node: " + TOSTR(width_status) + "  " + TOSTR(height_status))

    vx_status status;
    _affine_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size * 6);
    status = vxAddArrayItems(_affine_array, _batch_size * 6, _affine.data(), sizeof(vx_float32));
    _node = vxExtrppNode_WarpAffinebatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _dst_roi_width, _dst_roi_height,
                                           _affine_array, _batch_size);

    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the warp affine (vxExtrppNode_WarpAffinePD) node failed: " + TOSTR(status))
}

void WarpAffineNode::update_affine_array()
{
    if (_is_set_meta_data)
    {
        //Start of Half body transform
        auto scale_factor = _scale_factor.get();
        int output_size[2] = {(int)_outputs[0]->info().width(), (int)_outputs[0]->info().height_single()};
        float pi = 3.14;

        int ann_count = _meta_data_info->get_joints_data_batch().image_id_batch.size();
        std::vector<int> UPPER_BODY_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<int> LOWER_BODY_IDS = {11, 12, 13, 14, 15, 16};
        // std::cout<<"Entered affine calculation code: "<<std::endl;
        for (int i = 0; i < ann_count; i++)
        {
            float src[2][3] = {0.0};
            float dst[2][3] = {1.0};
            float dst_padded[3][3] = {1.0};
            float inv_dst[3][3] = {0.0};
            float src_dir[2] = {0.0};
            float dst_dir[2] = {0.0};
            float shift[2] = {0.0};
            float *affine_matrix;
            affine_matrix = _affine.data() + (i * 6);
            int input_img_width = _meta_data_info->get_img_sizes_batch()[i].data()->w;
            int input_img_height = _meta_data_info->get_img_sizes_batch()[i].data()->h;

            Center box_center;
            Scale box_scale;

            // std::cout<<"imageID: "<<_meta_data_info->get_joints_data_batch().image_id_batch[i]<<std::endl;
            box_center = _meta_data_info->get_joints_data_batch().center_batch[i];
            box_scale = _meta_data_info->get_joints_data_batch().scale_batch[i];

            //TO DO perform halfboady only based on the random probability and 2 other conditions
            bool is_half_body = false;
            if (is_half_body)
            {
                float aspect_ratio = output_size[0] * 1.0 / output_size[1];
                half_body_transform(i, box_center, box_scale, aspect_ratio);
            }

            //TO DO for scale
            //add the flag for is_scale passed from user
            bool is_scale = false;
            if (is_scale)
            {
                box_scale[0] = box_scale[0] * scale_factor;
                box_scale[1] = box_scale[1] * scale_factor;
            }

            //TO DO for rotate
            //1.add the constant value for rotate probability from the user 
            //2.get random probability for rotate
            float rotate_prob = 0.0;
            float r = 0.0;
            float random_rotate_prob = 0.0;//_rotate_probability.get();
            std::cout<<"Rotate probability: "<< random_rotate_prob<<std::endl;
            if (rotate_prob && random_rotate_prob< rotate_prob)
            {
                r = _rotation_factor.get();
            }

            _meta_data_info->get_joints_data_batch().rotation_batch[i] = r;

            //Get the correct scale values
            float scale_temp[2] = {PIXEL_STD * box_scale[0], PIXEL_STD * box_scale[1]};
            float src_w = scale_temp[0];
            float dst_w = output_size[0] * 1.0;
            float dst_h = output_size[1] * 1.0;
            auto r_rad = pi * r / 180;

            float src_point[2] = {0.0, -src_w / 2};
            float dst_point[2] = {0.0, -dst_w / 2};

            get_dir(src_point, src_dir, r_rad);
            get_dir(dst_point, dst_dir, r_rad);

            //std::cout << "Got the direction" << std::endl;
            // std::cout << std::endl<<"xc,yc:" << box_center[0] <<" "<< box_center[1];
            // std::cout << std::endl<<"xs,ys:" << box_scale[0] <<" "<< box_scale[1];
            src[0][0] = box_center[0] + (shift[0] * scale_temp[0]);
            src[1][0] = box_center[1] + (shift[1] * scale_temp[1]);
            src[0][1] = box_center[0] + src_dir[0] + (shift[0] * scale_temp[0]);
            src[1][1] = box_center[1] + src_dir[1] + (shift[1] * scale_temp[1]);

            dst[0][0] = dst_padded[0][0] = dst_w * 0.5;
            dst[1][0] = dst_padded[1][0] = dst_h * 0.5;
            dst[0][1] = dst_padded[0][1] = dst_w * 0.5 + dst_dir[0] + (shift[0] * scale_temp[0]);
            dst[1][1] = dst_padded[1][1] = dst_h * 0.5 + dst_dir[1] + (shift[1] * scale_temp[1]);
            dst_padded[2][0] = 1.0;
            dst_padded[2][1] = 1.0;
            dst_padded[2][2] = 1.0;

            //Get the 3rd point
            get_3rd_point(src);
            get_3rd_point(dst);

            dst_padded[0][2] = dst[0][2];
            dst_padded[1][2] = dst[1][2];

            //Get the inverse matrix
            get_inverse(dst_padded, inv_dst);

            //Get the affine array
            matrix_mult(src, inv_dst, affine_matrix);

            //TO DO 
            //Subtract the width and height of source image from the translation parameters 

            // affine_matrix[2] = affine_matrix[2] + ((input_img_width/2)*affine_matrix[0]+(input_img_height/2)*affine_matrix[1]-input_img_width/2);
            // affine_matrix[5] = affine_matrix[5] + ((input_img_height/2)*affine_matrix[3]+(input_img_width/2)*affine_matrix[4]-input_img_height/2);

            // std::cout << std::endl<<"Affine matrix:" << std::endl
            //           << affine_matrix[0] << " " << affine_matrix[1] << " " << affine_matrix[2] << std::endl
            //           << affine_matrix[3] << " " << affine_matrix[4] << " " << affine_matrix[5] << std::endl<<std::endl;

            //Copy these values to local bb_centers,bb_scales
            _meta_data_info->get_joints_data_batch().center_batch[i] = box_center;
            _meta_data_info->get_joints_data_batch().scale_batch[i] = box_scale;
        }
    }

    vx_status affine_status;
    affine_status = vxCopyArrayRange((vx_array)_affine_array, 0, _batch_size * 6, sizeof(vx_float32), _affine.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST); //vxAddArrayItems(_width_array,_batch_size, _width, sizeof(vx_uint32));
    if (affine_status != 0)
        THROW(" vxCopyArrayRange failed in the WarpAffine(vxExtrppNode_WarpAffinePD) node: " + TOSTR(affine_status))
}

void WarpAffineNode::init(float x0, float x1, float y0, float y1, float o0, float o1)
{
    _x0.set_param(x0);
    _x1.set_param(x1);
    _y0.set_param(y0);
    _y1.set_param(y1);
    _o0.set_param(o0);
    _o1.set_param(o1);
}

void WarpAffineNode::init(FloatParam *scale_factor, FloatParam *rotate_probablity, FloatParam *x0, FloatParam *x1, FloatParam *y0, FloatParam *y1, FloatParam *o0, FloatParam *o1)
{
    _x0.set_param(core(x0));
    _x1.set_param(core(x1));
    _y0.set_param(core(y0));
    _y1.set_param(core(y1));
    _o0.set_param(core(o0));
    _o1.set_param(core(o1));
    // _scale_factor.set_param(core(scale_factor));
    // _rotation_factor.set_param(core(rotation_factor));
    // _rotate_probability.set_param(core(rotate_probablity));
}

void WarpAffineNode::update_node()
{
    update_affine_array();
}

void WarpAffineNode::half_body_transform(int i, Center &box_center, Scale &box_scale, float aspect_ratio)
{
    Joint joint;
    Joints upper_joints, lower_joints, selected_joints;
    JointsVisibility joints_visibility = _meta_data_info->get_joints_data_batch().joints_visibility_batch[i];

    //Seperate the keypoints into upper body and lower body
    for (uint kp_idx = 0; kp_idx < NUMBER_OF_KEYPOINTS; kp_idx++)
    {
        auto v = joints_visibility[kp_idx][0];
        if (v > 0)
        {
            joint = _meta_data_info->get_joints_data_batch().joints_batch[i][kp_idx];
            if (kp_idx <= 10)
            {
                upper_joints.push_back(joint);
            }
            else
            {
                lower_joints.push_back(joint);
            }
        }
    }

    //Any of lower body/upper body joints should have minimum 3 joints
    if (lower_joints.size() < 2 && upper_joints.size() < 2)
    {
        return;
    }

    //select any of upper body and lower body joints
    selected_joints = upper_joints;
    if (lower_joints.size() > 2) //add random factor here
    {
        selected_joints = lower_joints;
    }

    auto mean_center_x = 0.0;
    auto mean_center_y = 0.0;
    auto min_idx = 0;
    auto max_idx = 0;
    for (uint kp_idx = 0; kp_idx < selected_joints.size(); kp_idx++)
    {
        mean_center_x = mean_center_x + selected_joints[kp_idx][0];
        mean_center_y = mean_center_y + selected_joints[kp_idx][1];

        if (selected_joints[kp_idx][0] < selected_joints[min_idx][0])
        {
            min_idx = kp_idx;
        }

        if (selected_joints[kp_idx][0] > selected_joints[max_idx][0])
        {
            max_idx = kp_idx;
        }
    }

    box_center[0] = mean_center_x / selected_joints.size();
    box_center[1] = mean_center_y / selected_joints.size();

    //Calculate the scale value
    float left_top[] = {selected_joints[min_idx][0], selected_joints[min_idx][1]};
    float right_bottom[] = {selected_joints[max_idx][0], selected_joints[max_idx][1]};
    float w, h;
    w = right_bottom[0] - left_top[0];
    h = right_bottom[1] - left_top[1];

    if (w > aspect_ratio * h)
    {
        h = w * 1.0 / aspect_ratio;
    }
    else if (w < aspect_ratio * h)
    {
        w = aspect_ratio * h;
    }

    box_scale[0] = (SCALE_CONSTANT_HALF_BODY * w * 1.0) / PIXEL_STD;
    box_scale[1] = (SCALE_CONSTANT_HALF_BODY * h * 1.0) / PIXEL_STD;

    //Clear the keypoint values
    upper_joints.clear();
    lower_joints.clear();
    selected_joints.clear();
}