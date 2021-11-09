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

#include "meta_node_warp_affine.h"

void WarpAffineMetaNode::initialize()
{
    _src_height_val.resize(_batch_size);
    _src_width_val.resize(_batch_size);
    // _affine_val.resize(6 * _batch_size);
}

void WarpAffineMetaNode::update_parameters(MetaDataBatch *input_meta_data, bool pose_estimation)
{
    //std::cout<<"Warp affine meta node is called:"<<std::endl;

    initialize();

    if (_batch_size != input_meta_data->size())
    {
        _batch_size = input_meta_data->size();
    }

    _src_width = _node->get_src_width();
    _src_height = _node->get_src_height();
    float *affine_matrix = _node->get_affine_array();

    vxCopyArrayRange((vx_array)_src_width, 0, _batch_size, sizeof(uint), _src_width_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_src_height, 0, _batch_size, sizeof(uint), _src_height_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    
    if (pose_estimation)
    {
        unsigned int ann_count = input_meta_data->get_joints_data_batch().image_id_batch.size();
        //Update Keypoints
        for (unsigned int ann_index = 0; ann_index < ann_count; ann_index++)
        {
            Joints joints;
            // std::cout <<"Inverse Affine matrix in node_warp_affine.cpp:" << std::endl
            //           << affine_matrix[6*ann_index+0] << " " << affine_matrix[6*ann_index+1] << " " << affine_matrix[6*ann_index+2] << std::endl
            //           << affine_matrix[6*ann_index+3] << " " << affine_matrix[6*ann_index+4] << " " << affine_matrix[6*ann_index+5] << std::endl<<std::endl;

            for (unsigned int keypoint_index = 0; keypoint_index < NUMBER_OF_JOINTS; keypoint_index++)
            {
                Joint joint;
                JointVisibility joint_visibility;
                float temp_x, temp_y;
                joint = input_meta_data->get_joints_data_batch().joints_batch[ann_index][keypoint_index];
                joint_visibility = input_meta_data->get_joints_data_batch().joints_visibility_batch[ann_index][keypoint_index];

                //Matrix multiplication of Affine matrix with keypoint values
                temp_x = (affine_matrix[6*ann_index+0] * joint[0]) + (affine_matrix[6*ann_index+1] * joint[1]) + (affine_matrix[6*ann_index+2] * joint_visibility[0]);
                temp_y = (affine_matrix[6*ann_index+3] * joint[0]) + (affine_matrix[6*ann_index+4] * joint[1]) + (affine_matrix[6*ann_index+5] * joint_visibility[0]);
                joint[0] = temp_x;
                joint[1] = temp_y;
                joints.push_back(joint);
            }
            input_meta_data->get_joints_data_batch().joints_batch[ann_index] = joints;
            joints.clear();
        }
    }
}




