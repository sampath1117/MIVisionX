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

#include "meta_node_flip.h"
void FlipMetaNode::initialize()
{
    //std::cout<<"Initialize called:"<<std::endl;
    _src_height_val.resize(_batch_size);
    _src_width_val.resize(_batch_size);
    _horizontal_flip_axis_val.resize(_batch_size);
    _vertical_flip_axis_val.resize(_batch_size);
}
void FlipMetaNode::update_parameters(MetaDataBatch *input_meta_data, bool pose_estimation)
{
    //std::cout<<"flip meta node is called:"<<std::endl;

    initialize();

    if (_batch_size != input_meta_data->size())
    {
        _batch_size = input_meta_data->size();
    }

    _src_width = _node->get_src_width();
    _src_height = _node->get_src_height();
    _horizontal_flip_axis = _node->get_horizontal_flip_axis();
    _vertical_flip_axis = _node->get_vertical_flip_axis();

    vxCopyArrayRange((vx_array)_src_width, 0, _batch_size, sizeof(uint), _src_width_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_src_height, 0, _batch_size, sizeof(uint), _src_height_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_horizontal_flip_axis, 0, _batch_size, sizeof(int), _horizontal_flip_axis_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_vertical_flip_axis, 0, _batch_size, sizeof(int), _vertical_flip_axis_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    if (pose_estimation)
    {
        unsigned int ann_count = input_meta_data->get_joints_data_batch().center_batch.size();

        for (unsigned int ann_index = 0; ann_index < ann_count; ann_index++)
        {
            if (_horizontal_flip_axis_val[ann_index] == 1)
            {
                float img_width = input_meta_data->get_img_sizes_batch()[ann_index].data()->w;
                Joint joint0;
                JointVisibility joint0_visiblity;
                Joints joints;
                JointsVisibility joints_visibility;

                joint0 = input_meta_data->get_joints_data_batch().joints_batch[ann_index][0];
                joint0_visiblity = input_meta_data->get_joints_data_batch().joints_visibility_batch[ann_index][0];
                joint0[0] = (img_width - joint0[0] - 1) * (joint0_visiblity[0]);

                Center center = input_meta_data->get_joints_data_batch().center_batch[ann_index];
                center[0] = img_width - center[0] - 1;

                joints.push_back(joint0);
                joints_visibility.push_back(joint0_visiblity);
                //std::cout<<"Difference:"<<key_point0.x<<std::endl;

                for (unsigned int joint_index = 1; joint_index < NUMBER_OF_JOINTS; joint_index = joint_index + 2)
                {
                    //std::cout<<"Flipping keypoints: "<<  joint_index<<" "<< joint_index+1<<std::endl;
                    Joint joint1, joint2;
                    JointVisibility joint1_visibility, joint2_visibility;
                    joint1 = input_meta_data->get_joints_data_batch().joints_batch[ann_index][joint_index];
                    joint2 = input_meta_data->get_joints_data_batch().joints_batch[ann_index][joint_index + 1];
                    joint1_visibility = input_meta_data->get_joints_data_batch().joints_visibility_batch[ann_index][joint_index];
                    joint2_visibility = input_meta_data->get_joints_data_batch().joints_visibility_batch[ann_index][joint_index + 1];

                    //Update the keypoint co-ordinates
                    joint1[0] = (img_width - joint1[0] - 1) * (joint1_visibility[0]);
                    joint2[0] = (img_width - joint2[0] - 1) * (joint2_visibility[0]);

                    joints.push_back(joint2);
                    joints.push_back(joint1);
                    joints_visibility.push_back(joint2_visibility);
                    joints_visibility.push_back(joint1_visibility);
                }
                input_meta_data->get_joints_data_batch().joints_batch[ann_index] = joints;
                input_meta_data->get_joints_data_batch().joints_visibility_batch[ann_index] = joints_visibility;
                input_meta_data->get_joints_data_batch().center_batch[ann_index] = center;
                joints.clear();
                joints_visibility.clear();
            }
        }
    }
    else
    {
        for (int i = 0; i < _batch_size; i++)
        {
            auto bb_count = input_meta_data->get_bb_labels_batch()[i].size();
            BoundingBoxLabels labels_buf;
            BoundingBoxCords coords_buf;
            coords_buf.resize(bb_count);
            labels_buf.resize(bb_count);
            memcpy(labels_buf.data(), input_meta_data->get_bb_labels_batch()[i].data(), sizeof(int) * bb_count);
            memcpy(coords_buf.data(), input_meta_data->get_bb_cords_batch()[i].data(), input_meta_data->get_bb_cords_batch()[i].size() * sizeof(BoundingBoxCord));
            BoundingBoxCords bb_coords;

            for (uint j = 0; j < bb_count; j++)
            {
                if (_horizontal_flip_axis_val[i] == 1)
                {
                    float l = 1 - coords_buf[j].r;
                    coords_buf[j].r = 1 - coords_buf[j].l;
                    coords_buf[j].l = l;
                }
                if (_vertical_flip_axis_val[i] == 1)
                {
                    float t = 1 - coords_buf[j].b;
                    coords_buf[j].b = 1 - coords_buf[j].t;
                    coords_buf[j].t = t;
                }

                bb_coords.push_back(coords_buf[j]);
            }
            input_meta_data->get_bb_cords_batch()[i] = bb_coords;
            input_meta_data->get_bb_labels_batch()[i] = labels_buf;
        }
    }
}
