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
    _flip_axis_val.resize(_batch_size);
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
    _flip_axis = _node->get_flip_axis();


    vxCopyArrayRange((vx_array)_src_width, 0, _batch_size, sizeof(uint), _src_width_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_src_height, 0, _batch_size, sizeof(uint), _src_height_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_flip_axis, 0, _batch_size, sizeof(int), _flip_axis_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    for (int i = 0; i < _batch_size; i++)
    {
        float img_width = input_meta_data->get_img_sizes_batch()[i].data()->w;
        //std::cout<<"src_width:"<<img_width<<std::endl;

        auto bb_count = input_meta_data->get_bb_labels_batch()[i].size();
        //int labels_buf[bb_count];
        float coords_buf[bb_count * 4];

        //memcpy(labels_buf, input_meta_data->get_bb_labels_batch()[i].data(),  sizeof(int)*bb_count);
        memcpy(coords_buf, input_meta_data->get_bb_cords_batch()[i].data(), input_meta_data->get_bb_cords_batch()[i].size() * sizeof(BoundingBoxCord));

        BoundingBoxCords bb_coords;
        ImageJointsData img_joints_data;
        //BoundingBoxLabels bb_labels;
        
        if (pose_estimation)
        {
            for (unsigned int object_index = 0; object_index < bb_count; object_index++)
            {
                KeyPoints key_points;
                KeyPointsVisibility key_points_visibility;
                KeyPoint key_point0;
                KeyPointVisibility key_point0_vis;
                BoundingBoxCenter bb_center;
                JointsData joints_data;
                
                joints_data = input_meta_data->get_img_joints_data_batch()[i][object_index];
                key_point0 = joints_data.joints[0];
                key_point0_vis = joints_data.joints_visibility[0];
                key_point0.x = (img_width - key_point0.x - 1) * (key_point0_vis.xv);
                bb_center = joints_data.center;
                bb_center.xc = img_width - bb_center.xc - 1;
                
                key_points.push_back(key_point0);
                key_points_visibility.push_back(key_point0_vis);
                //std::cout<<"Difference:"<<key_point0.x<<std::endl;

                for (unsigned int keypoint_index = 1; keypoint_index < NUMBER_OF_KEYPOINTS; keypoint_index = keypoint_index + 2)
                {
                    //std::cout<<"Flipping keypoints: "<< keypoint_index<<" "<<keypoint_index+1<<std::endl;
                    KeyPoint key_point1, key_point2;
                    KeyPointVisibility key_point1_vis,key_point2_vis;
                    key_point1 = joints_data.joints[keypoint_index];
                    key_point2 = joints_data.joints[keypoint_index+1];
                    key_point1_vis = joints_data.joints_visibility[keypoint_index];
                    key_point2_vis = joints_data.joints_visibility[keypoint_index+1];
                    
                    //Update the keypoint co-ordinates
                    key_point1.x = (img_width - key_point1.x - 1) * (key_point1_vis.xv);
                    key_point2.x = (img_width - key_point2.x - 1) * (key_point2_vis.xv);

                    key_points.push_back(key_point2);
                    key_points.push_back(key_point1);
                    key_points_visibility.push_back(key_point2_vis);
                    key_points_visibility.push_back(key_point1_vis);   
                }
                joints_data.joints = key_points;
                joints_data.joints_visibility = key_points_visibility;
                joints_data.center = bb_center;
                img_joints_data.push_back(joints_data);
                key_points.clear();
                key_points_visibility.clear();
            }
            input_meta_data->get_img_joints_data_batch()[i] = img_joints_data;
            // input_meta_data->get_img_key_points_visibility_batch()[i] = img_key_points_visibility;
            // input_meta_data->get_bb_centers_batch()[i] = bb_centers;
        }

            for (uint j = 0, m = 0; j < bb_count; j++)
            {
                BoundingBoxCord box;
                box.l = coords_buf[m++];
                box.t = coords_buf[m++];
                box.r = coords_buf[m++];
                box.b = coords_buf[m++];
                //std::cout<<"l:"<<box.l<<" t:"<<box.t<<" r:"<<box.r<<" b:"<<box.b<<std::endl;

                if (_flip_axis_val[i] == 0)
                {
                    float l = 1 - box.r;
                    box.r = 1 - box.l;
                    box.l = l;
                }
                else if (_flip_axis_val[i] == 1)
                {
                    float t = 1 - box.b;
                    box.b = 1 - box.t;
                    box.t = t;
                }

                bb_coords.push_back(box);
                //bb_labels.push_back(labels_buf[j]);
            }
            if (bb_coords.size() == 0)
            {
                BoundingBoxCord temp_box;
                temp_box.l = temp_box.t = 0;
                temp_box.r = temp_box.b = 1;
                bb_coords.push_back(temp_box);
                //bb_labels.push_back(0);
            }
            input_meta_data->get_bb_cords_batch()[i] = bb_coords;
            //input_meta_data->get_bb_labels_batch()[i] = bb_labels;
        }
    }
