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
    _src_height_val.resize(_batch_size);
    _src_width_val.resize(_batch_size);
    _flip_axis_val.resize(_batch_size);
}
void FlipMetaNode::update_parameters(MetaDataBatch* input_meta_data)
{
    std::cout<<"flip meta node is called:"<<std::endl;

    initialize();
    std::cout<<"Flip meta node is initialized:"<<std::endl;
    
    if(_batch_size != input_meta_data->size())
    {
        _batch_size = input_meta_data->size();
    }
    _src_width = _node->get_src_width();
    _src_height = _node->get_src_height();
    //_flip_axis = _node->get_flip_axis();
    std::cout<<"Got the required meta init values:"<<std::endl;
    _flip_axis = 0;
    vxCopyArrayRange((vx_array)_src_width, 0, _batch_size, sizeof(uint),_src_width_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_src_height, 0, _batch_size, sizeof(uint),_src_height_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_flip_axis, 0, _batch_size, sizeof(int),_flip_axis_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    std::cout<<"Vx_arrays created:"<<std::endl;
    
    for(int i = 0; i < _batch_size; i++)
    {
        std::cout<<"Started processing batch data:"<<std::endl;
        auto bb_count = input_meta_data->get_bb_labels_batch()[i].size();
        auto kp_count = input_meta_data->get_img_key_points_batch()[i].size();
        std::cout<<"Got the number of keypointd:"<<std::endl;
        float keypoint_buffer[kp_count*17*3];

        int labels_buf[bb_count];
        float coords_buf[bb_count*4];
        memcpy(labels_buf, input_meta_data->get_bb_labels_batch()[i].data(),  sizeof(int)*bb_count);
        memcpy(coords_buf, input_meta_data->get_bb_cords_batch()[i].data(), input_meta_data->get_bb_cords_batch()[i].size() * sizeof(BoundingBoxCord));
        BoundingBoxCords bb_coords;
        BoundingBoxLabels bb_labels;
        ImageKeyPoints image_key_points;
        

        for(uint k = 0;k < kp_count; k++)
        {
            KeyPoints key_points;
            std::cout<<"Started processing each keypoint data:"<<std::endl;
            for(uint h = 0,n = 0;h<17;h++)
            {
                KeyPoint kp;
                kp.x=keypoint_buffer[n++];
                kp.y=keypoint_buffer[n++];
                kp.v=keypoint_buffer[n++];
                std::cout << " fx : " << kp.x << " , fy: " << kp.y << " , fv : " << kp.v << std::endl;

                if(_flip_axis_val[i] == 0)
                {
                    float temp = kp.x;
                    kp.x = kp.y;
                    kp.x = temp;     
                }
                key_points.push_back(kp);
            }
            image_key_points.push_back(key_points);
            std::cout<<"Pushed back keypoint data:"<<std::endl;
        }
        if(image_key_points.size() == 0)
        {
            KeyPoints key_points;
            for(uint h = 0;h<17;h++)
            {
                KeyPoint kp;
                kp.x=-143;
                kp.y=-143;
                kp.v=-143;
                key_points.push_back(kp);
                std::cout << " fx : " << kp.x << " , fy: " << kp.y << " , fv : " << kp.v << std::endl;
            }
            std::cout<<"keypoints are null:"<<std::endl;
            image_key_points.push_back(key_points);
        }
        input_meta_data->get_img_key_points_batch()[i] = image_key_points;
        std::cout<<"Updated keypoints in the memory:"<<std::endl;
        
        for(uint j = 0, m = 0; j < bb_count; j++)
        {
            BoundingBoxCord box;
            box.l = coords_buf[m++];
            box.t = coords_buf[m++];
            box.r = coords_buf[m++];
            box.b = coords_buf[m++];
            
            if(_flip_axis_val[i] == 0)
            {
                float l = 1 - box.r;
                box.r = 1 - box.l;
                box.l = l;     
            }
            else if(_flip_axis_val[i] == 1)
            {
                float t = 1 - box.b;
                box.b = 1 - box.t;
                box.t = t;
            }
            
            bb_coords.push_back(box);
            bb_labels.push_back(labels_buf[j]);
        }
        if(bb_coords.size() == 0)
        {
            BoundingBoxCord temp_box;
            temp_box.l = temp_box.t = 0;
            temp_box.r = temp_box.b = 1;
            bb_coords.push_back(temp_box);
            bb_labels.push_back(0);
        }
        input_meta_data->get_bb_cords_batch()[i] = bb_coords;
        input_meta_data->get_bb_labels_batch()[i] = bb_labels;
    }
}
