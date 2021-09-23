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
    _affine_val.resize(6*_batch_size);
}
void WarpAffineMetaNode::update_parameters(MetaDataBatch* input_meta_data)
{
    //std::cout<<"Warp affine meta node is called:"<<std::endl;

    initialize();
    
    if(_batch_size != input_meta_data->size())
    {
        _batch_size = input_meta_data->size();
    }

    _src_width = _node->get_src_width();
    _src_height = _node->get_src_height();
    _affine_array=_node->get_affine_array();

    vxCopyArrayRange((vx_array)_src_width, 0, _batch_size, sizeof(uint),_src_width_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_src_height, 0, _batch_size, sizeof(uint),_src_height_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_affine_array, 0, (6*_batch_size) , sizeof(float) ,_affine_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    
    for(int i = 0; i < _batch_size; i++)
    {
        auto bb_count = input_meta_data->get_bb_labels_batch()[i].size();        
        int labels_buf[bb_count];
        float coords_buf[bb_count*4];
        
        memcpy(labels_buf, input_meta_data->get_bb_labels_batch()[i].data(),  sizeof(int)*bb_count);
        memcpy(coords_buf, input_meta_data->get_bb_cords_batch()[i].data(), input_meta_data->get_bb_cords_batch()[i].size() * sizeof(BoundingBoxCord));
        
        //memcpy(keypoint_buf, input_meta_data->get_img_key_points_batch()[i][0].data(), sizeof(float) * input_meta_data->get_img_key_points_batch()[i][0].size());
        //std::cout<<"Copied keypoints to buffer:"<<std::endl;

        //std::cout<<"Keypoint values"<<std::endl;
        //for(uint i=0;i<9;i++)
        //{
            //std::cout<<keypoint_buf[i]<<std::endl;
        //}

        BoundingBoxCords bb_coords;
        BoundingBoxLabels bb_labels;
        ImageKeyPoints img_key_points;
        unsigned int num_keypoints=17;
        
        //Update Keypoints
        for (unsigned int object_index = 0; object_index < bb_count; object_index++)
        {
            KeyPoints key_points;

            for (unsigned int keypoint_index = 0; keypoint_index < num_keypoints; keypoint_index++)
            {
                KeyPoint key_point;
                float temp_x,temp_y;
                key_point=input_meta_data->get_img_key_points_batch()[i][object_index][keypoint_index];
            
                //Matrix multiplication of Affine matrix with keypoint values
                temp_x = (_affine_val[i*6+0]*key_point.x)+(_affine_val[i*6+1]*key_point.y)+(_affine_val[i*6+2]*1);
                temp_y = (_affine_val[i*6+3]*key_point.x)+(_affine_val[i*6+4]*key_point.y)+(_affine_val[i*6+5]*1);
                key_point.x=temp_x;
                key_point.y=temp_y;
                key_points.push_back(key_point);
            }
            img_key_points.push_back(key_points);
        }
        input_meta_data->get_img_key_points_batch()[i] = img_key_points; 

        for(uint j = 0, m = 0; j < bb_count; j++)
        {
            //std::cout<<"Computing BBox dot product:"<<std::endl;
            BoundingBoxCord box;
            float temp_l,temp_t,temp_r,temp_b;
            box.l = coords_buf[m++];
            box.t = coords_buf[m++];
            box.r = coords_buf[m++];
            box.b = coords_buf[m++];

            //Matrix multiplication of Afifine matrix with ltrb values
            temp_l = (_affine_val[i*6+0]*box.l)+(_affine_val[i*6+1]*box.t)+(_affine_val[i*6+2]*1);
            temp_t = (_affine_val[i*6+3]*box.l)+(_affine_val[i*6+4]*box.t)+(_affine_val[i*6+5]*1);
            temp_r = (_affine_val[i*6+0]*box.r)+(_affine_val[i*6+1]*box.b)+(_affine_val[i*6+2]*1);
            temp_b = (_affine_val[i*6+3]*box.r)+(_affine_val[i*6+4]*box.b)+(_affine_val[i*6+5]*1);
            
            box.l = temp_l;
            box.t = temp_t;
            box.r = temp_r;
            box.b = temp_b;

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
