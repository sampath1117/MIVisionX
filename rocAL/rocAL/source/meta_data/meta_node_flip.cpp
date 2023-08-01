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

#include "meta_node_flip.h"
void FlipMetaNode::initialize()
{
    _src_height_val.resize(_batch_size);
    _src_width_val.resize(_batch_size);
    _h_flag_val.resize(_batch_size);
    _v_flag_val.resize(_batch_size);
}
void FlipMetaNode::update_parameters(pMetaDataBatch input_meta_data, pMetaDataBatch output_meta_data)
{
    std::cerr<<"check flip.cpp\n ";
    initialize();
    if(_batch_size != input_meta_data->size())
    {
        _batch_size = input_meta_data->size();
    }
    auto input_roi = _node->get_src_roi();
    _h_flag = _node->get_horizontal_axis();
    _v_flag = _node->get_vertical_axis();
    vxCopyArrayRange((vx_array)_h_flag, 0, _batch_size, sizeof(int),_h_flag_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_v_flag, 0, _batch_size, sizeof(int),_v_flag_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    for(int i = 0; i < _batch_size; i++)
    {
        auto bb_count = input_meta_data->get_labels_batch()[i].size();
        BoundingBoxCords coords_buf = input_meta_data->get_bb_cords_batch()[i];
        Labels bb_labels = input_meta_data->get_labels_batch()[i];
        BoundingBoxCords bb_coords;
        for (uint j = 0; j < bb_count; j++)
        {
            if(_h_flag_val[i])
            {
                // float l = _src_width_val[i] - coords_buf[j].r;
                // coords_buf[j].r = _src_width_val[i] - coords_buf[j].l;
                // coords_buf[j].l = l; 
                // std::cerr<<"coords_buf[j].l "<<coords_buf[j].l<<" "<<coords_buf[j].r<<"  "<<_src_width_val[i]<<"\n ";
                auto l = coords_buf[j].l;
                auto r = coords_buf[j].r;
                coords_buf[j].l = input_roi[i].x2 - r;
                coords_buf[j].r = input_roi[i].x2 - l;    
                // std::cerr<<"\n after coords_buf[j].l "<<coords_buf[j].l<<" "<<coords_buf[j].r<<" \n";

            }
            if(_v_flag_val[i])
            {
                // float t = 1 - coords_buf[j].b;
                // coords_buf[j].b = 1 - coords_buf[j].t;
                // coords_buf[j].t = t;
                // float t = input_roi[i].y2 - coords_buf[j].b;
                // coords_buf[j].b = input_roi[i].y2 - coords_buf[j].t;
                // coords_buf[j].t = t;
                std::cerr<<"coords_buf[j].l "<<coords_buf[j].t<<" "<<coords_buf[j].b<<"  "<<_src_width_val[i]<<"\n ";

                auto t = coords_buf[j].t;
                auto b = coords_buf[j].b;
                coords_buf[j].t = input_roi[i].y2 - b;
                coords_buf[j].b = input_roi[i].y2 - t; 
                std::cerr<<"after coords_buf[j].l "<<coords_buf[j].t<<" "<<coords_buf[j].b<<"  "<<_src_width_val[i]<<"\n ";

            }
            
            bb_coords.push_back(coords_buf[j]);
        }
        output_meta_data->get_bb_cords_batch()[i] = bb_coords;
        output_meta_data->get_labels_batch()[i] = bb_labels;
    }
}
