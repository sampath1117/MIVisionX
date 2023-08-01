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

#include "meta_node_resize.h"
void ResizeMetaNode::initialize()
{
    _src_height_val.resize(_batch_size);
    _src_width_val.resize(_batch_size);
    _dst_width_val.resize(_batch_size);
    _dst_height_val.resize(_batch_size);
}
void ResizeMetaNode::update_parameters(pMetaDataBatch input_meta_data, pMetaDataBatch output_meta_data)
{
    initialize();
    std::cerr<<"\n check metanode update parameters ";
    if(_batch_size != input_meta_data->size())
    {
        _batch_size = input_meta_data->size();
    }
    // _src_width = _node->get_dst_width();
    // _src_height = _node->get_dst_height();
    
    // vxCopyArrayRange((vx_array)_src_width, 0, _batch_size, sizeof(uint),_src_width_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    // vxCopyArrayRange((vx_array)_src_height, 0, _batch_size, sizeof(uint),_src_height_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    auto input_roi = _node->get_src_roi();
    auto output_roi = _node->get_dst_roi();
    // _dst_width = _node->get_dst_width();
    // _dst_height = _node->get_dst_height();
    // _dst_width = _node->get_src_width();
    // _dst_height = _node->get_src_height();
    // vxCopyArrayRange((vx_array)_dst_width, 0, _batch_size, sizeof(uint), _dst_width_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    // vxCopyArrayRange((vx_array)_dst_height, 0, _batch_size, sizeof(uint), _dst_height_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    for (int i = 0; i < _batch_size; i++)
    {
        _dst_to_src_width_ratio = float(output_roi[i].x2) / float(input_roi[i].x2);
        _dst_to_src_height_ratio = float(output_roi[i].y2) / float(input_roi[i].y2);
        std::cerr<<"_dst_width_val[i] / float(input_roi[i].x2 "<<float(output_roi[i].x2) <<"  "<< float(input_roi[i].x2);
        std::cerr<<"_dst_width_val[i] / float(input_roi[i].x2 "<<float(output_roi[i].y2) <<"  "<< float(input_roi[i].y2);
        unsigned bb_count = input_meta_data->get_labels_batch()[i].size();
        BoundingBoxCords coords_buf = input_meta_data->get_bb_cords_batch()[i];
        std::cerr<<"\n check in meta node ";
        Labels labels_buf = input_meta_data->get_labels_batch()[i];   
        BoundingBoxCords bb_coords;
        BoundingBoxCord temp_box;     
        Labels bb_labels;
        temp_box.l = temp_box.t = temp_box.r = temp_box.b = 0;
        for (uint j = 0; j < bb_count; j++)
        {
            coords_buf[j].l *= _dst_to_src_width_ratio;
            coords_buf[j].t *= _dst_to_src_height_ratio;
            coords_buf[j].r *= _dst_to_src_width_ratio;
            coords_buf[j].b *= _dst_to_src_height_ratio;
            bb_coords.push_back(coords_buf[j]);
            bb_labels.push_back(labels_buf[j]);
        }
        if (bb_coords.size() == 0)
        {
            bb_coords.push_back(temp_box);
        }
        output_meta_data->get_bb_cords_batch()[i] = bb_coords;
        output_meta_data->get_labels_batch()[i] = bb_labels;
    }
}