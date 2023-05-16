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

#pragma once
#include "node.h"
#include "graph.h"
#include "rocal_api_types.h"

class ResampleNode : public Node
{
public:
    ResampleNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs);
    ResampleNode() = delete;
    void init(RocalTensor resample_rate, float quality);

protected:
    void create_node() override;
    void update_node() override;
private:
    RocalTensor _resample_rate;
    float _quality, _scale_ratio;
    float* _out_sample_rate_array;
    uint _max_dst_width, _resample_rate_dims;
    std::vector<float> _resample_rate_vec;
    vx_scalar _max_dst_width_scalar;
    vx_array _src_frames_array = nullptr, _src_channels_array = nullptr, _src_sample_rate_array ;
    std::vector<unsigned> _dst_roi_width_vec, _dst_roi_height_vec, _src_frames, _src_channels;

};