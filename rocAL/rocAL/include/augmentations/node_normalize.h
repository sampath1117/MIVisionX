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

class NormalizeNode : public Node
{
public:
    NormalizeNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs);
    NormalizeNode() = delete;
    void init(float mean, float std_dev, std::vector<int> axes, bool batch, float scale, float shift, int ddof, float epsilon);

protected:
    void create_node() override;
    void update_node() override;

private:
    float _mean, _std_dev, _scale, _shift, _epsilon;
    int _ddof, _axis_mask = 0;
    std::vector<int> _axes;
    bool _batch, _calculate_mean, _calculate_stddev;
    std::vector<int> _src_frames, _src_channels;
    vx_array _src_frames_array, _src_channels_array;
    unsigned _num_of_dims;
    std::vector<std::vector<unsigned>> _param_shape;
    std::vector<unsigned> _dst_roi_width_vec, _dst_roi_height_vec;
};