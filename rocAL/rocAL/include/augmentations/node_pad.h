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

class PadNode : public Node
{
public:
    PadNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs);
    PadNode() = delete;
    void init(float fill_value);

protected:
    void create_node() override;
    void update_node() override;

private:
    float _fill_value;
    bool _normalized_anchor = false;
    bool _normalized_shape = false;
    RocalOutOfBoundsPolicy _policy = RocalOutOfBoundsPolicy::PAD;
    int _axis_mask = 0;
    vx_array _fill_values_array;
    vx_tensor _anchors_tensor, _shapes_tensor;
    std::vector<float> _anchor_vec, _shape_vec, _fill_values_vec;
    unsigned _num_of_dims;
    vx_size * _stride;
};