/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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
#include "parameter_factory.h"
#include "parameter_vx.h"
#include "graph.h"

class FlipNode : public Node
{
public:
    FlipNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs);
    FlipNode() = delete;

    void init( int h_flag, int v_flag);
    void init( IntParam* h_flag_param, IntParam* v_flag_param);

protected:
    void create_node() override ;
    void update_node(MetaDataBatch* meta_data) override;
private:

    ParameterVX<int> _horizontal;
    ParameterVX<int> _vertical;
    unsigned _layout, _roi_type;
    constexpr static int HORIZONTAL_RANGE [2] = {0, 2};
    constexpr static int   VERTICAL_RANGE [2] = {0, 2};
    std::vector<int> _horizontal_val, _vertical_val;
};