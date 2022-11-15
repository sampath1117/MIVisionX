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

class ContrastNode : public Node
{
public:
    ContrastNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs);
    ContrastNode() = delete;

    void init( float c_factor, float c_center);
    void init( FloatParam* c_factor_param, FloatParam* c_center_param);

protected:
    void create_node() override ;
    void update_node() override;
private:

    ParameterVX<float> _factor;
    ParameterVX<float> _center;
    unsigned _layout=0, _roi_type=0;
    constexpr static float FACTOR_RANGE [2] = {0.1, 3.0};
    constexpr static float   CENTER_RANGE [2] = {0, 128};
};