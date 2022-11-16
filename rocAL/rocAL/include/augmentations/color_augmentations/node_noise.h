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

class NoiseTensorNode : public Node
{
public:
    NoiseTensorNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs);
    NoiseTensorNode() = delete;

    void init( float _noise_prob, float _salt_prob,float hue , float _noise_value,int _salt_value);
    void init( FloatParam* _noise_prob_param, FloatParam* _salt_prob_param,  FloatParam* _noise_value_param,  FloatParam* _salt_value_param, int seed );

protected:
    void create_node() override ;
    void update_node(MetaDataBatch* meta_data) override;
private:

    ParameterVX<float> _noise_prob;
    ParameterVX<float> _salt_prob;
    ParameterVX<float> _noise_value;
    ParameterVX<float> _salt_value;

    unsigned _layout=0, _roi_type=0;
    int _seed;
    constexpr static float NOISE_PROB_RANGE [2] = {0.1, 1};
    constexpr static float SALT_PROB_RANGE [2] = {0.1, 1};
    constexpr static float NOISE_RANGE [2] = {0, 0.5};
    constexpr static float SALT_RANGE [2] = {0.1, 1};
};
