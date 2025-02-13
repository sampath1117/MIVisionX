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
#include <set>
#include <memory>
#include <algorithm>
#include "bounding_box_graph.h"
#include "meta_data.h"
#include "node.h"
#include "node_rotate.h"
#include "parameter_vx.h"
#define PI 3.14159265
#define RAD(deg) (deg * PI / 180)

class RotateMetaNode:public MetaNode
{
    public:
        RotateMetaNode() {};
        void update_parameters(MetaDataBatch* input_meta_data)override;
        std::shared_ptr<RotateNode> _node = nullptr;
    private:
        void initialize();
        vx_array _src_width, _src_height;
        std::vector<uint> _src_width_val, _src_height_val;
        unsigned int _dst_width, _dst_height;
        vx_array _angle;
        std::vector<float> _angle_val;
};
