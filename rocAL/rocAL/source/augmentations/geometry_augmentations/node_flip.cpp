/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:[]

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

#include <vx_ext_rpp.h>
#include "node_flip.h"
#include "exception.h"

FlipNode::FlipNode(const std::vector<rocalTensor *> &inputs,const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs),
        _horizontal(HORIZONTAL_RANGE[0], HORIZONTAL_RANGE[1]),
        _vertical (VERTICAL_RANGE[0], VERTICAL_RANGE[1])
{
}

void FlipNode::create_node()
{
    if(_node)
        return;

    _horizontal.create_array(_graph , VX_TYPE_UINT32, _batch_size);
    _vertical.create_array(_graph , VX_TYPE_UINT32, _batch_size);

    // if(_inputs[0]->info().layout() == RocalTensorlayout::NCHW)
    //     _layout = 1;
    // else if(_inputs[0]->info().layout() == RocalTensorlayout::NFHWC)
    //     _layout = 2;
    // else if(_inputs[0]->info().layout() == RocalTensorlayout::NFCHW)
    //     _layout = 3;

    if(_inputs[0]->info().roi_type() == RocalROIType::XYWH)
        _roi_type = 1;
    vx_scalar layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_layout);
    vx_scalar roi_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_roi_type);
    _node = vxExtrppNode_Flip(_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(), _horizontal.default_array(), _vertical.default_array(), layout, roi_type, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the brightness_batch (vxExtrppNode_BrightnessbatchPD) node failed: "+ TOSTR(status))
}

void FlipNode::init( int h_flag, int v_flag)
{
    _horizontal.set_param(h_flag);
    _vertical.set_param(v_flag);
    _layout = (int)_inputs[0]->info().layout();
    _roi_type = (int)_inputs[0]->info().roi_type();
    _horizontal_val.resize(_batch_size);
    _vertical_val.resize(_batch_size);

}

void FlipNode::init( IntParam* h_flag, IntParam* v_flag)
{
    _horizontal.set_param(core(h_flag));
    _vertical.set_param(core(v_flag));
    _layout = (int)_inputs[0]->info().layout();
    _roi_type = (int)_inputs[0]->info().roi_type();
    _horizontal_val.resize(_batch_size);
    _vertical_val.resize(_batch_size);
}


void FlipNode::update_node(MetaDataBatch* meta_data)
{
    _horizontal.update_array();
    _vertical.update_array();
    bool pose_estimation = true;
    vxCopyArrayRange(_horizontal.default_array(), 0, _batch_size, sizeof(int), _horizontal_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange(_vertical.default_array(), 0, _batch_size, sizeof(int), _vertical_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    if (pose_estimation)
    {
        unsigned int ann_count = meta_data->get_joints_data_batch().center_batch.size();
        for (unsigned int ann_index = 0; ann_index < ann_count; ann_index++)
        {
            if (_horizontal_val[ann_index] == 1)
            {
                float img_width = meta_data->get_img_sizes_batch()[ann_index].w;
                
                Joint joint0;
                JointVisibility joint0_visiblity;
                Joints joints;
                JointsVisibility joints_visibility;

                joint0 = meta_data->get_joints_data_batch().joints_batch[ann_index][0];

                
                joint0_visiblity = meta_data->get_joints_data_batch().joints_visibility_batch[ann_index][0];

                
                joint0[0] = (img_width - joint0[0] - 1) * (joint0_visiblity[0]);

                
                std::vector<float> center = meta_data->get_joints_data_batch().center_batch[ann_index];
                center[0] = img_width - center[0] - 1;

                
                joints.push_back(joint0);
                joints_visibility.push_back(joint0_visiblity);
                
                for (unsigned int joint_index = 1; joint_index < NUMBER_OF_JOINTS; joint_index = joint_index + 2)
                {
                    //std::cout<<"Flipping keypoints: "<<  joint_index<<" "<< joint_index+1<<std::endl;
                    Joint joint1, joint2;
                    JointVisibility joint1_visibility, joint2_visibility;
                    joint1 = meta_data->get_joints_data_batch().joints_batch[ann_index][joint_index];
                    joint2 = meta_data->get_joints_data_batch().joints_batch[ann_index][joint_index + 1];
                    joint1_visibility = meta_data->get_joints_data_batch().joints_visibility_batch[ann_index][joint_index];
                    joint2_visibility = meta_data->get_joints_data_batch().joints_visibility_batch[ann_index][joint_index + 1];

                    //Update the keypoint co-ordinates
                    joint1[0] = (img_width - joint1[0] - 1) * (joint1_visibility[0]);
                    joint2[0] = (img_width - joint2[0] - 1) * (joint2_visibility[0]);

                    joints.push_back(joint2);
                    joints.push_back(joint1);
                    joints_visibility.push_back(joint2_visibility);
                    joints_visibility.push_back(joint1_visibility);
                }
                meta_data->get_joints_data_batch().joints_batch[ann_index] = joints;
                meta_data->get_joints_data_batch().joints_visibility_batch[ann_index] = joints_visibility;
                meta_data->get_joints_data_batch().center_batch[ann_index] = center;
                joints.clear();
                joints_visibility.clear();
                center.clear();
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < _batch_size; i++)
        {
            auto bb_count = meta_data->get_bb_labels_batch()[i].size();
            BoundingBoxLabels labels_buf;
            BoundingBoxCords coords_buf;
            coords_buf.resize(bb_count);
            labels_buf.resize(bb_count);
            memcpy(labels_buf.data(), meta_data->get_bb_labels_batch()[i].data(), sizeof(int) * bb_count);
            memcpy(coords_buf.data(), meta_data->get_bb_cords_batch()[i].data(), meta_data->get_bb_cords_batch()[i].size() * sizeof(BoundingBoxCord));
            BoundingBoxCords bb_coords;

            for (uint j = 0; j < bb_count; j++)
            {
                if (_horizontal_val[i] == 1)
                {
                    float l = 1 - coords_buf[j].r;
                    coords_buf[j].r = 1 - coords_buf[j].l;
                    coords_buf[j].l = l;
                }
                if (_vertical_val[i] == 1)
                {
                    float t = 1 - coords_buf[j].b;
                    coords_buf[j].b = 1 - coords_buf[j].t;
                    coords_buf[j].t = t;
                }

                bb_coords.push_back(coords_buf[j]);
            }
            meta_data->get_bb_cords_batch()[i] = bb_coords;
            meta_data->get_bb_labels_batch()[i] = labels_buf;
        }
    } 

}

