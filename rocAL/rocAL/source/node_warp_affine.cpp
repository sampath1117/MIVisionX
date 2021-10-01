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

#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include "node_warp_affine.h"
#include "exception.h"

void get_dir(float point[], float dir[], float r);
void get_3rd_point(float mat[][2]);

WarpAffineNode::WarpAffineNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _x0(COEFFICIENT_RANGE_1[0], COEFFICIENT_RANGE_1[1]),
        _x1(COEFFICIENT_RANGE_0[0], COEFFICIENT_RANGE_0[1]),
        _y0(COEFFICIENT_RANGE_0[0], COEFFICIENT_RANGE_0[1]),
        _y1(COEFFICIENT_RANGE_1[0], COEFFICIENT_RANGE_1[1]),
        _o0(COEFFICIENT_RANGE_OFFSET[0], COEFFICIENT_RANGE_OFFSET[1]),
        _o1(COEFFICIENT_RANGE_OFFSET[0], COEFFICIENT_RANGE_OFFSET[1])
{
}

void WarpAffineNode::create_node()
{
    if(_node)
        return;

    vx_status width_status, height_status;
    _affine.resize(6 * _batch_size);

    uint batch_size = _batch_size;
    for (uint i=0; i < batch_size; i++ )
    {
         _affine[i*6 + 0] = _x0.renew();
         _affine[i*6 + 1] = _y0.renew();
         _affine[i*6 + 2] = _x1.renew();
         _affine[i*6 + 3] = _y1.renew();
         _affine[i*6 + 4] = _o0.renew();
         _affine[i*6 + 5] = _o1.renew();

    }
    _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    std::vector<uint32_t> dst_roi_width(_batch_size,_outputs[0]->info().width());
    std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().height_single());
    width_status = vxAddArrayItems(_dst_roi_width, _batch_size, dst_roi_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_dst_roi_height, _batch_size, dst_roi_height.data(), sizeof(vx_uint32));
    if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the rotate (vxExtrppNode_WarpAffinePD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))

    vx_status status;
    _affine_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size * 6);
    status = vxAddArrayItems(_affine_array,_batch_size * 6, _affine.data(), sizeof(vx_float32));
    _node = vxExtrppNode_WarpAffinebatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _dst_roi_width, _dst_roi_height,
                                           _affine_array, _batch_size);
    
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the warp affine (vxExtrppNode_WarpAffinePD) node failed: "+ TOSTR(status))
}

void WarpAffineNode::update_affine_array()
{
    bool pose_estimation = false;

    if(pose_estimation)
    {
        //Start of Half body transform
        auto pixel_std = PIXEL_STD;
        auto half_body_constant = SCALE_CONSTANT_HALF_BODY;
        auto scale_factor = 0.35;
        auto rotation_factor = 0.0;
        float output_size[2] = {288.0, 384.0};
        float pi = 3.14;
        auto kps = NUMBER_OF_KEYPOINTS;

        for (int i = 0; i < _meta_data_info->size(); i++)
        {
            ImageJointsData img_joints_data;
            JointsData joints_data;
            auto bb_count = _meta_data_info->get_bb_labels_batch()[i].size();
            float src[3][2] = {0.0};
            float dst[3][2] = {0.0};
            float src_dir[2] = {0.0};
            float dst_dir[2] = {0.0};
            float shift[2] = {0.0};

            for (uint bb_idx = 0; bb_idx < bb_count; bb_idx++)
            {
                joints_data = _meta_data_info->get_img_joints_data_batch()[i][bb_idx];
                BoundingBoxCenter box_center;
                BoundingBoxScale box_scale;
                KeyPoints upper_joints;
                KeyPoints lower_joints;
                KeyPoints selected_joints;
                KeyPoint key_point;
                KeyPointsVisibility key_points_visibility = joints_data.joints_visility;

                //Seperate the keypoints into upper body and lower body
                for (uint kp_idx = 0; kp_idx < kps; kp_idx++)
                {
                    auto v = key_points_visibility[kp_idx].xv;
                    if (v > 0)
                    {
                        key_point = joints_data.joints[kp_idx];
                        if (kp_idx <= 10)
                        {
                            upper_joints.push_back(key_point);
                        }
                        else
                        {
                            lower_joints.push_back(key_point);
                        }
                    }
                }

                //Any of lower body/upper body joints should have minimum 3 joints
                if (lower_joints.size() < 3 && upper_joints.size() < 3)
                {
                    return;
                }

                //select any of upper body and lower body joints
                selected_joints = upper_joints;
                if (lower_joints.size() > 2) //add random factor here
                {
                    selected_joints = lower_joints;
                }

                auto mean_center_x = 0.0;
                auto mean_center_y = 0.0;
                auto min_idx = 0;
                auto max_idx = 0;
                for (uint kp_idx = 0; kp_idx < selected_joints.size(); kp_idx++)
                {
                    mean_center_x = mean_center_x + selected_joints[kp_idx].x;
                    mean_center_y = mean_center_y + selected_joints[kp_idx].y;

                    if (selected_joints[kp_idx].x < selected_joints[min_idx].x)
                    {
                        min_idx = kp_idx;
                    }

                    if (selected_joints[kp_idx].x > selected_joints[max_idx].x)
                    {
                        max_idx = kp_idx;
                    }
                }
                box_center.xc = mean_center_x / selected_joints.size();
                box_center.yc = mean_center_y / selected_joints.size();

                //Calculate the scale value
                float left_top[] = {selected_joints[min_idx].x, selected_joints[min_idx].y};
                float right_bottom[] = {selected_joints[max_idx].x, selected_joints[max_idx].y};
                float w, h;
                w = right_bottom[0] - left_top[0];
                h = right_bottom[1] - left_top[1];

                float aspect_ratio = output_size[0] * 1.0 / output_size[1];
                if (w > aspect_ratio * h)
                {
                    h = w * 1.0 / aspect_ratio;
                }
                else if (w < aspect_ratio * h)
                {
                    w = aspect_ratio * h;
                }

                box_scale.ws = (half_body_constant * w * 1.0) / pixel_std;
                box_scale.hs = (half_body_constant * h * 1.0) / pixel_std;
                //End of Half body transform

                //Multiply scale with random scale factor clipped between [1-sf,1+sf]
                box_scale.ws = box_scale.ws * scale_factor;
                box_scale.hs = box_scale.hs * scale_factor;

                //Get random rotation factor
                auto r = rotation_factor;
                joints_data.rotation = r;

                //Generate the affine matrix based on this values

                //Get the correct scale values
                float scale_temp[2] = {pixel_std * box_scale.ws, pixel_std * box_scale.hs};
                float src_w = scale_temp[0];
                float dst_w = output_size[0]*1.0;
                float dst_h = output_size[1]*1.0;
                auto r_rad = pi * r / 180;

                std::cout << "src_w: " << src_w << std::endl;
                std::cout << "dst_w: " << dst_w << std::endl;
                std::cout << "dst_h: " << dst_h << std::endl;

                float src_point[2] = {0.0, src_w * -0.5};
                float dst_point[2] = {0.0, dst_w * -0.5};

                get_dir(src_point, src_dir, r_rad);
                get_dir(dst_point, dst_dir, r_rad);
            
                src[0][0] = box_center.xc + (shift[0] * scale_temp[0]);
                src[0][1] = box_center.yc + (shift[1] * scale_temp[1]);
                src[1][0] = box_center.xc + src_dir[0] + (shift[0] * scale_temp[0]);
                src[1][1] = box_center.yc + src_dir[1] + (shift[1] * scale_temp[1]);

                dst[0][0] = dst_w * 0.5;
                dst[0][1] = dst_h * 0.5;
                dst[1][0] = dst_w * 0.5 + dst_dir[0] + (shift[0] * scale_temp[0]);
                dst[1][1] = dst_h * 0.5 + dst_dir[1] + (shift[1] * scale_temp[1]);

                get_3rd_point(src);
                get_3rd_point(dst);

                //Calculate the affine

                //Copy these values to local bb_centers,bb_scales
                joints_data.center = box_center;
                joints_data.scale = box_scale;
                img_joints_data.push_back(joints_data);

                //Clear the keypoint values
                upper_joints.clear();
                lower_joints.clear();
                selected_joints.clear();
            }
            _meta_data_info->get_img_joints_data_batch()[i] = img_joints_data;
        }
    }
    
    for (uint i = 0; i < _batch_size; i++ )
    {
        //std::cout<<"Original array:"<<std::endl;
        _affine[i*6 + 0] = _x0.renew();
        _affine[i*6 + 1] = _y0.renew();
        _affine[i*6 + 2] = _x1.renew();
        _affine[i*6 + 3] = _y1.renew();
        _affine[i*6 + 4] = _o0.renew();
        _affine[i*6 + 5] = _o1.renew();
        //std::cout<<_affine[i*6 + 0]<<" "<<_affine[i*6 + 1]<<" "<<_affine[i*6 + 2]<<std::endl;
        //std::cout<<_affine[i*6 + 3]<<" "<<_affine[i*6 + 4]<<" "<<_affine[i*6 + 5]<<std::endl;
    }
    vx_status affine_status;
    affine_status = vxCopyArrayRange((vx_array)_affine_array, 0, _batch_size * 6, sizeof(vx_float32), _affine.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST); //vxAddArrayItems(_width_array,_batch_size, _width, sizeof(vx_uint32));
    if(affine_status != 0)
        THROW(" vxCopyArrayRange failed in the WarpAffine(vxExtrppNode_WarpAffinePD) node: "+ TOSTR(affine_status))
}

void WarpAffineNode::init(float x0, float x1, float y0, float y1, float o0, float o1)
{
    _x0.set_param(x0);
    _x1.set_param(x1);
    _y0.set_param(y0);
    _y1.set_param(y1);
    _o0.set_param(o0);
    _o1.set_param(o1);
}

void WarpAffineNode::init(FloatParam* x0, FloatParam* x1, FloatParam* y0, FloatParam* y1, FloatParam* o0, FloatParam* o1)
{
    _x0.set_param(core(x0));
    _x1.set_param(core(x1));
    _y0.set_param(core(y0));
    _y1.set_param(core(y1));
    _o0.set_param(core(o0));
    _o1.set_param(core(o1));
}

void WarpAffineNode::update_node()
{
    update_affine_array();
}

void get_dir(float point[], float dir[], float r)
{
    // std::cout<<"direction array before:"<<std::endl;
    // std::cout<<dir[0]<<" "<<dir[1]<<std::endl;
    float sn = std::sin(r);
    float cs = std::cos(r);
    dir[0] = point[0] * cs - point[1] * sn;
    dir[1] = point[0] * sn + point[1] * cs;
    // std::cout<<"direction array after:"<<std::endl;
    // std::cout<<dir[0]<<" "<<dir[1]<<std::endl;
}
void get_3rd_point(float mat[][2])
{
    // std::cout << "matrix before:" << std::endl
    // << mat[0][0] << " " << mat[0][1] << std::endl
    // << mat[1][0] << " " << mat[1][1] << std::endl
    // << mat[2][0] << " " << mat[2][1] << std::endl;

    float direct[2] = {mat[0][0] - mat[1][0], mat[0][1] - mat[1][1]};
    mat[2][0] = mat[1][0] - direct[1];
    mat[2][1] = mat[1][1] + direct[0];

    // std::cout << "matrix after:" << std::endl
    // << mat[0][0] << " " << mat[0][1] << std::endl
    // << mat[1][0] << " " << mat[1][1] << std::endl
    // << mat[2][0] << " " << mat[2][1] << std::endl;
}

 