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

#include "meta_node_warp_affine.h"
void get_dir(float point[], float dir[], float r);
void get_3rd_point(float mat[][3]);
void get_inverse(float m[][3], float inv_m[3][3]);
void matrix_mult(float src[2][3], float dst[3][3], float affine[2][3]);

void WarpAffineMetaNode::initialize()
{
    _src_height_val.resize(_batch_size);
    _src_width_val.resize(_batch_size);
    _affine_val.resize(6 * _batch_size);
}
void WarpAffineMetaNode::update_parameters(MetaDataBatch *input_meta_data, bool pose_estimation)
{
    //std::cout<<"Warp affine meta node is called:"<<std::endl;

    initialize();

    if (_batch_size != input_meta_data->size())
    {
        _batch_size = input_meta_data->size();
    }

    _src_width = _node->get_src_width();
    _src_height = _node->get_src_height();
    _affine_array = _node->get_affine_array();

    vxCopyArrayRange((vx_array)_src_width, 0, _batch_size, sizeof(uint), _src_width_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_src_height, 0, _batch_size, sizeof(uint), _src_height_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_affine_array, 0, (6 * _batch_size), sizeof(float), _affine_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    if (pose_estimation)
    {
        //std::cout << "Entered update affine array function" << std::endl;

        //Start of Half body transform
        auto scale_factor = 0.35;
        auto rotation_factor = 0.0;
        float output_size[2] = {288.0, 384.0};
        float pi = 3.14;

        //std::cout << "Enter the affine calculation code" << std::endl;
        int ann_count = input_meta_data->get_joints_data_batch().image_id_batch.size();
        std::vector<int> UPPER_BODY_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<int> LOWER_BODY_IDS = {11, 12, 13, 14, 15, 16};
        for (int i = 0; i < ann_count; i++)
        {
            float src[2][3] = {0.0};
            float dst[2][3] = {1.0};
            float dst_padded[3][3] = {1.0};
            float inv_dst[3][3];
            float src_dir[2] = {0.0};
            float dst_dir[2] = {0.0};
            float shift[2] = {0.0};
            float affine_matrix[2][3];
            //std::cout << "Initialized matrices needed" << std::endl;

            Center box_center;
            Scale box_scale;
            Joints upper_joints, lower_joints, selected_joints;
            JointsVisibility joints_visibility = input_meta_data->get_joints_data_batch().joints_visibility_batch[i];
            Joint joint;

            box_center = input_meta_data->get_joints_data_batch().center_batch[i];
            box_scale = input_meta_data->get_joints_data_batch().scale_batch[i];

            //Seperate the keypoints into upper body and lower body
            for (uint kp_idx = 0; kp_idx < NUMBER_OF_KEYPOINTS; kp_idx++)
            {
                auto v = joints_visibility[kp_idx][0];
                if (v > 0)
                {
                    joint = input_meta_data->get_joints_data_batch().joints_batch[i][kp_idx];
                    if (kp_idx <= 10)
                    {
                        upper_joints.push_back(joint);
                    }
                    else
                    {
                        lower_joints.push_back(joint);
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
                mean_center_x = mean_center_x + selected_joints[kp_idx][0];
                mean_center_y = mean_center_y + selected_joints[kp_idx][1];

                if (selected_joints[kp_idx][0] < selected_joints[min_idx][0])
                {
                    min_idx = kp_idx;
                }

                if (selected_joints[kp_idx][0] > selected_joints[max_idx][0])
                {
                    max_idx = kp_idx;
                }
            }
            box_center[0] = mean_center_x / selected_joints.size();
            box_center[1] = mean_center_y / selected_joints.size();

            //Calculate the scale value
            float left_top[] = {selected_joints[min_idx][0], selected_joints[min_idx][1]};
            float right_bottom[] = {selected_joints[max_idx][0], selected_joints[max_idx][1]};
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

            box_scale[0] = (SCALE_CONSTANT_HALF_BODY * w * 1.0) / PIXEL_STD;
            box_scale[1] = (SCALE_CONSTANT_HALF_BODY * h * 1.0) / PIXEL_STD;
            //End of Half body transform

            //Multiply scale with random scale factor clipped between [1-sf,1+sf]
            box_scale[0] = box_scale[0] * scale_factor;
            box_scale[1] = box_scale[1] * scale_factor;

            //Get random rotation factor
            auto r = rotation_factor;
            input_meta_data->get_joints_data_batch().rotation_batch[i] = r;
            

            //Generate the affine matrix based on this values

            //Get the correct scale values
            float scale_temp[2] = {PIXEL_STD * box_scale[0], PIXEL_STD * box_scale[1]};
            float src_w = scale_temp[0];
            float dst_w = output_size[0] * 1.0;
            float dst_h = output_size[1] * 1.0;
            auto r_rad = pi * r / 180;

            // std::cout << "src_w: " << src_w << std::endl;
            // std::cout << "dst_w: " << dst_w << std::endl;
            // std::cout << "dst_h: " << dst_h << std::endl<<std::endl;

            float src_point[2] = {0.0, src_w * -0.5};
            float dst_point[2] = {0.0, dst_w * -0.5};
            // std::cout << "src_point" << std::endl<< src_point[0] << " " << src_point[1] << std::endl<<std::endl;

            //std::cout << "Calculated the source and destination point" << std::endl;

            get_dir(src_point, src_dir, r_rad);
            get_dir(dst_point, dst_dir, r_rad);

            //std::cout << "Got the direction" << std::endl;
            // std::cout << "xc,yc:" << box_center[0] <<" "<< box_center[1] << std::endl<<std::endl;
            src[0][0] = box_center[0] + (shift[0] * scale_temp[0]);
            src[1][0] = box_center[1] + (shift[1] * scale_temp[1]);
            src[0][1] = box_center[0] + src_dir[0] + (shift[0] * scale_temp[0]);
            src[1][1] = box_center[1] + src_dir[1] + (shift[1] * scale_temp[1]);

            // std::cout << "src matrix:" << std::endl
            //           << src[0][0] << " " << src[0][1] << std::endl
            //           << src[1][0] << " " << src[1][1] << std::endl<<std::endl;

            dst[0][0] = dst_padded[0][0] = dst_w * 0.5;
            dst[1][0] = dst_padded[1][0] = dst_h * 0.5;
            dst[0][1] = dst_padded[0][1] = dst_w * 0.5 + dst_dir[0] + (shift[0] * scale_temp[0]);
            dst[1][1] = dst_padded[1][1] = dst_h * 0.5 + dst_dir[1] + (shift[1] * scale_temp[1]);
            dst_padded[2][0] = 1.0;
            dst_padded[2][1] = 1.0;
            dst_padded[2][2] = 1.0;

            //Get the 3rd point
            get_3rd_point(src);
            get_3rd_point(dst);

            dst_padded[0][2] = dst[0][2];
            dst_padded[1][2] = dst[1][2];

            // std::cout << "dst matrix:" << std::endl
            // << dst_padded[0][0] << " " << dst_padded[0][1] << " " << dst_padded[0][2] << std::endl
            // << dst_padded[1][0] << " " << dst_padded[1][1] << " " << dst_padded[1][2] << std::endl
            // << dst_padded[2][0] << " " << dst_padded[2][1] << " " << dst_padded[2][2] << std::endl<<std::endl;;

            //Get the inverse matrix
            get_inverse(dst_padded, inv_dst);
            // std::cout << "inverse matrix:" << std::endl
            //           << inv_dst[0][0] << " " << inv_dst[0][1] << " " << inv_dst[0][2] << std::endl
            //           << inv_dst[1][0] << " " << inv_dst[1][1] << " " << inv_dst[1][2] << std::endl
            //           << inv_dst[2][0] << " " << inv_dst[2][1] << " " << inv_dst[2][2] << std::endl;

            //Get the affine array
            matrix_mult(src, inv_dst, affine_matrix);

            // std::cout << "affine matrix:" << std::endl
            //           << affine_matrix[0][0] << " " << affine_matrix[0][1] << " " << affine_matrix[0][2] << std::endl
            //           << affine_matrix[1][0] << " " << affine_matrix[1][1] << " " << affine_matrix[1][2] << std::endl<<std::endl;;

            //Copy these values to local bb_centers,bb_scales
            input_meta_data->get_joints_data_batch().center_batch[i] =  box_center;
            input_meta_data->get_joints_data_batch().scale_batch[i] = box_scale;

            //Clear the keypoint values
            upper_joints.clear();
            lower_joints.clear();
            selected_joints.clear();
        }

        //Update Keypoints
        for (unsigned int ann_index = 0; ann_index < ann_count; ann_index++)
        {
            Joints joints;
            for (unsigned int keypoint_index = 0; keypoint_index < NUMBER_OF_KEYPOINTS; keypoint_index++)
            {
                Joint joint;
                float temp_x, temp_y;
                joint = input_meta_data->get_joints_data_batch().joints_batch[ann_index][keypoint_index];

                //Matrix multiplication of Affine matrix with keypoint values
                temp_x = (_affine_val[ann_index * 6 + 0] * joint[0]) + (_affine_val[ann_index * 6 + 1] * joint[1]) + (_affine_val[ann_index * 6 + 2] * 1);
                temp_y = (_affine_val[ann_index * 6 + 3] * joint[0]) + (_affine_val[ann_index * 6 + 4] * joint[1]) + (_affine_val[ann_index * 6 + 5] * 1);
                joint[0] = temp_x;
                joint[1] = temp_y;
                joints.push_back(joint);
            }
            input_meta_data->get_joints_data_batch().joints_batch[ann_index] = joints;
            joints.clear();
        }
    }
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
void get_3rd_point(float mat[][3])
{
    // std::cout << "matrix before:" << std::endl
    // << mat[0][0] << " " << mat[0][1] << " " << mat[0][2] << std::endl
    // << mat[1][0] << " " << mat[1][1] << " " << mat[1][2] << std::endl;

    float direct[2] = {mat[0][0] - mat[0][1], mat[1][0] - mat[1][1]};
    mat[0][2] = mat[0][1] - direct[1];
    mat[1][2] = mat[1][1] + direct[0];

    // std::cout << "matrix after:" << std::endl
    // << mat[0][0] << " " << mat[0][1] << " " << mat[0][2] << std::endl
    // << mat[1][0] << " " << mat[1][1] << " " << mat[1][2] << std::endl;
}

void get_inverse(float m[3][3], float inv_m[3][3])
{
    float det = 0.0;

    //Calculate determinant of the matrix
    for (int i = 0; i < 3; i++)
    {
        det = det + (m[0][i] * (m[1][(i + 1) % 3] * m[2][(i + 2) % 3] - m[1][(i + 2) % 3] * m[2][(i + 1) % 3]));
    }

    if (det == 0)
    {
        return;
    }

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            inv_m[j][i] = ((m[(i + 1) % 3][(j + 1) % 3] * m[(i + 2) % 3][(j + 2) % 3]) - (m[(i + 1) % 3][(j + 2) % 3] * m[(i + 2) % 3][(j + 1) % 3])) / det;
        }
    }

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            //std::cout<<inv_m[j][i]<<" ";
        }
        //std::cout<<std::endl;
    }
}

void matrix_mult(float src[2][3], float dst[3][3], float affine[2][3])
{
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                affine[i][j] += src[i][k] * dst[k][j];
            }
        }
    }

    //Print the affine matrix generated
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            //std::cout<<affine[i][j]<<" ";
        }
        //std::cout<<std::endl;
    }
}