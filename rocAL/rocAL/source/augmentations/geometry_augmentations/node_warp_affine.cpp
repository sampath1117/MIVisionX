#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include "node_warp_affine.h"
#include "exception.h"


WarpAffineNode::WarpAffineNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs) :
        Node(inputs, outputs),
        _x0(COEFFICIENT_RANGE_1[0], COEFFICIENT_RANGE_1[1]),
        _x1(COEFFICIENT_RANGE_0[0], COEFFICIENT_RANGE_0[1]),
        _y0(COEFFICIENT_RANGE_0[0], COEFFICIENT_RANGE_0[1]),
        _y1(COEFFICIENT_RANGE_1[0], COEFFICIENT_RANGE_1[1]),
        _o0(COEFFICIENT_RANGE_OFFSET[0], COEFFICIENT_RANGE_OFFSET[1]),
        _o1(COEFFICIENT_RANGE_OFFSET[0], COEFFICIENT_RANGE_OFFSET[1]),
        _random_value(RANDOM_VALUE_RANGE[0], RANDOM_VALUE_RANGE[1])
{
}

void WarpAffineNode::create_node() {
    if(_node)
        return;

    vx_status width_status, height_status;
    _affine.resize(6 * _batch_size);
    _inv_affine.resize(6 * _batch_size);
    uint batch_size = _batch_size;
    for (uint i=0; i < batch_size; i++ ) {
         _affine[i*6 + 0] = _x0.renew();
         _affine[i*6 + 1] = _y0.renew();
         _affine[i*6 + 2] = _x1.renew();
         _affine[i*6 + 3] = _y1.renew();
         _affine[i*6 + 4] = _o0.renew();
         _affine[i*6 + 5] = _o1.renew();
    }

    vx_status status;
    _affine_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size * 6);

    status = vxAddArrayItems(_affine_array,_batch_size * 6, _affine.data(), sizeof(vx_float32));
    if(_inputs[0]->info().roi_type() == RocalROIType::XYWH)
        _roi_type = 1;
    vx_scalar layout = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_layout);
    vx_scalar roi_type = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_roi_type);
    vx_scalar interpolation = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&_interpolation_type);
    _node = vxExtrppNode_WarpAffine (_graph->get(), _inputs[0]->handle(), _src_tensor_roi, _outputs[0]->handle(), _affine_array,interpolation, layout, roi_type, _batch_size);

    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the warp affine (vxExtrppNode_WarpAffinePD) node failed: "+ TOSTR(status))
}

void WarpAffineNode::update_affine_array() {
    bool _is_set_meta_data = true;
    if (_is_set_meta_data)
    {
        int ann_count = _meta_data_info->get_joints_data_batch().image_id_batch.size();
        int output_size[2] = {(int)(_outputs[0]->info().max_dims()).at(0), (int)(_outputs[0]->info().max_dims()).at(1)};

        float pi = 3.14;
        std::vector<float> rotate_probability;
        std::vector<float> half_body_probability;

        for (int i = 0; i < ann_count; i++)
        {
            float src[2][3] = {0.0};
            float dst[2][3] = {1.0};
            float dst_padded[3][3] = {1.0};
            float inv_dst[3][3] = {0.0};
            float src_dir[2] = {0.0};
            float dst_dir[2] = {0.0};
            float shift[2] = {0.0};
            float *affine_matrix, *inverse_affine_matrix, rotate_deg;
            std::vector<float> box_center;
            std::vector<float> box_scale;

            //Get the variables needed for affine calculation
            affine_matrix = _affine.data() + (i * 6);
            inverse_affine_matrix = _inv_affine.data() + (i * 6);
            box_center = _meta_data_info->get_joints_data_batch().center_batch[i];
            box_scale = _meta_data_info->get_joints_data_batch().scale_batch[i];
            rotate_deg = _meta_data_info->get_joints_data_batch().rotation_batch[i];
            int input_img_width = _meta_data_info->get_img_sizes_batch()[i].w;
            int input_img_height = _meta_data_info->get_img_sizes_batch()[i].h;

            // std::cout<<"imageID: "<<_meta_data_info->get_joints_data_batch().image_id_batch[i]<<std::endl;
            if (_is_train)
            {
                if (_half_body_probability > 0 && _random_value.renew() < _half_body_probability)
                {
                    float aspect_ratio = output_size[0] * 1.0 / output_size[1];
                    half_body_transform(i, box_center, box_scale, aspect_ratio);
                }

                float scale_factor = _random_value.renew()*_scale_factor + 1;
                scale_factor = std::clamp(scale_factor, 1-_scale_factor, 1+_scale_factor);
                box_scale[0] = box_scale[0] * scale_factor;
                box_scale[1] = box_scale[1] * scale_factor;
                // std::cout<<"scale factor: "<< scale_factor<<std::endl;

                if (_rotate_probability > 0 && _random_value.renew() < _rotate_probability)
                {
                    rotate_deg = std::clamp(_random_value.renew()*_rotation_factor, -2*_rotation_factor, 2*_rotation_factor);
                }
                // std::cout<<"rotation factor: "<< rotate_deg<<std::endl;
            }

            //Get the correct scale values
            float scale_temp[2] = {PIXEL_STD * box_scale[0], PIXEL_STD * box_scale[1]};
            float src_w = scale_temp[0];
            float dst_w = output_size[0] * 1.0;
            float dst_h = output_size[1] * 1.0;
            float r_rad = pi * rotate_deg / 180.0;

            float src_point[2] = {0.0, -src_w / 2};
            dst_dir[1]  = -dst_w / 2;
            get_dir(src_point, src_dir, r_rad);

            src[0][0] = box_center[0] + (shift[0] * scale_temp[0]);
            src[1][0] = box_center[1] + (shift[1] * scale_temp[1]);
            src[0][1] = box_center[0] + src_dir[0] + (shift[0] * scale_temp[0]);
            src[1][1] = box_center[1] + src_dir[1] + (shift[1] * scale_temp[1]);

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

            //Get the inverse matrix
            get_inverse(dst_padded, inv_dst);

            //Get the affine array
            matrix_mult(src, inv_dst, affine_matrix);

            //Get inverse of calculated affine matrix - needed for joint meta data updation
            invert_affine_tranform(affine_matrix, inverse_affine_matrix);

            // //Subtract the width and height of source image from the translation parameters
            // std::cout << std::endl<<"Affine matrix:" << std::endl
            //           << affine_matrix[0] << " " << affine_matrix[1] << " " << affine_matrix[2] << std::endl
            //           << affine_matrix[3] << " " << affine_matrix[4] << " " << affine_matrix[5] << std::endl<<std::endl;

            affine_matrix[2] = affine_matrix[2] + ((input_img_width / 2) * affine_matrix[0] + (input_img_height / 2) * affine_matrix[1] - input_img_width / 2);
            affine_matrix[5] = affine_matrix[5] + ((input_img_height / 2) * affine_matrix[4] + (input_img_width / 2) * affine_matrix[3] - input_img_height / 2);

            //Copy these values to local bb_centers,bb_scales
            _meta_data_info->get_joints_data_batch().center_batch[i] = box_center;
            _meta_data_info->get_joints_data_batch().scale_batch[i] = box_scale;
            _meta_data_info->get_joints_data_batch().rotation_batch[i] = rotate_deg;

            //Update keypoint meta data
            Joints joints;
            for (unsigned int keypoint_index = 0; keypoint_index < NUMBER_OF_JOINTS; keypoint_index++)
            {
                Joint joint;
                JointVisibility joint_visibility;
                float temp_x, temp_y;
                joint = _meta_data_info->get_joints_data_batch().joints_batch[i][keypoint_index];
                joint_visibility = _meta_data_info->get_joints_data_batch().joints_visibility_batch[i][keypoint_index];

                //Matrix multiplication of Affine matrix with keypoint values
                temp_x = (inverse_affine_matrix[0] * joint[0]) + (inverse_affine_matrix[1] * joint[1]) + (inverse_affine_matrix[2] * joint_visibility[0]);
                temp_y = (inverse_affine_matrix[3] * joint[0]) + (inverse_affine_matrix[4] * joint[1]) + (inverse_affine_matrix[5] * joint_visibility[0]);
                joint[0] = temp_x;
                joint[1] = temp_y;
                joints.push_back(joint);
            }
            _meta_data_info->get_joints_data_batch().joints_batch[i] = joints;
            joints.clear();
        }
    } else {
        for (uint i = 0; i < _batch_size; i++ )
        {
            _affine[i*6 + 0] = _x0.renew();
            _affine[i*6 + 1] = _y0.renew();
            _affine[i*6 + 2] = _x1.renew();
            _affine[i*6 + 3] = _y1.renew();
            _affine[i*6 + 4] = _o0.renew();
            _affine[i*6 + 5] = _o1.renew();
        }
    }
    vx_status affine_status;
    affine_status = vxCopyArrayRange((vx_array)_affine_array, 0, _batch_size * 6, sizeof(vx_float32), _affine.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST); //vxAddArrayItems(_width_array,_batch_size, _width, sizeof(vx_uint32));
    if(affine_status != 0)
        THROW(" vxCopyArrayRange failed in the WarpAffine(vxExtrppNode_WarpAffinePD) node: "+ TOSTR(affine_status))
}

void WarpAffineNode::init(float x0, float x1, float y0, float y1, float o0, float o1,int interpolation_type) {
    _x0.set_param(x0);
    _x1.set_param(x1);
    _y0.set_param(y0);
    _y1.set_param(y1);
    _o0.set_param(o0);
    _o1.set_param(o1);
    _interpolation_type=interpolation_type;
    _layout = (int)_inputs[0]->info().layout();
    _roi_type = (int)_inputs[0]->info().roi_type();
}

// void WarpAffineNode::init(FloatParam* x0, FloatParam* x1, FloatParam* y0, FloatParam* y1, FloatParam* o0, FloatParam* o1,int interpolation_type) {
//     _x0.set_param(core(x0));
//     _x1.set_param(core(x1));
//     _y0.set_param(core(y0));
//     _y1.set_param(core(y1));
//     _o0.set_param(core(o0));
//     _o1.set_param(core(o1));
//     _interpolation_type=interpolation_type;
//     _layout = (int)_inputs[0]->info().layout();
//     _roi_type = (int)_inputs[0]->info().roi_type();
// }

void WarpAffineNode::init(bool is_train, float rotate_probability, float half_body_probability, float scale_factor, float rotation_factor, FloatParam *x0, FloatParam *x1, FloatParam *y0, FloatParam *y1, FloatParam *o0, FloatParam *o1, int interpolation_type)
{
    //Paramters used in affine calculation
    _rotate_probability = rotate_probability;
    _half_body_probability = half_body_probability;
    _is_train = is_train;
    _scale_factor = scale_factor;
    _rotation_factor = rotation_factor;
    _x0.set_param(core(x0));
    _x1.set_param(core(x1));
    _y0.set_param(core(y0));
    _y1.set_param(core(y1));
    _o0.set_param(core(o0));
    _o1.set_param(core(o1));
    _interpolation_type=interpolation_type;
    _layout = (int)_inputs[0]->info().layout();
    _roi_type = (int)_inputs[0]->info().roi_type();
}


void WarpAffineNode::update_node(MetaDataBatch* meta_data) {
    _meta_data_info = meta_data;
    update_affine_array();
}

void WarpAffineNode::half_body_transform(int i, Center &box_center, Scale &box_scale, float aspect_ratio)
{
    Joint joint;
    Joints upper_joints, lower_joints, selected_joints;
    JointsVisibility joints_visibility = _meta_data_info->get_joints_data_batch().joints_visibility_batch[i];
    std::vector<float> vis_joints(NUMBER_OF_JOINTS);
    std::vector<int> UPPER_BODY_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> LOWER_BODY_IDS = {11, 12, 13, 14, 15, 16};
    uint num_vis_joints = 0;

    //Get the number of visible joints in given annotation
    for (uint kp_idx = 0; kp_idx < NUMBER_OF_JOINTS; kp_idx++)
    {
        vis_joints[num_vis_joints] = kp_idx;
        num_vis_joints = num_vis_joints + joints_visibility[kp_idx][0];
    }

    //Resize to number of visible joints
    vis_joints.resize(num_vis_joints);

    //Return if no of visible joints is <= no of halfbody joints
    if (num_vis_joints <= NUMBER_OF_JOINTS_HALFBODY)
    {
        return;
    }

    // std::cout<<"Proceeding for HalfBody Augmentation , Number of visible joints: "<<num_vis_joints<<std::endl;
    //Seperate the keypoints into upper body and lower body
    for (uint kp_idx = 0; kp_idx < num_vis_joints; kp_idx++)
    {
        joint = _meta_data_info->get_joints_data_batch().joints_batch[i][vis_joints[kp_idx]];
        if (vis_joints[kp_idx] <= 10)
        {
            upper_joints.push_back(joint);
        }
        else
        {
            lower_joints.push_back(joint);
        }
    }

    //Any of lower body/upper body joints should have minimum 3 joints
    if (lower_joints.size() < 2 && upper_joints.size() < 2)
    {
        return;
    }

    //select any of upper body and lower body joints
    selected_joints = upper_joints;
    if (upper_joints.size() > 2 &&  _random_value.renew() < 0.5)
    {
        selected_joints = upper_joints;
    }
    else if (lower_joints.size() > 2)
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

    //Clear the keypoint values
    upper_joints.clear();
    lower_joints.clear();
    selected_joints.clear();
}