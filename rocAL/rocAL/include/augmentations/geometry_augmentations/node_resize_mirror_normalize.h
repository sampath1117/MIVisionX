#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_crop_factory.h"
#include "parameter_vx.h"
#include "rocal_api_types.h"
//final
class ResizeMirrorNormalizeNode : public Node
{
public:
    ResizeMirrorNormalizeNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs);
    ResizeMirrorNormalizeNode() = delete;
    void init(unsigned dest_width, unsigned dest_height, RocalResizeScalingMode scaling_mode,
              std::vector<unsigned> max_size, RocalResizeInterpolationType interpolation_type,std::vector<float>& mean,  std::vector<float>& std_dev, IntParam *mirror, int layout);
    vx_array return_mirror(){ return _mirror.default_array();  }

    // unsigned int get_dst_width() { return _outputs[0]->info().max_dims()[0]; }
    // unsigned int get_dst_height() { return _outputs[0]->info().max_dims()[1]; }
    vx_array get_src_roi() { return _src_tensor_roi; }
    // vx_array get_src_height() { return _src_roi_height; }
    vx_array get_dst_width() { return _dst_roi_width; }
    vx_array get_dst_height() { return _dst_roi_height; }
    // std::vector<RocalROI> get_src_roi() { return *_inputs[0]->info().get_roi(); }
    // std::vector<RocalROI> get_dst_roi() { return *_outputs[0]->info().get_roi(); }
    void adjust_out_roi_size();

protected:
    void create_node() override ;
    void update_node() override;
private:
    std::vector<vx_float32> _mean_vx, _std_dev_vx;
    vx_array  _mean_array, _std_dev_array,_mirror_array, _dst_roi_width , _dst_roi_height,_src_roi_width, _src_roi_height;
    unsigned _layout, _roi_type;
    std::vector<float> _mean;
    std::vector<float> _std_dev;
    int _interpolation_type;
    ParameterVX<int> _mirror;
    constexpr static int   MIRROR_RANGE [2] =  {0, 1};
    RocalResizeScalingMode _scaling_mode;
    unsigned _src_width, _src_height, _dst_width, _dst_height, _out_width, _out_height;
    unsigned _max_width = 0, _max_height = 0;
    std::vector<unsigned> _dst_roi_width_vec, _dst_roi_height_vec;
};
