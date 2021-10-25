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

#include "coco_meta_data_reader.h"
#include <iostream>
#include <utility>
#include <algorithm>
#include <fstream>
#include "lookahead_parser.h"

using namespace std;

void COCOMetaDataReader::init(const MetaDataConfig &cfg)
{
    _path = cfg.path();
    _output = new BoundingBoxBatch();
    _keypoint = cfg.keypoint();
    _out_img_width = cfg.out_img_height();
    _out_img_height = cfg.out_img_height();
}

bool COCOMetaDataReader::exists(const std::string &image_name)
{
    return _map_content.find(image_name) != _map_content.end();
}

void COCOMetaDataReader::lookup(const std::vector<std::string> &image_names)
{
    if (image_names.empty())
    {
        WRN("No image names passed")
        return;
    }
    if (image_names.size() != (unsigned)_output->size())
        _output->resize(image_names.size());

    JointsDataBatch joints_data_batch;
    
    for (unsigned i = 0; i < image_names.size(); i++)
    {
        auto image_name = image_names[i];
        auto it = _map_content.find(image_name);
        if (_map_content.end() == it)
            THROW("ERROR: Given name not present in the map" + image_name)
        _output->get_bb_cords_batch()[i] = it->second->get_bb_cords();
        _output->get_bb_labels_batch()[i] = it->second->get_bb_labels();
        _output->get_img_sizes_batch()[i] = it->second->get_img_sizes();
        joints_data_batch.image_id_batch.push_back(it->second->get_joints_data().image_id);
        joints_data_batch.annotation_id_batch.push_back(it->second->get_joints_data().annotation_id);
        joints_data_batch.image_path_batch.push_back(it->second->get_joints_data().image_path);
        joints_data_batch.center_batch.push_back(it->second->get_joints_data().center);
        joints_data_batch.scale_batch.push_back(it->second->get_joints_data().scale);
        joints_data_batch.joints_batch.push_back(it->second->get_joints_data().joints);
        joints_data_batch.joints_visibility_batch.push_back(it->second->get_joints_data().joints_visibility);
        joints_data_batch.score_batch.push_back(it->second->get_joints_data().score);
        joints_data_batch.rotation_batch.push_back(it->second->get_joints_data().rotation);
    }
    _output->get_joints_data_batch() = joints_data_batch;
}

void COCOMetaDataReader::add(std::string image_name, BoundingBoxCords bb_coords, BoundingBoxLabels bb_labels, ImgSizes image_size)
{
    if (exists(image_name))
    {
        auto it = _map_content.find(image_name);
        it->second->get_bb_cords().push_back(bb_coords[0]);
        it->second->get_bb_labels().push_back(bb_labels[0]);
        return;
    }
    pMetaDataBox info = std::make_shared<BoundingBox>(bb_coords, bb_labels, image_size);
    _map_content.insert(pair<std::string, std::shared_ptr<BoundingBox>>(image_name, info));
}

void COCOMetaDataReader::add(std::string image_name, ImgSizes image_size, JointsData joints_data)
{
    if (exists(image_name))
    {
        // auto it = _map_content.find(image_name);
        return;
    }
    pMetaDataBox info = std::make_shared<BoundingBox>(image_size, joints_data);
    _map_content.insert(pair<std::string, std::shared_ptr<BoundingBox>>(image_name, info));
}


void COCOMetaDataReader::print_map_contents()
{
    BoundingBoxCords bb_coords;
    BoundingBoxLabels bb_labels;
    ImgSizes img_sizes;
    JointsData joints_data;

    std::cout<<"Printing Map contents"<<std::endl;
    std::cout << "\nBBox Annotations List: \n";
    for (auto &elem : _map_content)
    {
        std::cout << "\nName :\t " << elem.first;
        bb_coords = elem.second->get_bb_cords();
        bb_labels = elem.second->get_bb_labels();
        img_sizes = elem.second->get_img_sizes();
        joints_data = elem.second->get_joints_data();
        std::cout << "<wxh, num of bboxes>: " << img_sizes[0].w << " X " << img_sizes[0].h << " , " << bb_coords.size() << std::endl;

        for (unsigned int i = 0; i < bb_coords.size(); i++)
        {
            std::cout << " l : " << bb_coords[i].l << " t: :" << bb_coords[i].t << " r : " << bb_coords[i].r << " b: :" << bb_coords[i].b << "Label Id : " << bb_labels[i] << std::endl;
        }


        for (unsigned int i = 0; i < (joints_data.center.size()/2) ; i++)
        {
            std::cout << " center (x,y) : " << joints_data.center[0] << " " << joints_data.center[1] << std::endl;
            std::cout << " scale (w,h) : " << joints_data.scale[0] << " " << joints_data.scale[1] << std::endl;
        }

        for (unsigned int i = 0; i < NUMBER_OF_KEYPOINTS; i++)
        {
            std::cout << " x : " << joints_data.joints[i][0]<< " , y : " << joints_data.joints[i][1] << " , v : " << joints_data.joints_visibility[i][0] << std::endl;
        }
    }
}

void COCOMetaDataReader::read_all(const std::string &path)
{
    _coco_metadata_read_time.start(); // Debug timing
    std::string annotations_file = path;
    std::ifstream f(annotations_file);
    f.seekg(0, std::ios::end);
    size_t file_size = f.tellg();
    std::unique_ptr<char, std::function<void(char *)>> buff(
        new char[file_size + 1],
        [](char *data)
        { delete[] data; });
    f.seekg(0, std::ios::beg);
    buff.get()[file_size] = '\0';
    f.read(buff.get(), file_size);
    f.close();

    LookaheadParser parser(buff.get());

    BoundingBoxCords bb_coords;
    BoundingBoxLabels bb_labels;
    ImgSizes img_sizes;
    JointsData joints_data;

    BoundingBoxCord box;
    std::vector<float> box_center,box_scale;
    ImgSize img_size;
    float score = 1.0;
    float rotation = 0.0;

    // KeyPoints key_points(NUMBER_OF_KEYPOINTS);
    // KeyPointsVisibility key_points_visibility(NUMBER_OF_KEYPOINTS);

    

    RAPIDJSON_ASSERT(parser.PeekType() == kObjectType);
    parser.EnterObject();
    while (const char *key = parser.NextObjectKey())
    {
        if (0 == std::strcmp(key, "images"))
        {
            RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
            parser.EnterArray();
            while (parser.NextArrayValue())
            {
                string image_name;
                if (parser.PeekType() != kObjectType)
                {
                    continue;
                }
                parser.EnterObject();
                while (const char *internal_key = parser.NextObjectKey())
                {
                    if (0 == std::strcmp(internal_key, "width"))
                    {
                        img_size.w = parser.GetInt();
                    }
                    else if (0 == std::strcmp(internal_key, "height"))
                    {
                        img_size.h = parser.GetInt();
                    }
                    else if (0 == std::strcmp(internal_key, "file_name"))
                    {
                        image_name = parser.GetString();
                    }
                    else
                    {
                        parser.SkipValue();
                    }
                }
                img_sizes.push_back(img_size);
                _map_img_sizes.insert(pair<std::string, std::vector<ImgSize>>(image_name, img_sizes));
                img_sizes.clear();
            }
        }
        else if (0 == std::strcmp(key, "categories"))
        {
            RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
            parser.EnterArray();

            int id = 1, continuous_idx = 1;

            while (parser.NextArrayValue())
            {
                if (parser.PeekType() != kObjectType)
                {
                    continue;
                }
                parser.EnterObject();
                while (const char *internal_key = parser.NextObjectKey())
                {
                    if (0 == std::strcmp(internal_key, "id"))
                    {
                        id = parser.GetInt();
                    }
                    else
                    {
                        parser.SkipValue();
                    }
                }
                _label_info.insert(std::make_pair(id, continuous_idx));
                continuous_idx++;
            }
        }
        else if (0 == std::strcmp(key, "annotations"))
        {
            RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
            parser.EnterArray();
            while (parser.NextArrayValue())
            {
                int id = 1, label = 0, ann_id = 0;
                std::array<float, 4> bbox = {};
                std::array<float, NUMBER_OF_KEYPOINTS * 3> keypoint{}; 
                if (parser.PeekType() != kObjectType)
                {
                    continue;
                }
                parser.EnterObject();
                while (const char *internal_key = parser.NextObjectKey())
                {
                    if (0 == std::strcmp(internal_key, "image_id"))
                    {
                        id = parser.GetInt();
                    }
                    else if (0 == std::strcmp(internal_key, "category_id"))
                    {
                        label = parser.GetInt();
                    }
                    else if (0 == std::strcmp(internal_key, "id"))
                    {   
                        ann_id = parser.GetInt();
                    }
                    else if (0 == std::strcmp(internal_key, "bbox"))
                    {
                        RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
                        parser.EnterArray();

                        if (_keypoint)
                        {
                            box_center.push_back(parser.NextArrayValue() * parser.GetDouble());
                            box_center.push_back(parser.NextArrayValue() * parser.GetDouble());
                            box_scale.push_back(parser.NextArrayValue() * parser.GetDouble());
                            box_scale.push_back(parser.NextArrayValue() * parser.GetDouble());

                            // box_center[0] +=  (0.5 * box_scale[0]);
                            // box_center[1] +=  (0.5 * box_scale[1]);
                            
                            // box_scale[1] = (box_scale[0] > aspect_ratio * box_scale[1]) ? ((box_scale[1] = box_scale[0] * 1.0 / aspect_ratio) / PIXEL_STD) : box_scale[1] / PIXEL_STD; 
                            // box_scale[0] =  (box_scale[1] > aspect_ratio * box_scale[0]) ? ((box_scale[0] = aspect_ratio * box_scale[1]) /  PIXEL_STD) : box_scale[0] / PIXEL_STD;

                            // if (box_center[0] != -1) 
                            // {
                            //     box_scale[0] = SCALE_CONSTANT_CS * box_scale[0];
                            //     box_scale[1] = SCALE_CONSTANT_CS * box_scale[1];
                            // }

                            //Move to next section
                            parser.NextArrayValue();
                        }
                        else
                        {
                            int i = 0;
                            while (parser.NextArrayValue())
                            {
                                bbox[i] = parser.GetDouble();
                                ++i;
                            }
                        }
                    }
                    else if (0 == std::strcmp(internal_key, "keypoints")) 
                    {
                        RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
                        parser.EnterArray();
                        int i = 0;
                        while (parser.NextArrayValue())
                        {
                            keypoint[i] = parser.GetDouble();
                            ++i;
                        }
                    }
                    else
                    {
                        parser.SkipValue();
                    }
                }
                char buffer[13];
                sprintf(buffer, "%012d", id);
                string str(buffer);
                std::string file_name = str + ".jpg";
                auto it = _map_img_sizes.find(file_name);
                ImgSizes image_size = it->second; //Normalizing the co-ordinates & convert to "ltrb" format

                
                if (!_keypoint) // format conversion not required for key point processing 
                {
                    box.l = bbox[0] / image_size[0].w;
                    box.t = bbox[1] / image_size[0].h;
                    box.r = (bbox[0] + bbox[2]) / image_size[0].w;
                    box.b = (bbox[1] + bbox[3]) / image_size[0].h;

                    bb_coords.push_back(box);
                    bb_labels.push_back(label);
                    add(file_name, bb_coords, bb_labels, image_size);
                    bb_coords.clear();
                    bb_labels.clear();
                }
                //Store the keypoint values in Joints, Joints Visibility
                else
                {
                    //Preprocess bbox
                    float x1, y1, x2, y2;
                    float aspect_ratio = (288 * 1.0 / _out_img_height);
                    x1 = (box_center[0] > 0)? box_center[0] : 0;
                    y1 = (box_center[1] > 0)? box_center[1] : 0;
                    float proc_w =  ((box_scale[0] - 1) > 0)? (box_scale[0] - 1) : 0;
                    float proc_h = ((box_scale[1] - 1) > 0)? (box_scale[1] - 1) : 0;

                    x2 = ((img_size.w - 1) < (x1 + proc_w)) ? (img_size.w - 1) : (x1 + proc_w);
                    y2 = ((img_size.h - 1) < (y1 + proc_h)) ? (img_size.h - 1) : (y1 + proc_h);

                    //check area
                    if (x2 >= x1 && y2 >= y1)
                    {
                        box_center = {x1, y1};
                        box_scale = {x2-x1, y2-y1};
                    }
                   
                    //xywh2cs
                    box_center[0] +=  (0.5 * box_scale[0]);
                    box_center[1] +=  (0.5 * box_scale[1]);
                    
                    if (box_scale[0] > aspect_ratio * box_scale[1])
                    {
                        box_scale[1] = box_scale[0] * 1.0 / aspect_ratio;
                    }
                    else
                    {
                        box_scale[0] = box_scale[1] * aspect_ratio;
                    }

                    box_scale[0] = box_scale[0] / PIXEL_STD;
                    box_scale[1] = box_scale[1] / PIXEL_STD;

                    if (box_center[0] != -1)
                    {
                        box_scale[0] = SCALE_CONSTANT_CS * box_scale[0];
                        box_scale[1] = SCALE_CONSTANT_CS * box_scale[1];
                    }

                    std::vector<std::vector<float>> key_points(NUMBER_OF_KEYPOINTS),key_points_visibility(NUMBER_OF_KEYPOINTS);
                    unsigned int j = 0; 
                    for (unsigned int i = 0; i < NUMBER_OF_KEYPOINTS; i++)
                    {
                        key_points[i].push_back(keypoint[j]);
                        key_points[i].push_back(keypoint[j+1]);
                        key_points_visibility[i].push_back(!(!keypoint[j + 2]));
                        key_points_visibility[i].push_back(!(!keypoint[j + 2]));
                        j = j + 3;
                    }

                    joints_data.annotation_id = ann_id;
                    joints_data.image_id = id;
                    joints_data.image_path = file_name;
                    joints_data.center = box_center;
                    joints_data.scale = box_scale;
                    joints_data.joints = key_points;
                    joints_data.joints_visibility = key_points_visibility;
                    joints_data.score = score;
                    joints_data.rotation = rotation;

                    add(file_name, image_size, joints_data);
                    joints_data = {};
                    box_center.clear();
                    box_scale.clear();
                    key_points.clear();
                    key_points_visibility.clear();
                }
            }
        }
        else
        {
            parser.SkipValue();
        }
    }
    for (auto &elem : _map_content)
    {
        bb_coords = elem.second->get_bb_cords();
        bb_labels = elem.second->get_bb_labels();
        BoundingBoxLabels continuous_label_id;
        for (unsigned int i = 0; i < bb_coords.size(); i++)
        {
            auto _it_label = _label_info.find(bb_labels[i]);
            int cnt_idx = _it_label->second;
            continuous_label_id.push_back(cnt_idx);
        }
        elem.second->set_bb_labels(continuous_label_id);
    }
    _coco_metadata_read_time.end(); // Debug timing
    //std::cout<<"Printing map contents:"<<std::endl;
    //print_map_contents();
    std::cout << "coco read time in sec: " << _coco_metadata_read_time.get_timing() / 1000 << std::endl;
}

void COCOMetaDataReader::release(std::string image_name)
{
    if (!exists(image_name))
    {
        WRN("ERROR: Given name not present in the map" + image_name);
        return;
    }
    _map_content.erase(image_name);
}

void COCOMetaDataReader::release()
{
    _map_content.clear();
    _map_img_sizes.clear();
}

COCOMetaDataReader::COCOMetaDataReader() : _coco_metadata_read_time("coco meta read time", DBG_TIMING)
{
}
