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

    for (unsigned i = 0; i < image_names.size(); i++)
    {
        auto image_name = image_names[i];
        auto it = _map_content.find(image_name);
        if (_map_content.end() == it)
            THROW("ERROR: Given name not present in the map" + image_name)
        _output->get_bb_cords_batch()[i] = it->second->get_bb_cords();
        _output->get_bb_labels_batch()[i] = it->second->get_bb_labels();
        _output->get_img_sizes_batch()[i] = it->second->get_img_sizes();
        _output->get_img_key_points_batch()[i] = it->second->get_img_key_points();
    }
}

void COCOMetaDataReader::add(std::string image_name, BoundingBoxCords bb_coords, BoundingBoxLabels bb_labels, ImgSizes image_size,ImageKeyPoints img_key_points) //add change
{
    if (exists(image_name))
    {
        auto it = _map_content.find(image_name);
        it->second->get_bb_cords().push_back(bb_coords[0]);
        it->second->get_bb_labels().push_back(bb_labels[0]);
        it->second->get_img_key_points().push_back(img_key_points[0]);
        return;
    }
    pMetaDataBox info = std::make_shared<BoundingBox>(bb_coords, bb_labels, image_size,img_key_points);
    _map_content.insert(pair<std::string, std::shared_ptr<BoundingBox>>(image_name, info));
}

void COCOMetaDataReader::print_map_contents()
{
    BoundingBoxCords bb_coords;
    BoundingBoxLabels bb_labels;
    ImgSizes img_sizes;
    ImageKeyPoints img_key_points;
    size_t num_keypoints=17;

    std::cout << "\nBBox Annotations List: \n";
    for (auto &elem : _map_content)
    {
        std::cout << "\nName :\t " << elem.first;
        bb_coords = elem.second->get_bb_cords();
        bb_labels = elem.second->get_bb_labels();
        img_sizes = elem.second->get_img_sizes();
        img_key_points=elem.second->get_img_key_points();
        std::cout << "<wxh, num of bboxes>: " << img_sizes[0].w << " X " << img_sizes[0].h << " , " << bb_coords.size() << std::endl;
        for (unsigned int i = 0; i < bb_coords.size(); i++)
        {
            std::cout << " l : " << bb_coords[i].l << " t: :" << bb_coords[i].t << " r : " << bb_coords[i].r << " b: :" << bb_coords[i].b << "Label Id : " << bb_labels[i] << std::endl;
        }

       
        //std::cout<<"New Detection:"<<std::endl;
        for (unsigned int i = 0; i < img_key_points.size(); i++)
        {
            //std::cout<<"Size of key points index is:"<<img_key_points[i].size()<<std::endl;
            //std::cout<<"Entered keypoint printing loop:"<<std::endl;
            for (unsigned int j = 0; j < num_keypoints; j++)
            {
                std::cout << " x : " << img_key_points[i][j].x << " , y: " << img_key_points[i][j].y << " , v : " << img_key_points[i][j].v  << std::endl;
            }
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
    ImageKeyPoints img_key_points;

    BoundingBoxCord box;
    ImgSize img_size;
    size_t num_keypoints=17;
    KeyPoints key_points(num_keypoints);
    
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
                int id = 1, label = 0;
                std::array<float, 4> bbox;
                std::array<float,51> keypoint{};   //new change
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
                    else if (0 == std::strcmp(internal_key, "bbox"))
                    {
                        RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
                        parser.EnterArray();
                        int i = 0;
                        while (parser.NextArrayValue())
                        {
                            bbox[i] = parser.GetDouble();
                            ++i;
                        }
                    }
                    else if (0 == std::strcmp(internal_key, "keypoints")) //add change
                    {
                        RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
                        parser.EnterArray(); 
                        int i = 0;
                        while (parser.NextArrayValue())
                        {
                            keypoint[i]=parser.GetDouble();
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
                
                box.l = bbox[0] / image_size[0].w;
                box.t = bbox[1] / image_size[0].h;
                box.r = (bbox[0] + bbox[2]) / image_size[0].w;
                box.b = (bbox[1] + bbox[3]) / image_size[0].h;

                bb_coords.push_back(box);
                bb_labels.push_back(label);
                
               
                //Store the 1-D keypoint array into 17x3 Keypoint array
                unsigned int j=0;  //new change
                for(unsigned int i = 0; i < num_keypoints; i++)
                {
                    key_points[i].x = keypoint[j];
                    key_points[i].y = keypoint[j+1];
                    key_points[i].v = keypoint[j+2];
                    j=j+3;
                }
                //std::cout<<"Completed setting keypoint values"<<std::endl;
                img_key_points.push_back(key_points);
                //std::cout<<"Pushed keypoint values to the keypoint vector"<<std::endl;
                add(file_name, bb_coords, bb_labels, image_size,img_key_points);
                bb_coords.clear();
                bb_labels.clear();
                img_key_points.clear();
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
    print_map_contents();
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
