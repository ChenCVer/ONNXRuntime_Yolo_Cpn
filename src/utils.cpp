#include "utils.h"

void custom_resize(Mat src, Mat &pad_img, map<string, float> &parameter_dict,
                   int res_h, int res_w, bool lb)
{

    Mat dst;
    float scale;
    int top;
    int bottom;
    int left;
    int right;
    int new_h;
    int new_w;
    int pad_h;
    int pad_w;
    int pad[4];
    int h = src.rows;
    int w = src.cols;

    if (h == res_h && w == res_w)
    {
        scale = 1.0;
        for (int i = 0; i < 4; i++)
        {
            pad[i] = 0;
        }
    }
    else
    {
        if (w / res_w >= h / res_h)
        {
            scale = 1.0 * res_w / w;
        }
        else
        {
            scale = 1.0 * res_h / h;
        }
        new_h = int(h * scale);
        new_w = int(w * scale);
        if (new_w == res_w && new_h == res_h)
        {
            for (int i = 0; i < 4; i++)
            {
                pad[i] = 0;
            }
        }
        else
        {
            if (lb)
            {
                pad_w = res_w - new_w;
                pad_h = res_h - new_h;
                pad[0] = 0;
                pad[1] = int(pad_h + 0.5);
                pad[2] = 0;
                pad[3] = int(pad_w + 0.5);
            }
            else
            {
                pad_w = (res_w - new_w) * 1.0 / 2;
                pad_h = (res_h - new_h) * 1.0 / 2;
                pad[0] = int(pad_h);
                pad[1] = int(pad_h + 0.5);
                pad[2] = int(pad_w);
                pad[3] = int(pad_w + 0.5);
            }
        }
    }

    parameter_dict["scale"] = scale;
    parameter_dict["pad_t"] = pad[0];
    parameter_dict["pad_b"] = pad[1];
    parameter_dict["pad_l"] = pad[2];
    parameter_dict["pad_r"] = pad[3];

    if (parameter_dict["scale"] != 1.0)
    {
        new_w = int(w * parameter_dict["scale"]);
        new_h = int(h * parameter_dict["scale"]);
        resize(src, dst, Size(new_w, new_h), 0.0, 0.0, cv::INTER_LINEAR);
    }

    top = int(parameter_dict["pad_t"]);
    bottom = int(parameter_dict["pad_b"]);
    left = int(parameter_dict["pad_l"]);
    right = int(parameter_dict["pad_r"]);
    if(top < 0) top = 0;
    if(bottom < 0) bottom = 0;
    if(left < 0) left = 0;
    if(right < 0) right = 0;
    copyMakeBorder(dst, pad_img, top, bottom, left, right, cv::BORDER_CONSTANT, 0);
}

void convert_img_with_origshape(map<string, float> &parameter_dict,
                                float array[][4], int numbers)
{
    int top = parameter_dict["pad_t"];
    int left = parameter_dict["pad_l"];
    float scale = parameter_dict["scale"];
    for (int i = 0; i < numbers; i++){
        for (int j = 0; j < 4; j++){
            if (j % 2 == 0){
                array[i][j] = array[i][j] - float(left);
            }
            else{
                array[i][j] = array[i][j] - float(top);
            }
            array[i][j] /= scale;
        }
    }
}

vector<Point2f> sortPoints(vector<Point2f> XYcenter)
{
    map<int, float> temp_map;
    vector<pair<int, float>> vec_xy;
    vector<cv::Point2f> sortedPoints;
    for (int i = 0; i < XYcenter.size();++i)
    {
        float xyLenght = sqrt(XYcenter[i].x*XYcenter[i].x + XYcenter[i].y*XYcenter[i].y);
        temp_map.insert(map<int,float>::value_type(i, xyLenght));
        vec_xy.push_back(make_pair(i, xyLenght));
    }

    std::sort(vec_xy.begin(), vec_xy.end(),[](const pair<int, float> &x, const pair<int, float> &y)-> int
    {
        return x.second < y.second;
    });

    for (vector<pair<int, float>>::iterator iter = vec_xy.begin(); iter != vec_xy.end(); iter++)
    {
        //cout << iter->first << ":" << iter->second << '\n';
        for (int j = 0; j < XYcenter.size(); ++j)
            if (iter->first == j)
                sortedPoints.push_back( XYcenter[j]);
    }
    return sortedPoints;
}
