#ifndef _UTILS_H
#define _UTILS_H

#include <assert.h>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;

void custom_resize(Mat src, Mat &pad_img,
                   map<string, float> &parameter_dict,
                   int res_h, int res_w, bool lb);

void convert_img_with_origshape(map<string, float> &parameter_dict,
                                float array[][4], int numbers);


vector<Point2f> sortPoints(vector<Point2f> XYcenter);

#endif