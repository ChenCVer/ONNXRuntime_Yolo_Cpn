
#ifndef YOLO_CPN_INFERENCE_H
#define YOLO_CPN_INFERENCE_H

#include <assert.h>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;

/*
 * state: 0 0 0 0 0 1 1 1
 * bboxes:[xywh xywh xywh xywh xywh xywh xywh xywh]
 * x: x1x2x3x4 x1x2x3x4 x1x2x3x4 x1x2x3x4 x1x2x3x4
 * y: y1y2y3y4 y1y2y3y4 y1y2y3y4 y1y2y3y4 y1y2y3y4
 */
struct Result{
    vector<int> state;
    vector<Rect2f> bboxes;
    vector<int> x;
    vector<int> y;
};


class YoloCpnInference
{

public:
    YoloCpnInference();

    YoloCpnInference(int yolo_input_h,
                     int yolo_input_w,
                     int yolo_input_c,
                     float score_thr,
                     int cpn_input_h,
                     int cpn_input_w,
                     int cpn_input_c,
                     int downsample,
                     int cpn_output_c,
                     const char* yolo_model_path,
                     const char* cpn_model_path);

    ~YoloCpnInference()
    {
        delete env;
        delete yolo_session;
        delete cpn_session;
    }

    // print infos
    void print_infos();

    // inference
    void inference(const char *data_path, struct Result& result, bool debug);

private:
    Ort::Env *env = nullptr;
    Ort::Session *yolo_session = nullptr;
    Ort::Session *cpn_session = nullptr;

public:
    // yolo
    int yolo_input_h = 448;
    int yolo_input_w = 448;
    int yolo_input_c = 3;
    float score_thr = 0.1;

    // cpn
    int cpn_input_h = 320;
    int cpn_input_w = 320;
    int cpn_input_c = 3;
    int downsample = 4;
    int cpn_output_c = 4;

};

#endif