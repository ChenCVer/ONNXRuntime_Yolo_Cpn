#ifndef CPN_INFERENCE_CPU_H
#define CPN_INFERENCE_CPU_H

#include <assert.h>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;



class CpnInference
{
public:
    CpnInference();

    CpnInference(int input_h,
                 int input_w,
                 int input_c,
                 int downsample,
                 int output_c,
                 const char *model_path);

    ~CpnInference()
    {
        delete env;
        delete session;
    }

    // print infos
    void print_infos();

    // inference
    void inference(vector<vector<float> > bboxes, const char *data_path);

private:
    Ort::Env *env = nullptr;
    Ort::Session *session = nullptr;

public:
    int input_h = 448;
    int input_w = 448;
    int input_c = 3;
    int downsample = 4;
    int output_c = 4;

};

#endif