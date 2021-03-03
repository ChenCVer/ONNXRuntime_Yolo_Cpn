
#ifndef YOLO_INFERENCE_CPU_H
#define YOLO_INFERENCE_CPU_H

#include <assert.h>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;



class YoloInference
{

public:
    YoloInference();

    YoloInference(int input_h,
                  int input_w,
                  int input_c,
                  const char *yolo_model_path);

    ~YoloInference()
    {
        cout << "~YoloInference()" << endl;
        delete env;
        delete session;
    }

    // print infos
    void print_infos();

    // inference
    vector<vector<float> > inference(const char *data_path);

private:
    Ort::Env *env = nullptr;
    Ort::Session *session = nullptr;

public:
    int input_h = 448;
    int input_w = 448;
    int input_c = 3;

};

#endif