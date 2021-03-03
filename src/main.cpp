#include "yolo_inference_cpu.h"
#include "cpn_inference_cpu.h"
#include "yolo_cpn_inference.h"

vector<vector<float> > yolov3inference(const char* yolo_model_path, const char* data_path)
{
    YoloInference yolo_inference(448, 448, 3, yolo_model_path);
    vector<vector<float> > bboxes = yolo_inference.inference(data_path);
    return bboxes;
}

void cpninference(const char* model_path, const char* data_path)
{
    CpnInference cpn_inference(320, 320, 3, 4, 4, model_path);
    cpn_inference.print_infos();
    vector<vector<float> > bboxes;
    vector<float> bbox{2657.36, 2436.62, 3008.31, 2769.65};
    bboxes.push_back(bbox);
    cpn_inference.inference(bboxes, data_path);
}

void yolo_cpn_inference(int yolo_input_h,
                        int yolo_input_w,
                        int yolo_input_c,
                        float score_thr,
                        int cpn_input_h,
                        int cpn_input_w,
                        int cpn_input_c,
                        int downsample,
                        int cpn_output_c,
                        const char* yolo_model_path,
                        const char* cpn_model_path,
                        const char* data_path,
                        struct Result& result){

    YoloCpnInference yolocpn_inference(yolo_input_h,
                                       yolo_input_w,
                                       yolo_input_c,
                                       score_thr,
                                       cpn_input_h,
                                       cpn_input_w,
                                       cpn_input_c,
                                       downsample,
                                       cpn_output_c,
                                       yolo_model_path,
                                       cpn_model_path);

    yolocpn_inference.print_infos();
    yolocpn_inference.inference(data_path, result, 1);

}


int main(int argc, char** argv){
    // 这里的相对路径为什么要这样写, 是应为CLion生成的可执行文件在cmake-build-debug/src/下.
    // 的onnx_yolo_cpn, 下面的相对路径就是相对于可执行文件进行追溯的, 因此是: ../../
    // 如果你是用cmake .. + make的形式进行编译, 则可执行文件生成在build/src/下.
    const char* yolo_model_path = "../../weights/yolo.onnx";
    const char* cpn_model_path = "../../weights/cpn_new.onnx";
    const char* data_path = "../../data/test.jpg";
//    yolov3inference(yolo_model_path, data_path);
//    cpninference(cpn_model_path, data_path);
    struct Result result;
    yolo_cpn_inference(448,448,3,0.1,256,256,
                       3,4,4,yolo_model_path, cpn_model_path, data_path, result);
    return 0;
}