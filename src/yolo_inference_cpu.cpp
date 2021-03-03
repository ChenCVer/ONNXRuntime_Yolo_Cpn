
#include "yolo_inference_cpu.h"
#include "utils.h"

YoloInference::YoloInference()
{
    // build InferenceSession
    // initialize  enviroment...one enviroment per process
    this->env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
    // initialize session options if needed
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    // Sets graph optimization level
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    this->session = new Ort::Session(*(this->env), "../weights/yolo.onnx", session_options);
}

YoloInference::YoloInference(int input_h,
                             int input_w,
                             int input_c,
                             const char *yolo_model_path)
{
    this->input_h = input_h;
    this->input_w = input_w;
    this->input_c = input_c;
    // build InferenceSession
    // initialize  enviroment...one enviroment per process
    this->env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
    // initialize session options if needed
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    // Sets graph optimization level
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    this->session = new Ort::Session(*(this->env), yolo_model_path, session_options);
}

void YoloInference::print_infos()
{
    // get number of model input nodes
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = this->session->GetInputCount();
    std::vector<const char *> input_node_names = {"input"};
    std::vector<const char *> output_node_names = {"boxes"};
    std::vector<int64_t> input_node_dims;
    printf("Number of inputs = %zu\n", num_input_nodes);
    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++)
    {
        // print input node names
        char *input_name = this->session->GetInputName(i, allocator);
        printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // print input node types
        Ort::TypeInfo type_info = this->session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        // print output node types
        Ort::TypeInfo out_info = this->session->GetOutputTypeInfo(i);
        auto out_tensor_info = out_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        // print input shapes/dims
        input_node_dims = tensor_info.GetShape();
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        for (int j = 0; j < input_node_dims.size(); j++)
            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
    }
}

vector<vector<float>> YoloInference::inference(const char *data_path)
{
    // pre-process input data
    Mat src = imread(data_path);
    Mat dst(this->input_h, this->input_w, CV_8UC3);

    map<string, float> parameter_dict;
    custom_resize(src, dst, parameter_dict, this->input_h, this->input_w, 1);
    int input_size = this->input_h * this->input_w * this->input_c;
    vector<float> input_tensor_values(input_size);

    int i = 0;
    for (int row = 0; row < this->input_h; row++)
    {
        uchar *uc_pixel = dst.data + row * dst.step;
        for (int col = 0; col < this->input_w; col++)
        {
            input_tensor_values[i + 0 * this->input_h * this->input_w] = (float)uc_pixel[2] / 255.0;
            input_tensor_values[i + 1 * this->input_h * this->input_w] = (float)uc_pixel[1] / 255.0;
            input_tensor_values[i + 2 * this->input_h * this->input_w] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }
    // create input tensor object from data values
    std::vector<const char *> input_node_names = {"input"};
    std::vector<const char *> output_node_names = {"boxes"};
    std::vector<int64_t> input_node_dims;
    size_t num_input_nodes = this->session->GetInputCount();
    assert(num_input_nodes == 1);
    Ort::TypeInfo type_info = this->session->GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    input_node_dims = tensor_info.GetShape();
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(),
                                                              input_size, input_node_dims.data(), 4);
    assert(input_tensor.IsTensor());

    // inference
    double timeStart = (double)getTickCount();

    auto output_tensors = this->session->Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                                             &input_tensor, 1, output_node_names.data(), 1);

    double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
    cout << "detect running time: " << nTime << " sec\n"
         << endl;
    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

    // post-process
    // 6. parser results
    // get output shape and dimensions infos
    Ort::TensorTypeAndShapeInfo output_infos = output_tensors[0].GetTensorTypeAndShapeInfo();
    size_t output_length;
    output_length = output_infos.GetElementCount();
    size_t output_dimensions;
    output_dimensions = output_infos.GetDimensionsCount();

    // Get pointer to output tensor float values
    float *results = output_tensors[0].GetTensorMutableData<float>();
    // Get numbers of predict bboxes.
    const size_t num_bboxes = output_length / 5;
    float bboxes[num_bboxes][4];
    float probs[num_bboxes];
    for (int i = 0; i < num_bboxes; i++){
        for (int j = 0; j < 5; j++){
            if (j == 4){
                probs[i] = results[i * 5 + j];
            }
            else{
                bboxes[i][j] = results[i * 5 + j];
            }
        }
    }

    // 7. convert bbox shape to orig shape
    convert_img_with_origshape(parameter_dict, bboxes, num_bboxes);
    vector<vector<float> > vec_bboxes;
    for(int i=0; i<num_bboxes; i++){
        vector<float> bbox;
        for(int j=0; j<4; j++){
            bbox.push_back(bboxes[i][j]);
        }
        vec_bboxes.push_back(bbox);
    }

    // debug
//    for (int i = 0; i < num_bboxes; i++)
//    {
//        rectangle(src, Point(int(bboxes[i][0]), int(bboxes[i][1])),
//                  Point(int(bboxes[i][2]), int(bboxes[i][3])),
//                  Scalar(0, 0, 255), 5);
//    }
//     namedWindow("det_result", cv::WINDOW_NORMAL);
//     imshow("det_result", src);
//     waitKey(0);
    // imwrite("../final.jpg", src);

    return vec_bboxes;
}