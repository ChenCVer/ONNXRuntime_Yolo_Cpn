#include "cpn_inference_cpu.h"
#include "utils.h"


CpnInference::CpnInference()
{
    // build InferenceSession
    // initialize  enviroment...one enviroment per process
    this->env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
    // initialize session options if needed
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    // Sets graph optimization level
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    this->session = new Ort::Session(*(this->env), "../weights/cpn.onnx", session_options);

}

CpnInference::CpnInference(int input_h,
                           int input_w,
                           int input_c,
                           int downsample,
                           int output_c,
                           const char *model_path)
{
    this->input_h = input_h;
    this->input_w = input_w;
    this->input_c = input_c;
    this->downsample = downsample;
    this->output_c = output_c;
    // build InferenceSession
    // initialize  enviroment...one enviroment per process
    this->env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
    // initialize session options if needed
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    // Sets graph optimization level
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    this->session = new Ort::Session(*(this->env), model_path, session_options);

}

void CpnInference::print_infos()
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

void CpnInference::inference(vector<vector<float> > bboxes, const char *data_path){
    Mat src = imread(data_path);
    // 3. pre-process input data
    vector<Mat> img_list;
    int num_bboxes = bboxes.size();
    vector<map<string, float>> parameter_dict_list(num_bboxes);
    for(int i=0; i<num_bboxes; i++){
        Rect roi = Rect(int(bboxes[i][0]), int(bboxes[i][1]),
                        int(bboxes[i][2]) - int(bboxes[i][0]),
                        int(bboxes[i][3]) - int(bboxes[i][1]));
        Mat img_crop = src(roi);
        Mat cpn_dst(this->input_h, this->input_w, CV_8UC3);
        custom_resize(img_crop, cpn_dst, parameter_dict_list[i], this->input_h, this->input_w, 0);
        img_list.push_back(cpn_dst);
    }

    int input_size = this->input_h * this->input_w * this->input_c;
    vector<float> cpn_input_tensor_values(num_bboxes * input_size);
    for (int b = 0; b < num_bboxes; b++) {
        Mat pr_img = img_list[b];
        int i = 0;
        for (int row = 0; row < this->input_h; ++row) {
            uchar* uc_pixel = pr_img.data + row * pr_img.step;
            for (int col = 0; col < this->input_w; ++col) {
                cpn_input_tensor_values[b * input_size + i + 0 * input_h * input_w] = (float)uc_pixel[2] / 255.0;
                cpn_input_tensor_values[b * input_size + i + 1 * input_h * input_w] = (float)uc_pixel[1] / 255.0;
                cpn_input_tensor_values[b * input_size + i + 2 * input_h * input_w] = (float)uc_pixel[0] / 255.0;
                uc_pixel += 3;
                ++i;
            }
        }
    }

    // 4. create input tensor object from data values
    auto memory_info_2 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<const char *> input_node_names = {"input"};
    std::vector<const char *> output_node_names = {"output"};
    std::vector<int64_t> input_node_dims;
    size_t num_input_nodes = this->session->GetInputCount();
    assert(num_input_nodes == 1);
    Ort::TypeInfo type_info = this->session->GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    input_node_dims = tensor_info.GetShape();
    // modify the input node dims. because cpn_input_node_dims[0] is equal to -1.
    input_node_dims[0] = num_bboxes;
    Ort::Value input_tensor_2 = Ort::Value::CreateTensor<float>(memory_info_2,
                                                                cpn_input_tensor_values.data(),
                                                                cpn_input_tensor_values.size(),
                                                                input_node_dims.data(), 4);
    assert(input_tensor_2.IsTensor());

    // 5. inference
    double timeStart_2 = (double)getTickCount();
    auto cpn_output_tensors = this->session->Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                                                 &input_tensor_2, 1, output_node_names.data(), 1);
    double nTime_2 = ((double)getTickCount() - timeStart_2) / getTickFrequency();
    cout << "cpn running time: " << nTime_2 << " sec\n" << endl;

    // 6. parser results
    // get output shape and dimensions infos
    vector<int>   max_indexes(num_bboxes * this->output_c);
    vector<int> max_indexes_x(num_bboxes * this->output_c);
    vector<int> max_indexes_y(num_bboxes * this->output_c);
    int output_h = this->input_h / downsample;
    int output_w = this->input_w / downsample;
    int output_size = output_h * output_w * output_c;
    float* results_2 = cpn_output_tensors[0].GetTensorMutableData<float>();
    for(int b=0; b<num_bboxes; b++){
        for(int c=0; c<this->output_c; c++){
            float max_value = -100000.0;
            float output_value = 0.0;
            for(int h=0; h<(input_h/downsample); h++){
                for(int w=0; w<(input_w/downsample); w++){
                    output_value = results_2[b*output_size + c*output_h*output_w+h*output_w+w];
                    if(max_value < output_value){
                        max_value = output_value;
                        max_indexes[b*output_c+c] = h*output_w+w;
                    }
                }
            }
            max_indexes_x[b*output_c+c] = (max_indexes[b*output_c+c] % output_h) * downsample;
            max_indexes_y[b*output_c+c] = (max_indexes[b*output_c+c] / output_w) * downsample;
        }
    }

    // convert to orig shape
    for(int i=0; i<num_bboxes; i++){
        int top = parameter_dict_list[i]["pad_t"];
        int left = parameter_dict_list[i]["pad_l"];
        float scale = parameter_dict_list[i]["scale"];
        for(int j=0; j<output_c; j++){
            max_indexes_x[i * output_c + j] -= left;
            max_indexes_y[i * output_c + j] -= top;
            max_indexes_x[i * output_c + j]  = float(max_indexes_x[i * output_c + j]) / scale;
            max_indexes_y[i * output_c + j]  = float(max_indexes_y[i * output_c + j]) / scale;
            max_indexes_x[i * output_c + j] += bboxes[i][0];
            max_indexes_y[i * output_c + j] += bboxes[i][1];
        }
    }

    // debug
    for(int i=0; i<num_bboxes; i++){
        rectangle(src, Point(int(bboxes[i][0]), int(bboxes[i][1])),
                  Point(int(bboxes[i][2]), int(bboxes[i][3])),
                  Scalar(0, 0, 255), 5);
    }
    for(int i=0; i<max_indexes_x.size(); i++){
        circle(src, Point(int(max_indexes_x[i]), int(max_indexes_y[i])),
               30, Scalar(0,255,0),3);
    }

    namedWindow("cpn_result", WINDOW_NORMAL);
    imshow("cpn_result", src);
    waitKey(0);
}