#include "yolo_cpn_inference.h"
#include "utils.h"

YoloCpnInference::YoloCpnInference()
{
    // build InferenceSession
    // initialize  enviroment...one enviroment per process
    this->env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
    // initialize session options if needed
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    // Sets graph optimization level
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    this->yolo_session = new Ort::Session(*(this->env), "../weights/yolo.onnx", session_options);
    this->cpn_session = new Ort::Session(*(this->env), "../weights/cpn.onnx", session_options);
}

YoloCpnInference::YoloCpnInference(int yolo_input_h,
                                   int yolo_input_w,
                                   int yolo_input_c,
                                   float score_thr,
                                   int cpn_input_h,
                                   int cpn_input_w,
                                   int cpn_input_c,
                                   int downsample,
                                   int cpn_output_c,
                                   const char* yolo_model_path,
                                   const char* cpn_model_path)
{
    this->yolo_input_h = yolo_input_h;
    this->yolo_input_w = yolo_input_w;
    this->yolo_input_c = yolo_input_c;
    this->score_thr = score_thr;
    this->cpn_input_h = cpn_input_h;
    this->cpn_input_w = cpn_input_w;
    this->cpn_input_c = cpn_input_c;
    this->downsample = downsample;
    this->cpn_output_c = cpn_output_c;
    // build InferenceSession
    // initialize  enviroment...one enviroment per process
    this->env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
    // initialize session options if needed
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    // Sets graph optimization level
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    this->yolo_session = new Ort::Session(*(this->env), yolo_model_path, session_options);
    this->cpn_session = new Ort::Session(*(this->env), cpn_model_path, session_options);
}

void YoloCpnInference::print_infos(){

    // get number of model input nodes
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;
    cout << "yolo input infos: " << endl;
    size_t num_input_nodes = this->yolo_session->GetInputCount();
    std::vector<const char *> input_node_names = {"input"};
    std::vector<const char *> output_node_names = {"boxes"};
    std::vector<int64_t> input_node_dims;
    printf("Number of inputs = %zu\n", num_input_nodes);
    // iterate over all input nodes
    for (int i = 0; i < num_input_nodes; i++)
    {
        // print input node names
        char *input_name = this->yolo_session->GetInputName(i, allocator);
        printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name;

        // print input node types
        Ort::TypeInfo type_info = this->yolo_session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        // print output node types
        Ort::TypeInfo out_info = this->yolo_session->GetOutputTypeInfo(i);
        auto out_tensor_info = out_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        // print input shapes/dims
        input_node_dims = tensor_info.GetShape();
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        for (int j = 0; j < input_node_dims.size(); j++)
            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
    }

    cout << "cpn input infos: " << endl;
    size_t cpn_num_input_nodes = this->cpn_session->GetInputCount();
    std::vector<const char *> cpn_input_node_names = {"input"};
    std::vector<const char *> cpn_output_node_names = {"output"};
    std::vector<int64_t> cpn_input_node_dims;
    printf("Number of inputs = %zu\n", cpn_num_input_nodes);
    // iterate over all input nodes
    for (int i = 0; i < cpn_num_input_nodes; i++)
    {
        // print input node names
        char *input_name = this->cpn_session->GetInputName(i, allocator);
        printf("Input %d : name=%s\n", i, input_name);
        cpn_input_node_names[i] = input_name;

        // print input node types
        Ort::TypeInfo type_info = this->cpn_session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        // print output node types
        Ort::TypeInfo out_info = this->cpn_session->GetOutputTypeInfo(i);
        auto out_tensor_info = out_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);

        // print input shapes/dims
        cpn_input_node_dims = tensor_info.GetShape();
        printf("Input %d : num_dims=%zu\n", i, cpn_input_node_dims.size());
        for (int j = 0; j < cpn_input_node_dims.size(); j++)
            printf("Input %d : dim %d=%jd\n", i, j, cpn_input_node_dims[j]);
    }
}

void YoloCpnInference::inference(const char *data_path, struct Result& result, bool debug){
    // yolo inference
    // pre-process input data
    Mat src = imread(data_path);
    Mat cpn_src = src.clone();
    Mat result_src = src.clone();
    Mat dst(this->yolo_input_h, this->yolo_input_w, CV_8UC3);
    map<string, float> parameter_dict;
    custom_resize(src, dst, parameter_dict, this->yolo_input_h, this->yolo_input_w, 1);
    int input_size = this->yolo_input_h * this->yolo_input_w * this->yolo_input_c;
    vector<float> input_tensor_values(input_size);

    int i = 0;
    for (int row = 0; row < this->yolo_input_h; row++)
    {
        uchar *uc_pixel = dst.data + row * dst.step;
        for (int col = 0; col < this->yolo_input_w; col++)
        {
            input_tensor_values[i + 0 * this->yolo_input_h * this->yolo_input_w] = (float)uc_pixel[2] / 255.0;
            input_tensor_values[i + 1 * this->yolo_input_h * this->yolo_input_w] = (float)uc_pixel[1] / 255.0;
            input_tensor_values[i + 2 * this->yolo_input_h * this->yolo_input_w] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }
    // create input tensor object from data values
    std::vector<const char *> input_node_names = {"input"};
    std::vector<const char *> output_node_names = {"boxes"};
    std::vector<int64_t> input_node_dims;
    size_t num_input_nodes = this->yolo_session->GetInputCount();
    assert(num_input_nodes == 1);
    Ort::TypeInfo type_info = this->yolo_session->GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    input_node_dims = tensor_info.GetShape();
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(),
                                                              input_size, input_node_dims.data(), 4);
    assert(input_tensor.IsTensor());

    // inference
    vector<Ort::Value> output_tensors;
    double timeStart = (double)getTickCount();
    try{
        output_tensors = this->yolo_session->Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                                                      &input_tensor, 1, output_node_names.data(), 1);
    }
    catch (...) {
        // 需要特殊处理一类情况, 由于转换onnx文件时, 必须要用有输出的才能正确转出onnx文件,
        // 在工程实践中, 则肯定会遇到没有任何输出的图片, 对于这类图片需要特殊处理.
        return;
    }

    double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
    cout << "detect running time: " << nTime << " sec\n" << endl;
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
    const size_t num_bboxes = output_length / 6;
    float bboxes[num_bboxes][4];
    float probs[num_bboxes];
    int labels[num_bboxes];
    for (int i = 0; i < num_bboxes; i++){
        for (int j = 0; j < 6; j++){
            if (j == 4){
                probs[i] = results[i * 6 + j];
            }
            else if( j==5){
                labels[i] = int(results[i * 6 + j]);
            }
            else{
                bboxes[i][j] = results[i * 6 + j];
            }
        }
    }

    // convert bbox shape to orig shape
    convert_img_with_origshape(parameter_dict, bboxes, num_bboxes);

    // 7. filter bboxes
    size_t filter_num_bboxes = 0;
    vector<int> filter_index;

    // 置信度过滤
    for(int i=0; i<num_bboxes; i++){
        if(probs[i] > this->score_thr){
            filter_num_bboxes ++;
            filter_index.push_back(i);
        }
    }

    float filter_bboxes[filter_num_bboxes][4];
    int filter_labels[filter_num_bboxes];
    for(int i=0; i<filter_num_bboxes; i++){
        filter_labels[i] = labels[filter_index[i]];
        for(int j=0; j<4; j++){
            filter_bboxes[i][j] = bboxes[filter_index[i]][j];
        }
    }

    // 过滤后如果没有ok样本和ng样本:
    if(filter_num_bboxes == 0)
        return;

    // 统计正负样本.
    int ng_bbox_nums=0;
    int ok_bbox_nums=0;
    vector<int> ng_bbox_indexes;
    vector<int> ok_bbox_indexes;
    for(int i=0; i<filter_num_bboxes; i++){
        // ok
        if(filter_labels[i] == 0){
            ok_bbox_nums++;
            ok_bbox_indexes.push_back(i);
        }
        // ng
        else{
            ng_bbox_nums++;
            ng_bbox_indexes.push_back(i);
        }
    }

    // 将ok和ng按照ok在前, ng在后的顺序放在all_bboxes中.
    float all_bboxes[ng_bbox_nums+ok_bbox_nums][4];
    float ok_bboxes[ok_bbox_nums][4];
    float ng_bboxes[ng_bbox_nums][4];
    // 将ok的bboxes装进all_bboxes中
    for(int i=0; i<ok_bbox_nums; i++){
        for(int j=0; j<4; j++){
            ok_bboxes[i][j] = filter_bboxes[ok_bbox_indexes[i]][j];
            all_bboxes[i][j] = filter_bboxes[ok_bbox_indexes[i]][j];
        }
    }

    // 对ok的bboxes进行排序: ok_bboxes中的坐标点从左到右, 从上到下:
    // todo: 还没做.
    vector<Point2f> ok_points;
    for(int i=0; i<ok_bbox_nums; i++){
        ok_points.push_back(Point2f(ok_bboxes[i][0], ok_bboxes[i][1]));
    }
    vector<Point2f> ok_sorted_points = sortPoints(ok_points);

    // debug
    if (debug) {
        if (ok_bbox_nums != 0) {
            for (int i = 0; i < ok_bbox_nums; i++) {

                putText(src, to_string(i),Point2d(int(ok_sorted_points[i].x), int(ok_sorted_points[i].y)),
                        1,5.0,Scalar(255, 255, 0), 5);
            }
        }
        namedWindow("det_result", cv::WINDOW_NORMAL);
        imshow("det_result", src);
        waitKey(0);
    }
    // 将ng的bboxes装进all_bboxes中
    for(int i=0; i<ng_bbox_nums; i++){
        for(int j=0; j<4; j++){
            ng_bboxes[i][j] = filter_bboxes[ng_bbox_indexes[i]][j];
            all_bboxes[i+ok_bbox_nums][j] = filter_bboxes[ng_bbox_indexes[i]][j];
        }
    }

    // 如果只有ng样本, 没有ok样本
    if(ok_bbox_nums == 0 && ng_bbox_nums != 0){
        // ng result
        for(int i=ok_bbox_nums; i<filter_num_bboxes; i++){

            // construct ok bbox.
            Rect2f rect;
            rect.x = all_bboxes[i][0];
            rect.y = all_bboxes[i][1];
            rect.width = all_bboxes[i][2] - all_bboxes[i][0];
            rect.height = all_bboxes[i][3] - all_bboxes[i][1];
            result.bboxes.push_back(rect);

            // construct ng(1) or ok(0).
            result.state.push_back(1);

            // ng样本不需要关键点检测.
        }
        return;
    }

    // debug
    if (debug) {
        if(ok_bbox_nums != 0){
            for(int i=0; i<ok_bbox_nums; i++){
                rectangle(src, Point(int(all_bboxes[i][0]), int(all_bboxes[i][1])),
                          Point(int(all_bboxes[i][2]), int(all_bboxes[i][3])),
                          Scalar(255, 0, 0), 5);
                putText(src, to_string(i),Point(int(all_bboxes[i][0]), int(all_bboxes[i][1])),
                        1,5.0,Scalar(255, 0, 0), 5);
                putText(src, to_string(0),Point(int(all_bboxes[i][2] - 30), int(all_bboxes[i][1])),
                        1,5.0,Scalar(255, 255, 0), 5);
            }
        }

        if(ng_bbox_nums != 0){
            for(int i=ok_bbox_nums; i<filter_num_bboxes; i++){
                rectangle(src, Point(int(all_bboxes[i][0]), int(all_bboxes[i][1])),
                          Point(int(all_bboxes[i][2]), int(all_bboxes[i][3])),
                          Scalar(0, 0, 255), 5);
                putText(src, to_string(i),Point(int(all_bboxes[i][0]), int(all_bboxes[i][1])),
                        1,5.0,Scalar(0, 0, 255), 5);
                putText(src, to_string(1),Point(int(all_bboxes[i][2] - 30), int(all_bboxes[i][1])),
                        1,5.0,Scalar(255, 0, 255), 5);
            }
        }

        namedWindow("det_result", cv::WINDOW_NORMAL);
        imshow("det_result", src);
        waitKey(0);
    }

    // cpn inference
    vector<Mat> img_list;
    vector<map<string, float>> parameter_dict_list(ok_bbox_nums);
    for(int i=0; i<ok_bbox_nums; i++){
        // Sometimes it’s very strange, the value(ok_bboxes[i][0] or ok_bboxes[i][1])
        // will appear to be negative.
        if(ok_bboxes[i][0] < 0) ok_bboxes[i][0] = 0;
        if(ok_bboxes[i][1] < 0) ok_bboxes[i][1] = 0;
        if(ok_bboxes[i][2] >cpn_src.cols) ok_bboxes[i][2] = cpn_src.cols;
        if(ok_bboxes[i][3] >cpn_src.rows) ok_bboxes[i][2] = cpn_src.rows;
        Rect roi = Rect(int(ok_bboxes[i][0]), int(ok_bboxes[i][1]),
                        int(ok_bboxes[i][2]) - int(ok_bboxes[i][0]),
                        int(ok_bboxes[i][3]) - int(ok_bboxes[i][1]));
        Mat img_crop = cpn_src(roi);
        Mat cpn_dst(this->cpn_input_h, this->cpn_input_w, CV_8UC3);
        custom_resize(img_crop, cpn_dst, parameter_dict_list[i], this->cpn_input_h, this->cpn_input_w, 0);
        img_list.push_back(cpn_dst);
    }

    int cpn_input_size = this->cpn_input_h * this->cpn_input_w * this->cpn_input_c;
    vector<float> cpn_input_tensor_values(ok_bbox_nums * cpn_input_size);
    for (int b = 0; b < ok_bbox_nums; b++) {
        Mat pr_img = img_list[b];
        int i = 0;
        for (int row = 0; row < this->cpn_input_h; ++row) {
            uchar* uc_pixel = pr_img.data + row * pr_img.step;
            for (int col = 0; col < this->cpn_input_w; ++col) {
                cpn_input_tensor_values[b * cpn_input_size + i + 0 * cpn_input_h * cpn_input_w] = (float)uc_pixel[2] / 255.0;
                cpn_input_tensor_values[b * cpn_input_size + i + 1 * cpn_input_h * cpn_input_w] = (float)uc_pixel[1] / 255.0;
                cpn_input_tensor_values[b * cpn_input_size + i + 2 * cpn_input_h * cpn_input_w] = (float)uc_pixel[0] / 255.0;
                uc_pixel += 3;
                ++i;
            }
        }
    }

    // 4. create input tensor object from data values
    auto cpn_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<const char *> cpn_input_node_names = {"input"};
    std::vector<const char *> cpn_output_node_names = {"output"};
    std::vector<int64_t> cpn_input_node_dims;
    size_t cpn_num_input_nodes = this->cpn_session->GetInputCount();
    assert(cpn_num_input_nodes == 1);
    Ort::TypeInfo cpn_type_info = this->cpn_session->GetInputTypeInfo(0);
    auto cpn_tensor_info = cpn_type_info.GetTensorTypeAndShapeInfo();
    cpn_input_node_dims = cpn_tensor_info.GetShape();
    // modify the input node dims. because cpn_input_node_dims[0] is equal to -1.
    cpn_input_node_dims[0] = ok_bbox_nums;
    Ort::Value cpn_input_tensor = Ort::Value::CreateTensor<float>(cpn_memory_info,
                                                                  cpn_input_tensor_values.data(),
                                                                  cpn_input_tensor_values.size(),
                                                                  cpn_input_node_dims.data(), 4);
    assert(cpn_input_tensor.IsTensor());

    // 5. inference
    timeStart = (double)getTickCount();
    auto cpn_output_tensors = this->cpn_session->Run(Ort::RunOptions{nullptr}, cpn_input_node_names.data(),
                                                     &cpn_input_tensor, 1, cpn_output_node_names.data(), 1);
    nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
    cout << "cpn running time: " << nTime << " sec\n" << endl;
    assert(cpn_output_tensors.size() == 1 && cpn_output_tensors.front().IsTensor());

    // 6. parser results
    // get output shape and dimensions infos
    vector<int>   max_indexes(ok_bbox_nums * this->cpn_output_c);
    vector<int> max_indexes_x(ok_bbox_nums * this->cpn_output_c);
    vector<int> max_indexes_y(ok_bbox_nums * this->cpn_output_c);
    int cpn_output_h = this->cpn_input_h / downsample;
    int cpn_output_w = this->cpn_input_w / downsample;
    int output_size = cpn_output_h * cpn_output_w * this->cpn_output_c;
    float* results_2 = cpn_output_tensors[0].GetTensorMutableData<float>();
    for(int b=0; b<ok_bbox_nums; b++){
        for(int c=0; c<this->cpn_output_c; c++){
            float max_value = -100000.0;
            float output_value = 0.0;
            for(int h=0; h<(this->cpn_input_h/downsample); h++){
                for(int w=0; w<(this->cpn_input_w/downsample); w++){
                    output_value = results_2[b*output_size + c*cpn_output_h*cpn_output_w+h*cpn_output_w+w];
                    if(max_value < output_value){
                        max_value = output_value;
                        max_indexes[b*this->cpn_output_c+c] = h*cpn_output_w+w;
                    }
                }
            }
            max_indexes_x[b*this->cpn_output_c+c] = (max_indexes[b*this->cpn_output_c+c] % cpn_output_h) * downsample;
            max_indexes_y[b*this->cpn_output_c+c] = (max_indexes[b*this->cpn_output_c+c] / cpn_output_w) * downsample;
        }
    }

    // convert to orig shape
    for(int i=0; i<ok_bbox_nums; i++){
        int top = parameter_dict_list[i]["pad_t"];
        int left = parameter_dict_list[i]["pad_l"];
        float scale = parameter_dict_list[i]["scale"];
        for(int j=0; j<this->cpn_output_c; j++){
            max_indexes_x[i * this->cpn_output_c + j] -= left;
            max_indexes_y[i * this->cpn_output_c + j] -= top;
            max_indexes_x[i * this->cpn_output_c + j]  = float(max_indexes_x[i * this->cpn_output_c + j]) / scale;
            max_indexes_y[i * this->cpn_output_c + j]  = float(max_indexes_y[i * this->cpn_output_c + j]) / scale;
            max_indexes_x[i * this->cpn_output_c + j] += ok_bboxes[i][0];
            max_indexes_y[i * this->cpn_output_c + j] += ok_bboxes[i][1];
        }
    }

    // output result
    // ok result
    for(int i=0; i<ok_bbox_nums; i++){
        // construct ok bbox
        Rect2f rect;
        rect.x = all_bboxes[i][0];
        rect.y = all_bboxes[i][1];
        rect.width = all_bboxes[i][2] - all_bboxes[i][0];
        rect.height = all_bboxes[i][3] - all_bboxes[i][1];
        result.bboxes.push_back(rect);

        // construct ng(1) or ok(0),
        result.state.push_back(0);
    }

    for(int i=0; i<max_indexes_x.size(); i++){
        result.x.push_back(max_indexes_x[i]);
        result.y.push_back(max_indexes_y[i]);
    }

    // ng result
    for(int i=ok_bbox_nums; i<filter_num_bboxes; i++){
        // construct ok bbox
        Rect2f rect;
        rect.x = all_bboxes[i][0];
        rect.y = all_bboxes[i][1];
        rect.width = all_bboxes[i][2] - all_bboxes[i][0];
        rect.height = all_bboxes[i][3] - all_bboxes[i][1];
        result.bboxes.push_back(rect);

        // construct ng(1) or ok(0),
        result.state.push_back(1);
    }

    if(debug){
        for(int i=0; i<filter_num_bboxes; i++){
            cout << "x = " << result.bboxes[i].x << endl;
            cout << "y = " << result.bboxes[i].y << endl;
            cout << "w = " << result.bboxes[i].width << endl;
            cout << "h = " << result.bboxes[i].height << endl;
            cout << "------------" << endl;
            rectangle(result_src, Point(int(result.bboxes[i].x), int(result.bboxes[i].y)),
                           Point(int(result.bboxes[i].x + result.bboxes[i].width),
                            int(result.bboxes[i].y + result.bboxes[i].height)),
                      Scalar(255, 255, 0), 5);
        }

        namedWindow("result_bboxes", WINDOW_NORMAL);
        imshow("result_bboxes", result_src);
        waitKey(0);

    }

    // debug
    if(debug){
        for(int i=0; i<ok_bbox_nums; i++){
            for(int j=0; j<4; j++){
                circle(src, Point(max_indexes_x[i * 4 + j], max_indexes_y[i * 4 + j]),
                       10, Scalar(0,255,0),-1);
            }
        }

        namedWindow("yolo_cpn_result", WINDOW_NORMAL);
        imshow("yolo_cpn_result", src);
        imwrite("../results/final_result.jpg", src);
        waitKey(0);
    }
}
