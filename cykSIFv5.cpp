#include "cykSIFv5.h"
#include "cykTools.h"
#include <cmath>
#include <iomanip>
#include <time.h>

using namespace std;
using namespace arma;

cykSIF::cykSIF(){
    annonum = 68;
    nn = 20;
    imgnum = 3148;
    idx_pool_size_ = 100;
    son_pool_size_ = 40;
    train_scale = 80.0;
    // a = NULL;
    // lamda_final_train = "1,1000,100,10,1;";
    // sc_in_each_stage = "0.7, 0.3, 0.3, 0.2, 0.2;";
    // idx_pool_tube = "25,20,15,15,15,15,15,15,15,15,15;";
    // idx_pool_.load("idx_pool.dat");
    // a = new float**[5];
    // for (int i = 0; i < 5; ++i)
    // {
    //     a[i] = new float*[5441];
    //     for (int j = 0; j < 5441; ++j)
    //     {
    //         a[i][j] = new float[136];
    //     }
    // }
}

cykSIF::~cykSIF() {
    // for (int i = 0; i < 5; ++i)
    // {
        
    //     for (int j = 0; j < 5441; ++j)
    //     {
    //         delete[] a[i][j];
    //     }
    //     delete[] a[i];
    // }
    // delete[] a;
}

// cykSIF::cykSIF(float (*a_)[5441][136]){
//     annonum = 68;
//     nn = 20;
//     imgnum = 3148;
//     train_scale = 80.0;
//     a = a_; 
//     // lamda_final_train = "1,1000,100,10,1;";
//     // sc_in_each_stage = "0.7, 0.3, 0.3, 0.2, 0.2;";
//     // idx_pool_size_ = 50;
//     // idx_pool_tube = "25,20,15,15,15,15,15,15,15,15,15;";
//     // idx_pool_.load("idx_pool.dat");
// }

// ========================  train ==========================
// ========================  train ==========================
// ========================  train ==========================
// ========================  train ==========================
// ========================  train ==========================
// ========================  train ==========================
// ========================  train ==========================
// ========================  train ==========================

void cykSIF::init(int n_stages, int idx_pool_size, int son_pool_size, fmat lft, fmat sies, float ts){ // only init for training
    

    cout << "train initialising ... " << endl;

    lamda_final_train = lft;
    sc_in_each_stage = sies;
    train_scale = ts;

    n_stages_ = n_stages;
    idx_pool_size_ = idx_pool_size;
    son_pool_size_ = son_pool_size;

    indexed_features = zeros<fmat>(imgnum*nn, annonum * idx_pool_size_);  // pre malloc idx-f for all traindata
    final_features = zeros<fmat>(indexed_features.n_rows, son_pool_size_*annonum*2 + 1);
    face_last_stage = zeros<fmat>(imgnum*nn, annonum*2);          // pre malloc trainlabel's all face
    dist_diff = face_last_stage;                            // pre malloc final trainlabel

    son_pool_ = zeros<fcube>(son_pool_size_, annonum*4, n_stages_); // pre malloc son_pool_
    A = zeros<fcube>(final_features.n_cols, annonum*2, n_stages_); // pre malloc A
    // sdh_model = zeros<fcube>(son_pool_size_, son_pool_size_*annonum*2, n_stages_);// pre malloc sdh model
    // cout << idx_pool_.n_rows << endl;
    // indexed_features = zeros<fmat>(imgnum*nn, idx_pool_.n_rows * (idx_pool_.n_rows+1) / 2);
    cout << "|==================================================================================================" <<endl;
    cout << "|          <<<    TRAIN PARAMETERS    >>> \t" << endl;
    cout << "|   n_stages:      \t" << n_stages << endl;
    cout << "|   idx_pool_size: \t" << idx_pool_size << endl;
    cout << "|   son_pool_size: \t" << son_pool_size << endl;
    cout << "|   lamda_final_train: " << lamda_final_train;
    cout << "|   sc_in_each_stage:  " << sc_in_each_stage;
    cout << "|   train_scale: \t" << train_scale << endl;
    cout << "|==================================================================================================" <<endl;
    ofstream flog("log_v5_no_sdh.txt", ios::app);
    flog << "|==================================================================================================" <<endl;
    flog << "|          <<<    TRAIN PARAMETERS    >>> \t" << endl;
    flog << "|   n_stages:      \t" << n_stages << endl;
    flog << "|   idx_pool_size: \t" << idx_pool_size << endl;
    flog << "|   son_pool_size: \t" << son_pool_size << endl;
    flog << "|   lamda_final_train: " << lamda_final_train;
    flog << "|   sc_in_each_stage:  " << sc_in_each_stage;
    flog << "|   train_scale: \t" << train_scale << endl;
    flog << "|" << endl;
    flog.close();
    cout << "initialisation OK ." << endl;
}

void cykSIF::train_only_stage(int n_stage, int idx_pool_size, int son_pool_size, fmat lft, fmat sies, float ts){
    load_model(false);
    lamda_final_train = lft;
    sc_in_each_stage = sies;
    train_scale = ts;
    idx_pool_size_ = idx_pool_size;
    son_pool_size_ = son_pool_size;
    indexed_features = zeros<fmat>(imgnum*nn, annonum * idx_pool_size_);  // pre malloc idx-f for all traindata
    final_features = zeros<fmat>(indexed_features.n_rows, son_pool_size_*annonum*2 + 1);
    face_last_stage = zeros<fmat>(imgnum*nn, annonum*2);          // pre malloc trainlabel's all face
    dist_diff = face_last_stage;                            // pre malloc final trainlabel

    ofstream flog("log_v5_no_sdh.txt", ios::app);
    cout << "|======================================= train_only_stage =========================================" <<endl;
    cout << "| # stage: " << n_stage << endl;
    cout << "|          <<<    TRAIN PARAMETERS    >>> \t" << endl;
    cout << "|   n_stages:      \t" << n_stages_ << endl;
    cout << "|   idx_pool_size: \t" << idx_pool_size << endl;
    cout << "|   son_pool_size: \t" << son_pool_size << endl;
    cout << "|   lamda_final_train: " << lamda_final_train;
    cout << "|   sc_in_each_stage:  " << sc_in_each_stage;
    cout << "|   train_scale: \t" << train_scale << endl;
    cout << "|==================================================================================================" <<endl;
    flog << "|======================================= train_only_stage =========================================" <<endl;
    flog << "| # stage: " << n_stage << endl;
    flog << "|          <<<    TRAIN PARAMETERS    >>> \t" << endl;
    flog << "|   n_stages:      \t" << n_stages_ << endl;
    flog << "|   idx_pool_size: \t" << idx_pool_size << endl;
    flog << "|   son_pool_size: \t" << son_pool_size << endl;
    flog << "|   lamda_final_train: " << lamda_final_train;
    flog << "|   sc_in_each_stage:  " << sc_in_each_stage;
    flog << "|   train_scale: \t" << train_scale << endl;
    flog << "|" << endl;
    flog.close();

    for (int i = 0; i < n_stage-1; ++i)
    {
        prepare_data(i);
        test(i, false);
    }
    for (int i = n_stage-1; i < n_stage; ++i)
    {
        cout << "======= stage "<< i+1 << " ======="<< endl;
        prepare_data(i);
        select_son_pool(i); // select pcs from idxed-features: 
        sdh_train(i);       
        stage_train(i);
        cout << " ---------------------"<<endl;
        test(i, false);
    }
    save_model();
    cout << "======= train end ======="<< endl;
}

void cykSIF::prepare_data(int n_stage){
    cout << "preparing data ..." << endl;
    // fmat eye_angle_all = zeros<fmat>(imgnum*nn,1);
    cyktools cyk;

    char buffer[256];
    ifstream in("conf.txt"); // read dataset file path
    if (! in.is_open())
    {
        cout << "Error opening conf.txt "<< endl;
        exit(1);
    }
    in.getline(buffer, 200);
    string data_path = buffer;

    arma::fmat anno;
    anno.load(data_path+"train/annof.dat");
    for (int t=0; t<imgnum; t++)
    {
        // cout << "Dealing with "<< t+1 << "th image. "<<endl;
        stringstream ss;
        string c;
        ss << t+1;
        ss >> c;
        cv::Mat img = cv::imread(data_path+"train/"+c+".jpg");
        if (img.empty()) {
            cout << "Image loading error!"<<endl;
            exit(1);
        }
        float sc = sqrt((anno(t,39)-anno(t,45))*(anno(t,39)-anno(t,45)) + 
            (anno(t,39+annonum)-anno(t,45+annonum))*(anno(t,39+annonum)-anno(t,45+annonum)))/train_scale;
        cv::resize(img, img, cv::Size(img.cols/sc, img.rows/sc));
        anno.row(t) /= sc;

        cv::Mat gray;
        if (img.channels() == 3)
        {
            cv::cvtColor(img, gray, CV_RGB2GRAY);
        }else{
            gray = img.clone();
        }

        for (int k=0; k<nn; k++) {
            fmat face;
            if (n_stage == 0)
            {
                // mat tmp = cyk.readMat("../../common/avr_face68.mat");
                // face = conv_to<fmat>::from(tmp);
                face.load("../../common/avr_face68_f.dat");
                face *= 2.36 * cyk.random(0.92, 1.08) * train_scale; // scale random
                float theta = cyk.random(-13,13)/180*3.1415926; // rotation random
                for (int j=0;j<annonum;j++)
                {
                    float tmp = face(0,j);
                    face(0,j) = face(0,j)*cos(theta) + face(0,j+annonum)*sin(theta);
                    face(0,j+annonum) = face(0,j+annonum)*cos(theta) - tmp*sin(theta);
                }
                fmat cxg = mean(face.cols(0,annonum-1), 1); // find center of face
                fmat cyg = mean(face.cols(annonum, 2*annonum-1), 1);
                fmat cx = mean(anno.row(t).cols(0,annonum-1), 1);
                fmat cy = mean(anno.row(t).cols(annonum, 2*annonum-1), 1);
                fmat dx = cx - cxg;
                fmat dy = cy - cyg;
                // dx += cyk.random(-8,8);
                // dy += cyk.random(-8,8);
                face.cols(0,annonum-1) += dx(0,0);
                face.cols(annonum, annonum*2-1) += dy(0,0);   

                // cv::Mat ddddd = img.clone();
                // cout << theta << endl;
                // draw_face(ddddd, face, cv::Scalar(0,255,0));
                // cv::imshow("ddddd", ddddd);
                // cv::waitKey();
            }
            else{
                face = face_current_stage.row(t*nn+k);
            }
            // // 以两眼连线作为水平线，将随机生成的初始脸放正（但并不对原始图像进行旋转，不然太慢）
            // // 计算两眼连线角度：
            // float eye_angle = -atan2(face(0,45+annonum)-face(0,36+annonum), face(0,45)-face(0,36));//弧度 逆时针方向
            // // eye_angle_all(t*nn+k,0)= eye_angle;
            // // 修正idx_pool_, 得到indexed_features:
            arma::fmat idx_pool_t = idx_pool_.slice(n_stage);
            // idx_pool_t.col(1) = idx_pool_.slice(n_stage).col(1)*cos(eye_angle) + idx_pool_.slice(n_stage).col(2)*sin(eye_angle);
            // idx_pool_t.col(2) = idx_pool_.slice(n_stage).col(2)*cos(eye_angle) - idx_pool_.slice(n_stage).col(1)*sin(eye_angle);
            // // cout << eye_angle/3.1415926*180 << endl;
            // cout << 1 << endl;
            for (int i = 0; i < idx_pool_t.n_rows; ++i)
            {
                float x_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1));
                float y_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)+annonum) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1)+annonum);
                // idx_pool_t(i,1) += face(0, idx_pool_t(i,0));                
                // idx_pool_t(i,2) += face(0, idx_pool_t(i,0)+annonum);
                if (0>x_i || 0>y_i || y_i>=gray.rows || x_i>=gray.cols)
                    indexed_features(t*nn+k, i) = 128;
                else{
                    indexed_features(t*nn+k, i) = gray.ptr(y_i)[(int)x_i];
                // cout << (int)gray.ptr(idx_pool_t(i,2))[(int)idx_pool_t(i,1)] << endl;
                }
            }
            // cout << 2 << endl;
            face_last_stage.row(t*nn+k) = face; // 直接保存整个脸坐标
            // // 得到face_last_stage: 相对初始脸眼连线坐标系
            // face_last_stage.row(t*nn+k) = anno.row(t) - face;

// #define SHOW
// #ifdef SHOW
//             cv::Mat pano = img.clone();
//             for (int ccc=0; ccc<annonum; ccc++)
//             {
//                 cv::circle(pano, cv::Point(anno(t, ccc), anno(t, ccc+annonum)), 2, cv::Scalar(0,255,255));
//                 cv::circle(pano, cv::Point(face(0, ccc), face(0, ccc+annonum)), 2, cv::Scalar(0,0,255));
//             }
//             cv::imshow("pano", pano);

//             if(cv::waitKey()==27)
//                 exit(1);// -1;
// #endif
        }
    }
    // face_last_stage.save("face_last_stage_"+itoa(n_stage)+".dat");
    // if (n_stage == 0)
    // {
    //     face_last_stage.save("face_init.dat");
    // }
    gen_dist_diff(); // 映射到face坐标系下
    // indexed_features.save("indexed_features.dat");
    // face_last_stage.save("face_last_stage.dat");
    // indexed_features.load("indexed_features.dat");
    // face_last_stage.load("face_last_stage.dat");
    // cout << "indexed_features:  " << indexed_features.n_rows << ", " << indexed_features.n_cols << endl;
    // cout << "face_last_stage: " << face_last_stage.n_rows << ", " << face_last_stage.n_cols << endl;
}

void cykSIF::gen_dist_diff(){// 映射到face坐标系下的 delta face 用 -angle, this func. is only for train
    cyktools cyk;
    // fmat dist_diff = face_last_stage;
    char buffer[256];
    ifstream in("conf.txt"); // read dataset file path
    if (! in.is_open())
    {
        cout << "Error opening conf.txt "<< endl;
        exit(1);
    }
    in.getline(buffer, 200);
    string data_path = buffer;

    fmat anno;
    anno.load(data_path+"train/annof.dat");// = cyk.readMat(().c_str());
    fmat delta_face;
    for (int t=0; t<imgnum; t++)
    {
        float sc = sqrt((anno(t,39)-anno(t,45))*(anno(t,39)-anno(t,45)) + 
            (anno(t,39+annonum)-anno(t,45+annonum))*(anno(t,39+annonum)-anno(t,45+annonum)))/train_scale;
        anno.row(t) /= sc;
        for (int k=0; k<nn; k++) {
            delta_face = anno.row(t) - face_last_stage.row(t*nn+k);
            float eye_angle = atan2(face_last_stage(t*nn+k,45+annonum)-face_last_stage(t*nn+k,36+annonum),
                face_last_stage(t*nn+k,45)-face_last_stage(t*nn+k,36));//弧度 逆时针方向
            for (int i = 0; i < annonum; ++i)
            {
                float tmp1 = delta_face(0, i);
                float tmp2 = delta_face(0, i+annonum);
                dist_diff(t*nn+k, i) = tmp1*cos(eye_angle) + tmp2*sin(eye_angle);
                dist_diff(t*nn+k, i+annonum) = tmp2*cos(eye_angle) - tmp1*sin(eye_angle);
            }
        }
    }
    // return dist_diff;
}

void cykSIF::select_son_pool(int n_stage){ // select features for max correlation with trainlabel
    cout << "selecting son pool ... " << endl;

    cyktools cyk;
    // mat tmp = cyk.readMat("../../common/avr_face68.mat");
    // fmat face = conv_to<fmat>::from(tmp);
    fmat face;
    face.load("../../common/avr_face68_f.dat");
    arma::fmat idx_pool_t = idx_pool_.slice(n_stage);
    float ds = sc_in_each_stage(0, n_stage) * ((face(0,39)-face(0,45))*(face(0,39)-face(0,45)) + 
                (face(0,39+annonum)-face(0,45+annonum))*(face(0,39+annonum)-face(0,45+annonum)));
    // cout << "ds: " << ds << endl;
    
    for (int i_anno = 0; i_anno < annonum; ++i_anno)
    {   
        for (int idx_son = 0; idx_son < son_pool_size_; ++idx_son)
        {
            int tmp_idx;
            float x_i, y_i;
            float x_af = face(0, i_anno);
            float y_af = face(0, i_anno + annonum);
            for (int i = 0; i < 4; ++i)
            {
                do{
                    tmp_idx = floor(cyk.random(0,indexed_features.n_cols));
                    x_i = idx_pool_t(tmp_idx, 2) * face(0, idx_pool_t(tmp_idx, 0)) + (1 - idx_pool_t(tmp_idx, 2)) * face(0, idx_pool_t(tmp_idx, 1));
                    y_i = idx_pool_t(tmp_idx, 2) * face(0, idx_pool_t(tmp_idx, 0)+annonum) + (1 - idx_pool_t(tmp_idx, 2)) * face(0, idx_pool_t(tmp_idx, 1)+annonum);
                }while(((x_af - x_i)*(x_af - x_i) + (y_af - y_i)*(y_af - y_i)) > ds);
                son_pool_(idx_son, i_anno*4+i, n_stage) = tmp_idx;
            }
        }
        // cout << "haha" << endl;
    }
    // cout << "xixi "<< endl;

}

void cykSIF::sdh_train(int n_stage){
    cout << "SDH training ..." << endl;
    for (int i_anno = 0; i_anno < annonum*2; ++i_anno)
    {
        for (int i = 0; i < son_pool_size_; ++i)
        {
            final_features.col(son_pool_size_*i_anno+i) = 
                indexed_features.col(son_pool_(i, i_anno*2, n_stage)) - 
                indexed_features.col(son_pool_(i, i_anno*2+1, n_stage));
        }
    }
    
    // fmat td = zeros<fmat>(final_features.n_rows, son_pool_size_);
    // fmat tl = zeros<fmat>(final_features.n_rows, 1);
    // SDH cyk_sdh;
    // for (int i = 0; i < annonum*2; ++i)
    // {
    //     td = final_features.cols(i*son_pool_size_, i*son_pool_size_+son_pool_size_-1);
    //     tl = dist_diff.col(i);
    //     cyk_sdh.init(td, tl, son_pool_size_, 15, 1, 100, 0.001);
    //     sdh_model.slice(n_stage).cols(i*son_pool_size_, i*son_pool_size_+son_pool_size_-1) = cyk_sdh.train("...", false, false);
    //     final_features.cols(i*son_pool_size_, i*son_pool_size_+son_pool_size_-1) =
    //         final_features.cols(i*son_pool_size_, i*son_pool_size_+son_pool_size_-1) *
    //         sdh_model.slice(n_stage).cols(i*son_pool_size_, i*son_pool_size_+son_pool_size_-1);
    // }
    final_features = sign(final_features);
}

void cykSIF::stage_train(int n_stage){
    cout << "stage training (LSR) ... for LSRf not working well >> stage: " << n_stage << endl;
    cyktools cyk;
    // final_features = normalise(final_features, 2, 1);
    // A.slice(n_stage) = cyk.LSRf(final_features, dist_diff, lamda_final_train(0,n_stage));

    mat td = conv_to<mat>::from(final_features);
    mat tl = conv_to<mat>::from(dist_diff);
    mat a = cyk.LSR(td, tl, lamda_final_train(0,n_stage));
    A.slice(n_stage) = conv_to<fmat>::from(a);
    // A.load("A_f.dat");
    // A.save("A.dat");
}

// ========================  test ==========================
// ========================  test ==========================
// ========================  test ==========================
// ========================  test ==========================
// ========================  test ==========================
// ========================  test ==========================
// ========================  test ==========================

void cykSIF::test(int n_stage, bool if_show){
    // if (! if_training){
    //     load_model();
    //     indexed_features.load("indexed_features.dat");
    //     face_last_stage.load("face_last_stage.dat");
    // }
    update_face(n_stage);
    show_pre(face_last_stage, face_current_stage, if_show);
}

void cykSIF::update_face(int n_stage){ // update face using final indexed-diff-features
    gen_final_features(n_stage);
    // final_features = normalise(final_features, 2, 1);
    // cout << A.n_slices << endl;
    fmat pre = final_features * A.slice(n_stage);
    fmat eye_angle = -atan((face_last_stage.col(45+annonum)-face_last_stage.col(36+annonum))/
        (face_last_stage.col(45)-face_last_stage.col(36)));//弧度 逆时针方向
    for (int i = 0; i < annonum; ++i)
    {
        fmat tmp1 = pre.col(i);
        fmat tmp2 = pre.col(i+annonum);
        pre.col(i) = tmp1%cos(eye_angle) + tmp2%sin(eye_angle); // element-wise
        pre.col(i+annonum) = tmp2%cos(eye_angle) - tmp1%sin(eye_angle);
    }
    face_current_stage = face_last_stage + pre;
    // return current_face;
}

void cykSIF::gen_final_features(int n_stage){ // 串联所有的son_pool_ 得到全局特征 && sdh prediction
    // cout << son_pool_.n_slices << endl;
    for (int i_anno = 0; i_anno < annonum*2; ++i_anno)
    {
        for (int i = 0; i < son_pool_size_; ++i)
        {
            final_features.col(son_pool_size_*i_anno+i) = 
                indexed_features.col(son_pool_(i, i_anno*2, n_stage)) - 
                indexed_features.col(son_pool_(i, i_anno*2+1, n_stage));
        }
    }
    // for (int i = 0; i < annonum*2; ++i)
    // {
    //     final_features.cols(i*son_pool_size_, i*son_pool_size_+son_pool_size_-1) = 
    //         final_features.cols(i*son_pool_size_, i*son_pool_size_+son_pool_size_-1) *
    //         sdh_model.slice(n_stage).cols(i*son_pool_size_, i*son_pool_size_+son_pool_size_-1);
    // }
    final_features = sign(final_features);
}

void cykSIF::show_pre(fmat& lst_face, fmat& crt_face, bool if_show){ // only for train
    // cout << "show ! "<< endl;
    cyktools cyk;
    char buffer[256];
    ifstream in("conf.txt"); // read dataset file path
    if (! in.is_open())
    {
        cout << "Error opening conf.txt "<< endl;
        exit(1);
    }
    in.getline(buffer, 200);
    string data_path = buffer;
    arma::fmat anno;
    anno.load(data_path+"train/annof.dat"); //= cyk.readMat(().c_str());

    float dis1 = 0;
    float dis2 = 0;
    fmat err1 = lst_face;
    fmat err2 = crt_face;
    for (int i = 0; i < anno.n_rows; ++i)
    {
        float sc = sqrt((anno(i,39)-anno(i,45))*(anno(i,39)-anno(i,45)) + 
            (anno(i,39+annonum)-anno(i,45+annonum))*(anno(i,39+annonum)-anno(i,45+annonum)))/train_scale;
        anno.row(i) /= sc;
        for (int j = 0; j < nn; ++j)
        {

            err1.row(i*nn+j) = anno.row(i) - lst_face.row(i*nn+j);
            err2.row(i*nn+j) = anno.row(i) - crt_face.row(i*nn+j);
        }
    }
    for (int i = 0; i < err1.n_rows; ++i)
    {
        for (int j = 0; j < annonum; ++j)
        {
            dis1 += sqrt(err1(i,j)*err1(i,j)+err1(i,j+annonum)*err1(i,j+annonum));
            dis2 += sqrt(err2(i,j)*err2(i,j)+err2(i,j+annonum)*err2(i,j+annonum));
        }
    }
    dis1 /= err1.n_rows * annonum;
    dis2 /= err2.n_rows * annonum;
    cout << "last stage err: " << dis1*100/train_scale << "%" << endl;
    cout << "current stage err: " << dis2*100/train_scale << "%" << endl;

    ofstream flog("log_v5_no_sdh.txt", ios::app);
    flog << "|  stage err: "<< dis2*100/train_scale  << ", last stage: " <<dis1*100/train_scale << endl;
    flog.close();

    if (if_show)
    {
        anno.load(data_path+"train/annof.dat");// = cyk.readMat((data_path+"train/anno.mat").c_str());
        for (int t=0; t<imgnum; t++)
        {
            // cout << "Dealing with "<< t+1 << "th image. "<<endl;
            stringstream ss;
            string c;
            ss << t+1;
            ss >> c;
            cv::Mat img = cv::imread(data_path+"train/"+c+".jpg");
            if (img.empty()) {
                cout << "Image loading error!"<<endl;
                exit(1);
            }
            float sc = sqrt((anno(t,39)-anno(t,45))*(anno(t,39)-anno(t,45)) + 
                (anno(t,39+annonum)-anno(t,45+annonum))*(anno(t,39+annonum)-anno(t,45+annonum)))/train_scale;
            cv::resize(img, img, cv::Size(img.cols/sc, img.rows/sc));
            anno.row(t) /= sc;

            // cv::Mat gray;
            // if (img.channels() == 3)
            // {
            //     cv::cvtColor(img, gray, CV_RGB2GRAY);
            // }else{
            //     gray = img.clone();
            // }

            for (int k=0; k<nn; k++) {
                cout << "testing " << t << "th - " << k << " img" << endl;
                cv::Mat pano = img.clone();
                // draw_face(pano, lst_face.row(t*nn+k), cv::Scalar(0,255,255));
                draw_face(pano, crt_face.row(t*nn+k), cv::Scalar(0,0,255));
                draw_face(pano, anno.row(t), cv::Scalar(200,100,0));
                cv::imshow("pano", pano);

                if(cv::waitKey(0)==27){
                    if_show = false; // end showing 
                    break;
                    // exit(1);// -1;
                }
            }
            if (if_show)
            {
                break;
            }
        }   
    }
}

void cykSIF::draw_face(cv::Mat& pano, fmat face, cv::Scalar sc){
    for (int ccc=0; ccc<annonum; ccc++)
    {
        cv::circle(pano, cv::Point(face(0, ccc), face(0, ccc+annonum)), 2, sc);
    }

}

void cykSIF::genDesPool(){
    // cv::Mat img_;
    // arma::fmat idx_pool_;

    cyktools cyk;
    // nfl_ = anno.n_cols / 2; // number of face landmards
    // std::cout << "idx_pool_size_: " << idx_pool_size_ << ", " << 3 << std::endl;
    idx_pool_ = arma::zeros<fcube>(annonum * idx_pool_size_, 3, n_stages_); // feature pool (0:idx, 1:dx, 2:dy)
    for (int ns = 0; ns < n_stages_; ++ns)
    {
        for (int i = 0; i < annonum * idx_pool_size_; ++i)
        {
            idx_pool_(i, 0, ns) = cyk.random(0, annonum); // random point 1
            idx_pool_(i, 1, ns) = cyk.random(0, annonum); // random point 2
            idx_pool_(i, 2, ns) = cyk.random(-0.5,1.5); // random para.
        }
        // cout << "stage" << endl;
    }
}

void cykSIF::save_model(){
    cout << "saving model ... " << endl;
    A.save("A_f.dat");
    son_pool_.save("son_pool_f.dat");
    idx_pool_.save("SIF_pool_f.dat");
    // sdh_model.save("sdh_model.dat");
    cout << "save ok." << endl;
}

void cykSIF::load_model(bool if_final_use){
    // cube ta;
    // ta.load("A.dat");
    // cube ts1;
    // ts1.load("son_pool_.dat");
    // cube ts2;
    // ts2.load("SIF_pool.dat");

    // A = conv_to<fcube>::from(ta);
    // son_pool_ = conv_to<fcube>::from(ts1);
    // idx_pool_ = conv_to<fcube>::from(ts2);

    // A.save("A_f.dat");
    // son_pool_.save("son_pool_f.dat");
    // idx_pool_.save("SIF_pool_f.dat");

    cout << "Loading model ... "<<endl;
    A.load("A_f.dat");
    son_pool_.load("son_pool_f.dat");
    idx_pool_.load("SIF_pool_f.dat");

    // for (int i = 0; i < 5; ++i)
    // {
    //     for (int j = 0; j < 5441; ++j)
    //     {
    //         for (int k = 0; k < 136; ++k)
    //         {
    //             // (*(a+i))[j][k] = A.slice(i)(j, k);
    //             a[i][j][k] = A.slice(i)(j,k);
    //         }
    //     }
    // }
    cout << "model load OK."<<endl;
    // sdh_model.load("sdh_model.dat");

    n_stages_ = A.n_slices;
    idx_pool_size_ = idx_pool_.n_rows / annonum;
    son_pool_size_ = son_pool_.n_rows;

    if (if_final_use)
    {
        indexed_features = zeros<fmat>(1, idx_pool_.n_rows);
        final_features = zeros<fmat>(1, son_pool_size_*annonum*2 + 1);
        face_last_stage = zeros<fmat>(1, annonum*2);
        dist_diff = face_last_stage;
    }
}


// ========================  test dataset ==========================
// ========================  test dataset ==========================
// ========================  test dataset ==========================
// ========================  test dataset ==========================
// ========================  test dataset ==========================
// ========================  test dataset ==========================
// ========================  test dataset ==========================

void cykSIF::test_dataset(int imgn, bool if_show, int ni){
    cyktools cyk;

    imgnum = 554; // all 689, common 554, train 3148
    nn = 1;
    // fmat init_face;
    // init_face.load("face_init.dat");

    load_model();
    fmat total_err = zeros<fmat>((imgnum-imgn)*nn, annonum*2);
    float failure_count = 0;
    fmat diff_one_landmark_distribution = zeros<fmat>(1, 100);
    fmat diff_one_face_distribution = zeros<fmat>(1, 100);

    char buffer[256];
    ifstream in("conf.txt"); // read dataset file path
    if (! in.is_open())
    {
        cout << "Error opening conf.txt "<< endl;
        exit(1);
    }
    in.getline(buffer, 200);
    string data_path = buffer;

    arma::fmat anno;
    anno.load(data_path+"test/annof.dat");// = cyk.readMat((data_path+"test/anno.mat").c_str());
    for (int t=imgn; t<imgnum; t++)
    {
        // cout << "Dealing with "<< t+1 << "th image. "<<endl;
        stringstream ss;
        string c;
        ss << t+1;
        ss >> c;
        cv::Mat img = cv::imread(data_path+"test/"+c+".jpg");
        if (img.empty()) {
            cout << "Image loading error!"<<endl;
            exit(1);
        }
        float sc = sqrt((anno(t,39)-anno(t,45))*(anno(t,39)-anno(t,45)) + 
            (anno(t,39+annonum)-anno(t,45+annonum))*(anno(t,39+annonum)-anno(t,45+annonum)))/train_scale;
        cv::resize(img, img, cv::Size(img.cols/sc, img.rows/sc));
        anno.row(t) /= sc;

        cv::Mat gray;
        if (img.channels() == 3)
        {
            cv::cvtColor(img, gray, CV_RGB2GRAY);
        }else{
            gray = img.clone();
        }

        for (int k=0; k<nn; k++) {
            fmat face;
            // mat tmp = cyk.readMat("../../common/avr_face68.mat");
            // face = conv_to<fmat>::from(tmp);
            face.load("../../common/avr_face68_f.dat");
            // face *= 2.36 * cyk.random(0.92, 1.08) * train_scale; // scale random
            // float theta = cyk.random(-13,13)/180*3.1415926; // rotation random
            face *= 2.36  * train_scale; // scale random
            float theta = 0;
            for (int j=0;j<annonum;j++)
            {
                float tmp = face(0,j);
                face(0,j) = face(0,j)*cos(theta) + face(0,j+annonum)*sin(theta);
                face(0,j+annonum) = face(0,j+annonum)*cos(theta) - tmp*sin(theta);
            }
            fmat cxg = mean(face.cols(0,annonum-1), 1); // find center of face
            fmat cyg = mean(face.cols(annonum, 2*annonum-1), 1);
            fmat cx = mean(anno.row(t).cols(0,annonum-1), 1);
            fmat cy = mean(anno.row(t).cols(annonum, 2*annonum-1), 1);
            fmat dx = cx - cxg;
            fmat dy = cy - cyg;
            // dx += cyk.random(-8,8);
            // dy += cyk.random(-8,8);
            face.cols(0,annonum-1) += dx(0,0);
            face.cols(annonum, annonum*2-1) += dy(0,0);  

            // face = init_face.row(t*nn+k);

            // if(pred_pic(gray, face, true) == true)
                // return ;

            test_one(gray, face, if_show, ni);

            float diff_one_face = 0;

            total_err.row((t-imgn)*nn+k) = anno.row(t) - face_current_stage;
            for (int j = 0; j < annonum; ++j)
            {
                float diff_one_landmark = sqrt(total_err((t-imgn)*nn+k,j)*total_err((t-imgn)*nn+k,j)+
                    total_err((t-imgn)*nn+k,j+annonum)*total_err((t-imgn)*nn+k,j+annonum));

                diff_one_face += diff_one_landmark;

                if (diff_one_landmark >= 99)
                    diff_one_landmark_distribution(0,99)++;
                else
                    diff_one_landmark_distribution(0, floor(diff_one_landmark))++;
                if ( diff_one_landmark > 0.1 * train_scale)
                    failure_count ++;
            }
            diff_one_face /= annonum;
            if (diff_one_face >= 99)
                diff_one_face_distribution(0,99)++;
            else
                diff_one_face_distribution(0, floor(diff_one_face))++;
            // if (diff_one_face > 8)
            // {
            //     cv::Mat pano = gray.clone();
            //     cv::cvtColor(pano, pano, CV_GRAY2RGB);
            //     // draw_face(pano, face_last_stage, cv::Scalar(200,200,0));
            //     draw_face(pano, face_current_stage, cv::Scalar(100,0,255));
            //     cv::imshow("pano", pano);
            //     cv::waitKey();
            // }
        }
    }
    // diff_one_landmark_distribution.print("diff_one_landmark_distribution");
    // diff_one_face_distribution.print("diff_one_face_distribution");

    float dis1 = 0;
    for (int i = 0; i < total_err.n_rows; ++i)
    {
        for (int j = 0; j < annonum; ++j)
        {
            dis1 += sqrt(total_err(i,j)*total_err(i,j)+total_err(i,j+annonum)*total_err(i,j+annonum));
        }
    }
    cout << "failure: " << failure_count / total_err.n_rows / annonum * 100 << " %"<< endl;
    dis1 /= total_err.n_rows * annonum;
    cout << "total err: " << dis1*100/train_scale << "%" << endl;
    ofstream flog("log_v5_no_sdh.txt", ios::app);
    flog << "|                     >>>>     total err on testset: " << dis1*100/train_scale << "%" << endl;
    flog << "|==================================================================================================" <<endl;
    flog.close();

    mat total_err_double = conv_to<mat>::from(total_err);;
    // cyk.writeMat("total_err_test_4.Mat", total_err_double, "total_err");
    // face_last_stage
            // arma::fmat idx_pool_t = idx_pool_.slice(n_stage);
            // for (int i = 0; i < idx_pool_t.n_rows; ++i)
            // {
            //     float x_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1));
            //     float y_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)+annonum) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1)+annonum);
            //     // idx_pool_t(i,1) += face(0, idx_pool_t(i,0));                
            //     // idx_pool_t(i,2) += face(0, idx_pool_t(i,0)+annonum);
            //     if (0>x_i || 0>y_i || y_i>=gray.rows || x_i>=gray.cols)
            //         indexed_features(t*nn+k, i) = 128;
            //     else
            //         indexed_features(t*nn+k, i) = gray.ptr(y_i)[(int)x_i];
            //     // cout << (int)gray.ptr(idx_pool_t(i,2))[(int)idx_pool_t(i,1)] << endl;
            // }
            // face_last_stage.row(t*nn+k) = face; // 直接保存整个脸坐标
}

void cykSIF::test_dataset_multi_init(int imgn, bool if_show, int ni){
    cyktools cyk;

    imgnum = 554; // all 689, common 554, train 3148
    nn = 1;
    // fmat init_face;
    // init_face.load("face_init.dat");

    load_model();
    fmat total_err = zeros<fmat>((imgnum-imgn)*nn, annonum*2);
    float failure_count = 0;
    fmat diff_one_landmark_distribution = zeros<fmat>(1, 100);
    fmat diff_one_face_distribution = zeros<fmat>(1, 100);

    char buffer[256];
    ifstream in("conf.txt"); // read dataset file path
    if (! in.is_open())
    {
        cout << "Error opening conf.txt "<< endl;
        exit(1);
    }
    in.getline(buffer, 200);
    string data_path = buffer;

    arma::fmat anno;
    anno.load(data_path+"test/annof.dat");// = cyk.readMat((data_path+"test/anno.mat").c_str());
    for (int t=imgn; t<imgnum; t++)
    {
        // cout << "Dealing with "<< t+1 << "th image. "<<endl;
        stringstream ss;
        string c;
        ss << t+1;
        ss >> c;
        cv::Mat img = cv::imread(data_path+"test/"+c+".jpg");
        if (img.empty()) {
            cout << "Image loading error!"<<endl;
            exit(1);
        }
        float sc = sqrt((anno(t,39)-anno(t,45))*(anno(t,39)-anno(t,45)) + 
            (anno(t,39+annonum)-anno(t,45+annonum))*(anno(t,39+annonum)-anno(t,45+annonum)))/train_scale;
        cv::resize(img, img, cv::Size(img.cols/sc, img.rows/sc));
        anno.row(t) /= sc;

        cv::Mat gray;
        if (img.channels() == 3)
        {
            cv::cvtColor(img, gray, CV_RGB2GRAY);
        }else{
            gray = img.clone();
        }

        for (int k=0; k<nn; k++) {
            fmat tmp_face = zeros<fmat>(1, annonum*2);
            for (int i = 0; i < ni; ++i)
            {
                fmat face;
                // mat tmp = cyk.readMat("../../common/avr_face68.mat");
                // face = conv_to<fmat>::from(tmp);
                face.load("../../common/avr_face68_f.dat");
                face *= 2.36 * cyk.random(0.95, 1.05) * train_scale; // scale random
                float theta = cyk.random(-10,10)/180*3.1415926; // rotation random
                // face *= 2.36  * train_scale; // scale random
                // float theta = 0;
                for (int j=0;j<annonum;j++)
                {
                    float tmp = face(0,j);
                    face(0,j) = face(0,j)*cos(theta) + face(0,j+annonum)*sin(theta);
                    face(0,j+annonum) = face(0,j+annonum)*cos(theta) - tmp*sin(theta);
                }
                fmat cxg = mean(face.cols(0,annonum-1), 1); // find center of face
                fmat cyg = mean(face.cols(annonum, 2*annonum-1), 1);
                fmat cx = mean(anno.row(t).cols(0,annonum-1), 1);
                fmat cy = mean(anno.row(t).cols(annonum, 2*annonum-1), 1);
                fmat dx = cx - cxg;
                fmat dy = cy - cyg;
                // dx += cyk.random(-8,8);
                // dy += cyk.random(-8,8);
                face.cols(0,annonum-1) += dx(0,0);
                face.cols(annonum, annonum*2-1) += dy(0,0);  

                tmp_face += test_one(gray, face, if_show, -1);
            }
            face_current_stage = tmp_face/ni;

            float diff_one_face = 0;

            total_err.row((t-imgn)*nn+k) = anno.row(t) - face_current_stage;
            for (int j = 0; j < annonum; ++j)
            {
                float diff_one_landmark = sqrt(total_err((t-imgn)*nn+k,j)*total_err((t-imgn)*nn+k,j)+
                    total_err((t-imgn)*nn+k,j+annonum)*total_err((t-imgn)*nn+k,j+annonum));

                diff_one_face += diff_one_landmark;

                if (diff_one_landmark >= 99)
                    diff_one_landmark_distribution(0,99)++;
                else
                    diff_one_landmark_distribution(0, floor(diff_one_landmark))++;
                if ( diff_one_landmark > 0.1 * train_scale)
                    failure_count ++;
            }
            diff_one_face /= annonum;
            if (diff_one_face >= 99)
                diff_one_face_distribution(0,99)++;
            else
                diff_one_face_distribution(0, floor(diff_one_face))++;
            // if (diff_one_face > 8)
            // {
            //     cv::Mat pano = gray.clone();
            //     cv::cvtColor(pano, pano, CV_GRAY2RGB);
            //     // draw_face(pano, face_last_stage, cv::Scalar(200,200,0));
            //     draw_face(pano, face_current_stage, cv::Scalar(100,0,255));
            //     cv::imshow("pano", pano);
            //     cv::waitKey();
            // }
        }
    }
    // diff_one_landmark_distribution.print("diff_one_landmark_distribution");
    // diff_one_face_distribution.print("diff_one_face_distribution");

    float dis1 = 0;
    for (int i = 0; i < total_err.n_rows; ++i)
    {
        for (int j = 0; j < annonum; ++j)
        {
            dis1 += sqrt(total_err(i,j)*total_err(i,j)+total_err(i,j+annonum)*total_err(i,j+annonum));
        }
    }
    cout << "failure: " << failure_count / total_err.n_rows / annonum * 100 << " %"<< endl;
    dis1 /= total_err.n_rows * annonum;
    cout << "total err: " << dis1*100/train_scale << "%" << endl;
    ofstream flog("log_v5_no_sdh.txt", ios::app);
    flog << "|                     >>>>     total err on testset: " << dis1*100/train_scale << "%" << endl;
    flog << "|==================================================================================================" <<endl;
    flog.close();

    mat total_err_double = conv_to<mat>::from(total_err);;
    // cyk.writeMat("total_err_test_4.Mat", total_err_double, "total_err");
    // face_last_stage
            // arma::fmat idx_pool_t = idx_pool_.slice(n_stage);
            // for (int i = 0; i < idx_pool_t.n_rows; ++i)
            // {
            //     float x_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1));
            //     float y_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)+annonum) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1)+annonum);
            //     // idx_pool_t(i,1) += face(0, idx_pool_t(i,0));                
            //     // idx_pool_t(i,2) += face(0, idx_pool_t(i,0)+annonum);
            //     if (0>x_i || 0>y_i || y_i>=gray.rows || x_i>=gray.cols)
            //         indexed_features(t*nn+k, i) = 128;
            //     else
            //         indexed_features(t*nn+k, i) = gray.ptr(y_i)[(int)x_i];
            //     // cout << (int)gray.ptr(idx_pool_t(i,2))[(int)idx_pool_t(i,1)] << endl;
            // }
            // face_last_stage.row(t*nn+k) = face; // 直接保存整个脸坐标
}

bool cykSIF::pred_pic(cv::Mat& gray, arma::fmat face, bool if_show){
    face_last_stage = face;
    for (int si = 0; si < n_stages_; ++si)
    {
        gen_all_features(si, gray, face_last_stage);
        update_face(si);
        if (if_show && si == n_stages_ - 1)
        {
            cv::Mat pano = gray.clone();
            cv::cvtColor(pano, pano, CV_GRAY2RGB);
            // draw_face(pano, face_last_stage, cv::Scalar(200,200,0));
            draw_face(pano, face_current_stage, cv::Scalar(100,0,255));
            cv::imshow("pano", pano);
            if (cv::waitKey() == 27)
                return true;
        }
        face_last_stage = face_current_stage;
    }
    return false;
}

void cykSIF::gen_all_features(int n_stage, cv::Mat& gray, fmat face){
    // // 以两眼连线作为水平线，将随机生成的初始脸放正（但并不对原始图像进行旋转，不然太慢）
    // // 计算两眼连线角度：
    // float eye_angle = -atan2(face(0,45+annonum)-face(0,36+annonum), face(0,45)-face(0,36));//弧度 逆时针方向
    // // eye_angle_all(t*nn+k,0)= eye_angle;
    // // 修正idx_pool_, 得到indexed_features:
    arma::fmat idx_pool_t = idx_pool_.slice(n_stage);
    // idx_pool_t.col(1) = idx_pool_.slice(n_stage).col(1)*cos(eye_angle) + idx_pool_.slice(n_stage).col(2)*sin(eye_angle);
    // idx_pool_t.col(2) = idx_pool_.slice(n_stage).col(2)*cos(eye_angle) - idx_pool_.slice(n_stage).col(1)*sin(eye_angle);
    // // cout << eye_angle/3.1415926*180 << endl;
    for (int i = 0; i < idx_pool_t.n_rows; ++i)
    {
        float x_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1));
        float y_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)+annonum) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1)+annonum);
        // idx_pool_t(i,1) += face(0, idx_pool_t(i,0));                
        // idx_pool_t(i,2) += face(0, idx_pool_t(i,0)+annonum);
        if (0>x_i || 0>y_i || y_i>=gray.rows || x_i>=gray.cols)
            indexed_features(0, i) = 128;
        else
            indexed_features(0, i) = gray.ptr(y_i)[(int)x_i];
        // cout << (int)gray.ptr(idx_pool_t(i,2))[(int)idx_pool_t(i,1)] << endl;
    }
    // gen_dist_diff();
}

fmat cykSIF::test_one(cv::Mat& gray, fmat& face, bool if_show, int ni){
    if (ni < 0)
    {
        ni = n_stages_;
    }

    float start_t = clock();

    face_last_stage = face;
    fmat pre;
    float eye_angle;
    float x_i;
    float y_i;
    // for (int si = 0; si < 3; ++si)
    for (int si = 0; si < ni; ++si)
    {
        // arma::fmat idx_pool_.slice(si) = idx_pool_.slice(si);
        // fmat idx_pool_t = idx_pool_.slice(si);
        for (int i = 0; i < idx_pool_.slice(si).n_rows; ++i) // gen idx-features
        {
            x_i = idx_pool_.slice(si)(i, 2) * face_last_stage(0, idx_pool_.slice(si)(i, 0)) + (1 - idx_pool_.slice(si)(i, 2)) * face_last_stage(0, idx_pool_.slice(si)(i, 1));
            y_i = idx_pool_.slice(si)(i, 2) * face_last_stage(0, idx_pool_.slice(si)(i, 0)+annonum) + (1 - idx_pool_.slice(si)(i, 2)) * face_last_stage(0, idx_pool_.slice(si)(i, 1)+annonum);
            if (0>x_i || 0>y_i || y_i>=gray.rows || x_i>=gray.cols)
                indexed_features(0, i) = 128;
            else
                indexed_features(0, i) = gray.ptr(y_i)[(int)x_i];
        }
        for (int i_anno = 0; i_anno < 136; ++i_anno) // cascade all features
        {
            for (int i = 0; i < son_pool_size_; ++i)
            {
                final_features(0, son_pool_size_*i_anno+i) = 
                    indexed_features(0, son_pool_(i, i_anno+i_anno, si)) - 
                    indexed_features(0, son_pool_(i, i_anno+i_anno+1, si));
            }
        }
        // // for (int i = 0; i < annonum*2; ++i) // for sdh
        // // {
        // //     final_features.cols(i*son_pool_size_, i*son_pool_size_+son_pool_size_-1) = 
        // //         final_features.cols(i*son_pool_size_, i*son_pool_size_+son_pool_size_-1) *
        // //         sdh_model.slice(si).cols(i*son_pool_size_, i*son_pool_size_+son_pool_size_-1);
        // // }
        // // final_features = sign(final_features);
        final_features = sign(final_features);
        pre = final_features * A.slice(si);
        // fmat tmpa = A.slice(si);
        // pre = zeros<fmat>(1, 136); 
        // for (int i = 0; i < final_features.n_cols; ++i)
        // {
        //     if (final_features(0,i)>0)
        //         pre += tmpa.row(i);
        //     else if (final_features(0,i)<0)
        //         pre -= tmpa.row(i);
        // }
        eye_angle = -atan((face_last_stage(0, 113)-face_last_stage(0, 104))/ //68 + 45, 68 + 36
            (face_last_stage(0, 45)-face_last_stage(0, 36)));//弧度 逆时针方向
        for (int i = 0; i < annonum; ++i)
        {
            float tmp1 = pre(0, i);
            float tmp2 = pre(0, i+annonum);
            pre(0, i) = tmp1*cos(eye_angle) + tmp2*sin(eye_angle); // element-wise
            pre(0, i+annonum) = tmp2*cos(eye_angle) - tmp1*sin(eye_angle);
        }
        face_current_stage = face_last_stage + pre;
        face_last_stage = face_current_stage;
    }

    // face_current_stage = face_last_stage;
    // fmat tf = zeros<fmat>(1, son_pool_size_*2+1);
    // for (int si = 0; si < 5; ++si)
    // {
    //     for (int annoi = 0; annoi < annonum; ++annoi)
    //     {
    //         x_i = face_current_stage(0,annoi);
    //         y_i = face_current_stage(0,annoi+annonum);
    //         int idxp = 0;
    //         for (int i = y_i-8; i < y_i+9; i+=2)
    //         {
    //             for (int j = x_i-8; j < x_i+9; j+=2)
    //             {
    //                 if (i<0 || i>gray.rows-2 || j<0 || j>gray.cols-2)
    //                 {
    //                     tf(0, idxp++) = 0;
    //                     continue;
    //                 }
    //                 uchar a1 = gray.ptr(i)[j];
    //                 uchar a2 = gray.ptr(i)[j+1];
    //                 uchar a3 = gray.ptr(i+1)[j];
    //                 uchar a4 = gray.ptr(i+1)[j+1];
    //                 tf(0, idxp++) = abs(a1+a2-a3-a4) + abs(a2-a1+a4-a3);
    //             }
    //         }
    //         fmat m = max(tf, 1);
    //         if (m(0,0) != 0)
    //             tf /= m(0,0);
    //         // tf = sign(tf-0.5);
    //         int fp = annoi*son_pool_size_*2;
    //         final_features.cols(fp, fp+80) = tf;
    //     }
    //     pre  = final_features * A.slice(si);
    //     face_current_stage += pre;    
    // }

    float end_t = clock();

    if (if_show)
    {
        cv::Mat pano = gray.clone();
        cv::cvtColor(pano, pano, CV_GRAY2RGB);
        draw_face(pano, face_current_stage, cv::Scalar(100,0,255));
        cv::imshow("pano", pano);
        cv::waitKey();
    }
    // cout << "reg. cost: "<< (float)(end_t - start_t) /  CLOCKS_PER_SEC * 1000 << "ms"<<endl;
    return face_current_stage;
}

// void cykSIF::train_last_stage(double lmd){
//     load_model(false);
//     cyktools cyk;

//     int sti = 4;
//     cout << "--------------stage "<< sti << " -----------------------" << endl;
//     mat err = cyk.readMat("total_err_train_4.Mat");
//     mat err_test = cyk.readMat("total_err_test_4.Mat");
//     fmat ferr = conv_to<fmat>::from(err);
//     // fmat ferr;
//     // ferr.load("pre_err_f.dat");
//     final_features = zeros<fmat>(ferr.n_rows, son_pool_size_*annonum*2 + 1);
//     mat tdf;

//     char buffer[256];
//     ifstream in("conf.txt"); // read dataset file path
//     if (! in.is_open())
//     {
//         cout << "Error opening conf.txt "<< endl;
//         exit(1);
//     }
//     in.getline(buffer, 200);
//     string data_path = buffer;

//     arma::fmat anno;
//     anno.load(data_path+"train/annof.dat");
//     for (int t=0; t<imgnum; t++)
//     {
//         // cout << "Dealing with "<< t+1 << "th image. "<<endl;
//         stringstream ss;
//         string c;
//         ss << t+1;
//         ss >> c;
//         cv::Mat img = cv::imread(data_path+"train/"+c+".jpg");
//         if (img.empty()) {
//             cout << "Image loading error!"<<endl;
//             exit(1);
//         }
//         float sc = sqrt((anno(t,39)-anno(t,45))*(anno(t,39)-anno(t,45)) + 
//             (anno(t,39+annonum)-anno(t,45+annonum))*(anno(t,39+annonum)-anno(t,45+annonum)))/train_scale;
//         cv::resize(img, img, cv::Size(img.cols/sc, img.rows/sc));
//         anno.row(t) /= sc;

//         cv::Mat gray;
//         if (img.channels() == 3)
//         {
//             cv::cvtColor(img, gray, CV_RGB2GRAY);
//         }else{
//             gray = img.clone();
//         }

//         for (int k=0; k<nn; k++) {
//             fmat face = anno.row(t) - ferr.row(t*nn+k);
//             for (int annoi = 0; annoi < annonum; ++annoi)
//             {
//                 int x = face(0,annoi);
//                 int y = face(0,annoi+annonum);
//                 fmat tf = zeros<fmat>(1, son_pool_size_*2+1);
//                 int idxp = 0;
//                 for (int i = y-8; i < y+9; i+=2)
//                 {
//                     for (int j = x-8; j < x+9; j+=2)
//                     {
//                         if (i<0 || i>gray.rows-2 || j<0 || j>gray.cols-2)
//                         {
//                             tf(0, idxp++) = 0;
//                             continue;
//                         }
//                         uchar a1 = gray.ptr(i)[j];
//                         uchar a2 = gray.ptr(i)[j+1];
//                         uchar a3 = gray.ptr(i+1)[j];
//                         uchar a4 = gray.ptr(i+1)[j+1];
//                         // cout << annoi << endl;
//                         tf(0, idxp++) = abs(a1+a2-a3-a4) + abs(a2-a1+a4-a3);
//                     }
//                 }
//                 // tf = normalise(tf, 2, 1);
//                 fmat m = max(tf, 1);
//                 if (m(0,0) != 0)
//                     tf /= m(0,0);
//                 // tf = sign(tf-0.5);
//                 // tf = clamp(tf, 0, 1.0);
//                 int fp = annoi*son_pool_size_*2;
//                 final_features.row(t*nn+k).cols(fp, fp+80) = tf;
//             }
//         }
//     }
//     // final_features.row(0).print("ff:");
//     cout << "training ..." << endl;
//     tdf = conv_to<mat>::from(final_features);
//     cout << "lmd: -------> "<< lmd << endl;
//     mat At = cyk.LSR(tdf, err, lmd);
//     A.slice(sti) = conv_to<fmat>::from(At);
//     save_model();
//     cout << "train end." << endl;

//     ferr = conv_to<fmat>::from(err_test);
//     final_features = zeros<fmat>(ferr.n_rows, son_pool_size_*annonum*2 + 1);

//     anno.load(data_path+"test/annof.dat");
//     for (int t=0; t<err_test.n_rows/nn; t++)
//     {
//         // cout << "Dealing with "<< t+1 << "th image. "<<endl;
//         stringstream ss;
//         string c;
//         ss << t+1;
//         ss >> c;
//         cv::Mat img = cv::imread(data_path+"test/"+c+".jpg");
//         if (img.empty()) {
//             cout << "Image loading error!"<<endl;
//             exit(1);
//         }
//         float sc = sqrt((anno(t,39)-anno(t,45))*(anno(t,39)-anno(t,45)) + 
//             (anno(t,39+annonum)-anno(t,45+annonum))*(anno(t,39+annonum)-anno(t,45+annonum)))/train_scale;
//         cv::resize(img, img, cv::Size(img.cols/sc, img.rows/sc));
//         anno.row(t) /= sc;

//         cv::Mat gray;
//         if (img.channels() == 3)
//         {
//             cv::cvtColor(img, gray, CV_RGB2GRAY);
//         }else{
//             gray = img.clone();
//         }

//         for (int k=0; k<nn; k++) {
//             fmat face = anno.row(t) - ferr.row(t*nn+k);
//             for (int annoi = 0; annoi < annonum; ++annoi)
//             {
//                 int x = face(0,annoi);
//                 int y = face(0,annoi+annonum);
//                 fmat tf = zeros<fmat>(1, son_pool_size_*2+1);
//                 int idxp = 0;
//                 for (int i = y-8; i < y+9; i+=2)
//                 {
//                     for (int j = x-8; j < x+9; j+=2)
//                     {
//                         if (i<0 || i>gray.rows-2 || j<0 || j>gray.cols-2)
//                         {
//                             tf(0, idxp++) = 0;
//                             continue;
//                         }
//                         uchar a1 = gray.ptr(i)[j];
//                         uchar a2 = gray.ptr(i)[j+1];
//                         uchar a3 = gray.ptr(i+1)[j];
//                         uchar a4 = gray.ptr(i+1)[j+1];
//                         tf(0, idxp++) = abs(a1+a2-a3-a4) + abs(a2-a1+a4-a3);
//                     }
//                 }
//                 fmat m = max(tf, 1);
//                 if (m(0,0) != 0)
//                     tf /= m(0,0);
//                 // tf = sign(tf-0.5);
//                 int fp = annoi*son_pool_size_*2;
//                 final_features.row(t*nn+k).cols(fp, fp+80) = tf;
//             }
//         }
//     }

//     fmat pre  = final_features * A.slice(3);
//     fmat pre_err = ferr - pre;
    
//     double dis0 = 0;
//     double dis1 = 0;
//     for (int i = 0; i < pre_err.n_rows; ++i)
//     {
//         for (int j = 0; j < annonum; ++j)
//         {
//             dis1 += sqrt(pre_err(i,j)*pre_err(i,j)+pre_err(i,j+annonum)*pre_err(i,j+annonum));
//             dis0 += sqrt(err_test(i,j)*err_test(i,j)+err_test(i,j+annonum)*err_test(i,j+annonum));
//         }
//     // cout << dis1 << ", "<< i << endl;
//     // cv::Mat cccc = cv::Mat::zeros(100,100,CV_8UC1);
//     // cv::imshow("test", cccc);
//     // cv::waitKey();
//     }
//     // cout << dis1 << endl;
//     dis0 /= pre_err.n_rows * annonum;
//     dis1 /= pre_err.n_rows * annonum;
//     cout << "total err_last: " << dis0*100/train_scale << "%" << endl;
//     cout << "total err_test: " << dis1*100/train_scale << "%" << endl;
//     cout << "-----------------------------------------------" << endl;
// }

/*
void cykSIF::train_last_stage(double lg, double lf, double nu)
{
    load_model(false);
    cyktools cyk;

    mat err = cyk.readMat("total_err_train.Mat");
    fmat ferr = conv_to<fmat>::from(err);
    int sdh_feature_size = 40;

    indexed_features = zeros<fmat>(imgnum*nn, annonum * idx_pool_size_);  // pre malloc idx-f for all traindata
    final_features = zeros<fmat>(indexed_features.n_rows, son_pool_size_*annonum*2 + 1);
    // face_last_stage = zeros<fmat>(imgnum*nn, annonum*2);          // pre malloc trainlabel's all face

    // fmat train_features = zeros<fmat>(ferr.n_rows, sdh_feature_size*2*annonum);
    // fmat local_patch = zeros<fmat>(sdh_feature_size, 2);
    // for (int i = 0; i < sdh_feature_size; ++i)
    // {
    //     local_patch(i, 0) = cyk.random(-15, 15);
    //     local_patch(i, 1) = cyk.random(-15, 15);
    // }
    // local_patch.save("local_patch_f.dat");
    fmat local_patch;
    local_patch.load("local_patch_f.dat");
    char buffer[256];
    ifstream in("conf.txt"); // read dataset file path
    if (! in.is_open())
    {
        cout << "Error opening conf.txt "<< endl;
        exit(1);
    }
    in.getline(buffer, 200);
    string data_path = buffer;

    arma::fmat anno;
    anno.load(data_path+"train/annof.dat");
    int nb = son_pool_size_;
    cube sdh_model_final_stage = zeros(son_pool_size_*2, nb, annonum*2);
    // mat final_features_sdh = zeros(ferr.n_rows, nb*annonum*2 + 1);
    int si = 3;
    mat tdf;

    for (int t=0; t<imgnum; t++)
    {
        // cout << "Dealing with "<< t+1 << "th image. "<<endl;
        stringstream ss;
        string c;
        ss << t+1;
        ss >> c;
        cv::Mat img = cv::imread(data_path+"train/"+c+".jpg");
        if (img.empty()) {
            cout << "Image loading error!"<<endl;
            exit(1);
        }
        float sc = sqrt((anno(t,39)-anno(t,45))*(anno(t,39)-anno(t,45)) + 
            (anno(t,39+annonum)-anno(t,45+annonum))*(anno(t,39+annonum)-anno(t,45+annonum)))/train_scale;
        cv::resize(img, img, cv::Size(img.cols/sc, img.rows/sc));
        anno.row(t) /= sc;

        cv::Mat gray;
        if (img.channels() == 3)
        {
            cv::cvtColor(img, gray, CV_RGB2GRAY);
        }else{
            gray = img.clone();
        }

        for (int k=0; k<nn; k++) {
            fmat face = anno.row(t) - ferr.row(t*nn+k);
            arma::fmat idx_pool_t = idx_pool_.slice(si);
            for (int i = 0; i < idx_pool_t.n_rows; ++i)
            {
                float x_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1));
                float y_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)+annonum) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1)+annonum);
                if (0>x_i || 0>y_i || y_i>=gray.rows || x_i>=gray.cols)
                    indexed_features(t*nn+k, i) = 128;
                else{
                    indexed_features(t*nn+k, i) = gray.ptr(y_i)[(int)x_i];
                }
            }

            // for (int i = 0; i < annonum; ++i)
            // {
            //     for (int j = 0; j < sdh_feature_size; ++j)
            //     {
            //         int x = face(0, i) + local_patch(j, 0);
            //         int y = face(0, i+annonum) + local_patch(j, 1);
            //         if (x<0 || y<0 || x>=gray.cols || y>=gray.rows)
            //             train_features(t*nn+k, i*sdh_feature_size+j) = 128;
            //         else
            //             train_features(t*nn+k, i*sdh_feature_size+j) = gray.ptr(y)[x];
            //     }
            // }
        }
    }
    for (int i_anno = 0; i_anno < 136; ++i_anno) // cascade all features
    {
        for (int i = 0; i < son_pool_size_; ++i)
        {
            final_features.col(son_pool_size_*i_anno+i) = 
                indexed_features.col(son_pool_(i, i_anno+i_anno, si)) - 
                indexed_features.col(son_pool_(i, i_anno+i_anno+1, si));
        }
    }
    final_features = sign(final_features);

    SDH cyk_sdh;
    for (int i = 0; i < annonum; ++i)
    {
        // cout << i << endl;
        // mat td = conv_to<mat>::from(train_features.cols(i*sdh_feature_size, i*sdh_feature_size+sdh_feature_size-1));
        mat td = conv_to<mat>::from(final_features.cols(i*son_pool_size_*2, (i+1)*son_pool_size_*2-1));

        mat tl = conv_to<mat>::from(ferr.col(i*2));
        //                          lg  lf  nu
        cyk_sdh.init(td, tl, nb, 25, lg, lf, nu);
        sdh_model_final_stage.slice(2*i) = cyk_sdh.train("...", false, false);

        tl = conv_to<mat>::from(ferr.col(i*2+1));
        cyk_sdh.init(td, tl, nb, 25, lg, lf, nu);
        sdh_model_final_stage.slice(2*i+1) = cyk_sdh.train("...", false, false);
    }
    sdh_model_final_stage.save("sdh_model_final_stage.dat");
    sdh_model_final_stage.load("sdh_model_final_stage.dat");

    for (int i = 0; i < annonum; ++i)
    {
        mat td = conv_to<mat>::from(final_features.cols(i*son_pool_size_*2, (i+1)*son_pool_size_*2-1));
        final_features.cols(2*i*nb, 2*i*nb+nb-1) = conv_to<fmat>::from(td * sdh_model_final_stage.slice(2*i));
        final_features.cols(2*i*nb+nb, 2*i*nb+2*nb-1) = conv_to<fmat>::from(td * sdh_model_final_stage.slice(2*i+1));
    }
    tdf = conv_to<mat>::from(sign(final_features));
    mat At = cyk.LSR(tdf, err, 80000);
    A.slice(3) = conv_to<fmat>::from(At);
    save_model();
    cout << "train end." << endl;

    mat err_test = cyk.readMat("total_err_test.Mat");
    ferr = conv_to<fmat>::from(err_test);
    // train_features = zeros<fmat>(ferr.n_rows, sdh_feature_size*annonum);
    indexed_features = zeros<fmat>(ferr.n_rows, annonum * idx_pool_size_);  // pre malloc idx-f for all traindata
    final_features = zeros<fmat>(indexed_features.n_rows, son_pool_size_*annonum*2 + 1);
    anno.load(data_path+"test/annof.dat");
    for (int t=0; t<err_test.n_rows/nn; t++)
    {
        // cout << "Dealing with "<< t+1 << "th image. "<<endl;
        stringstream ss;
        string c;
        ss << t+1;
        ss >> c;
        cv::Mat img = cv::imread(data_path+"test/"+c+".jpg");
        if (img.empty()) {
            cout << "Image loading error!"<<endl;
            exit(1);
        }
        float sc = sqrt((anno(t,39)-anno(t,45))*(anno(t,39)-anno(t,45)) + 
            (anno(t,39+annonum)-anno(t,45+annonum))*(anno(t,39+annonum)-anno(t,45+annonum)))/train_scale;
        cv::resize(img, img, cv::Size(img.cols/sc, img.rows/sc));
        anno.row(t) /= sc;

        cv::Mat gray;
        if (img.channels() == 3)
        {
            cv::cvtColor(img, gray, CV_RGB2GRAY);
        }else{
            gray = img.clone();
        }

        for (int k=0; k<nn; k++) {
            fmat face = anno.row(t) - ferr.row(t*nn+k);
            arma::fmat idx_pool_t = idx_pool_.slice(si);
            for (int i = 0; i < idx_pool_t.n_rows; ++i)
            {
                float x_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1));
                float y_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)+annonum) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1)+annonum);
                if (0>x_i || 0>y_i || y_i>=gray.rows || x_i>=gray.cols)
                    indexed_features(t*nn+k, i) = 128;
                else{
                    indexed_features(t*nn+k, i) = gray.ptr(y_i)[(int)x_i];
                }
            }
            // // cv::Mat pano = gray.clone();
            // // for (int i = 0; i < annonum; ++i)
            // // {
            // //     cv::circle(pano, cv::Point(face(0, i), face(0, i+annonum)), 2, cv::Scalar(0,255,0));
            // // }
            // for (int i = 0; i < annonum; ++i)
            // {
            //     for (int j = 0; j < sdh_feature_size; ++j)
            //     {
            //         int x = face(0, i) + local_patch(j, 0);
            //         int y = face(0, i+annonum) + local_patch(j, 1);
            //         // cv::circle(pano, cv::Point(x, y), 1, cv::Scalar(0,0,255));
            //         // pano.ptr(y)[x] = 255;
            //         if (x<0 || y<0 || x>=gray.cols || y>=gray.rows)
            //             train_features(t*nn+k, i*sdh_feature_size+j) = 128;
            //         else
            //             train_features(t*nn+k, i*sdh_feature_size+j) = gray.ptr(y)[x];
            //     }
            // }
            // // cv::imshow("pano", pano);
            // // cv::waitKey();
        }
    }
    for (int i_anno = 0; i_anno < 136; ++i_anno) // cascade all features
    {
        for (int i = 0; i < son_pool_size_; ++i)
        {
            final_features.col(son_pool_size_*i_anno+i) = 
                indexed_features.col(son_pool_(i, i_anno+i_anno, si)) - 
                indexed_features.col(son_pool_(i, i_anno+i_anno+1, si));
        }
    }
    final_features = sign(final_features);

    for (int i = 0; i < annonum; ++i)
    {
        mat td = conv_to<mat>::from(final_features.cols(i*son_pool_size_*2, (i+1)*son_pool_size_*2-1));
        final_features.cols(2*i*nb, 2*i*nb+nb-1) = conv_to<fmat>::from(td * sdh_model_final_stage.slice(2*i));
        final_features.cols(2*i*nb+nb, 2*i*nb+2*nb-1) = conv_to<fmat>::from(td * sdh_model_final_stage.slice(2*i+1));
    }
    tdf = conv_to<mat>::from(sign(final_features));

    // mat At = cyk.LSR(tdf, err_test, 5000);
    // A.slice(3) = conv_to<fmat>::from(At);
    // save_model();

    mat pre = tdf * A.slice(3);
    mat pre_err = err_test - pre;
    float dis1 = 0;
    float dis0 = 0;
    for (int i = 0; i < pre_err.n_rows; ++i)
    {
        for (int j = 0; j < annonum; ++j)
        {
            dis1 += sqrt(pre_err(i,j)*pre_err(i,j)+pre_err(i,j+annonum)*pre_err(i,j+annonum));
            dis0 += sqrt(err_test(i,j)*err_test(i,j)+err_test(i,j+annonum)*err_test(i,j+annonum));
        }
    }
    dis1 /= pre_err.n_rows * annonum;
    dis0 /= pre_err.n_rows * annonum;
    cout << "total err_last: " << dis0*100/train_scale << "%" << endl;
    cout << "total err_test: " << dis1*100/train_scale << "%" << endl;
}*/







