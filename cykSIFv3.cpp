#include "cykSIFv3.h"
#include "cykTools.h"
#include "cykSDH.h"
#include <cmath>
#include <iomanip>

using namespace std;
using namespace arma;

cykSIF::cykSIF(){
    annonum = 68;
    nn = 20;
    imgnum = 2000;
    lamda_final_train = "1,1000,1000,1000,1000,1,1,1;";
    sc_in_each_stage = "0.7, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3;";
    train_scale = 80.0;
    idx_pool_size_ = 50;
    // idx_pool_tube = "25,20,15,15,15,15,15,15,15,15,15;";
    // idx_pool_.load("idx_pool.dat");
}

// ========================  train ==========================
// ========================  train ==========================
// ========================  train ==========================
// ========================  train ==========================
// ========================  train ==========================
// ========================  train ==========================
// ========================  train ==========================
// ========================  train ==========================

void cykSIF::init(int n_stages, int idx_pool_size, int son_pool_size){ // only init for training
    cout << "train initialising ... " << endl;
    n_stages_ = n_stages;
    idx_pool_size_ = idx_pool_size;
    son_pool_size_ = son_pool_size;
    // idx_pool_.load(pool_file_path);
    // if (idx_pool_.n_rows == 0)
    // {
    //     cout << "error opening: " << pool_file_path << endl;
    //     exit(1);
    // }
    genDesPool(); // generate whole pool for later selecting.
    son_pool_ = zeros(son_pool_size_, annonum*4, n_stages_); // pre malloc son_pool_
    A = zeros(son_pool_size_*annonum*2, annonum*2, n_stages_); // pre malloc A
    indexed_features = zeros(imgnum*nn, idx_pool_.n_rows);  // pre malloc idx-f for all traindata
    face_last_stage = zeros(imgnum*nn, annonum*2);          // pre malloc trainlabel's all face
    dist_diff = face_last_stage;                            // pre malloc final trainlabel
    final_features = zeros(indexed_features.n_rows, son_pool_size_*annonum*2);
    // cout << idx_pool_.n_rows << endl;
    // indexed_features = zeros(imgnum*nn, idx_pool_.n_rows * (idx_pool_.n_rows+1) / 2);
    cout << "initialisation OK ." << endl;
}

void cykSIF::train(int n_stages, int idx_pool_size, int son_pool_size){
    // cout << atan2(1,2)*180/3.1415926 << endl;
    init(n_stages, idx_pool_size, son_pool_size);
    for (int i = 0; i < n_stages_; ++i)
    {
        cout << "======= stage "<< i+1 << " ======="<< endl;
        prepare_data(i);
        select_son_pool(i); // select pcs from idxed-features: 
        sdh_train(i);
        stage_train(i);
        cout << "======================="<<endl;
        test(i, false);
    }
    save_model();
}

void cykSIF::prepare_data(int n_stage){
    cout << "preparing data ..." << endl;
    // mat eye_angle_all = zeros(imgnum*nn,1);
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

    arma::mat anno = cyk.readMat((data_path+"train/anno.mat").c_str());
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
        double sc = sqrt((anno(t,39)-anno(t,45))*(anno(t,39)-anno(t,45)) + 
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
            mat face;
            if (n_stage == 0)
            {
                face = cyk.readMat("../../common/avr_face68.mat");
                face *= 2.36 * cyk.random(0.92, 1.08) * train_scale; // scale random
                double theta = cyk.random(-13,13)/180*3.1415926; // rotation random
                for (int j=0;j<annonum;j++)
                {
                    double tmp = face(0,j);
                    face(0,j) = face(0,j)*cos(theta) + face(0,j+annonum)*sin(theta);
                    face(0,j+annonum) = face(0,j+annonum)*cos(theta) - tmp*sin(theta);
                }
                mat cxg = mean(face.cols(0,annonum-1), 1); // find center of face
                mat cyg = mean(face.cols(annonum, 2*annonum-1), 1);
                mat cx = mean(anno.row(t).cols(0,annonum-1), 1);
                mat cy = mean(anno.row(t).cols(annonum, 2*annonum-1), 1);
                mat dx = cx - cxg;
                mat dy = cy - cyg;
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
            // double eye_angle = -atan2(face(0,45+annonum)-face(0,36+annonum), face(0,45)-face(0,36));//弧度 逆时针方向
            // // eye_angle_all(t*nn+k,0)= eye_angle;
            // // 修正idx_pool_, 得到indexed_features:
            arma::mat idx_pool_t = idx_pool_.slice(n_stage);
            // idx_pool_t.col(1) = idx_pool_.slice(n_stage).col(1)*cos(eye_angle) + idx_pool_.slice(n_stage).col(2)*sin(eye_angle);
            // idx_pool_t.col(2) = idx_pool_.slice(n_stage).col(2)*cos(eye_angle) - idx_pool_.slice(n_stage).col(1)*sin(eye_angle);
            // // cout << eye_angle/3.1415926*180 << endl;
            for (int i = 0; i < idx_pool_t.n_rows; ++i)
            {
                double x_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1));
                double y_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)+annonum) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1)+annonum);
                // idx_pool_t(i,1) += face(0, idx_pool_t(i,0));                
                // idx_pool_t(i,2) += face(0, idx_pool_t(i,0)+annonum);
                if (0>x_i || 0>y_i || y_i>=gray.rows || x_i>=gray.cols)
                    indexed_features(t*nn+k, i) = 128;
                else{
                    indexed_features(t*nn+k, i) = gray.ptr(y_i)[(int)x_i];
                // cout << (int)gray.ptr(idx_pool_t(i,2))[(int)idx_pool_t(i,1)] << endl;
                }
            }
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
    // mat dist_diff = face_last_stage;
    char buffer[256];
    ifstream in("conf.txt"); // read dataset file path
    if (! in.is_open())
    {
        cout << "Error opening conf.txt "<< endl;
        exit(1);
    }
    in.getline(buffer, 200);
    string data_path = buffer;

    mat anno = cyk.readMat((data_path+"train/anno.mat").c_str());
    mat delta_face;
    for (int t=0; t<imgnum; t++)
    {
        double sc = sqrt((anno(t,39)-anno(t,45))*(anno(t,39)-anno(t,45)) + 
            (anno(t,39+annonum)-anno(t,45+annonum))*(anno(t,39+annonum)-anno(t,45+annonum)))/train_scale;
        anno.row(t) /= sc;
        for (int k=0; k<nn; k++) {
            delta_face = anno.row(t) - face_last_stage.row(t*nn+k);
            double eye_angle = atan2(face_last_stage(t*nn+k,45+annonum)-face_last_stage(t*nn+k,36+annonum),
                face_last_stage(t*nn+k,45)-face_last_stage(t*nn+k,36));//弧度 逆时针方向
            for (int i = 0; i < annonum; ++i)
            {
                double tmp1 = delta_face(0, i);
                double tmp2 = delta_face(0, i+annonum);
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
    mat face = cyk.readMat("../../common/avr_face68.mat");
    arma::mat idx_pool_t = idx_pool_.slice(n_stage);
    double ds = sc_in_each_stage(0, n_stage) * ((face(0,39)-face(0,45))*(face(0,39)-face(0,45)) + 
                (face(0,39+annonum)-face(0,45+annonum))*(face(0,39+annonum)-face(0,45+annonum)));
    // cout << "ds: " << ds << endl;
    
    for (int i_anno = 0; i_anno < annonum; ++i_anno)
    {   
        for (int idx_son = 0; idx_son < son_pool_size_; ++idx_son)
        {
            int tmp_idx;
            double x_i, y_i;
            double x_af = face(0, i_anno);
            double y_af = face(0, i_anno + annonum);
            do{
                tmp_idx = floor(cyk.random(0,indexed_features.n_cols));
                x_i = idx_pool_t(tmp_idx, 2) * face(0, idx_pool_t(tmp_idx, 0)) + (1 - idx_pool_t(tmp_idx, 2)) * face(0, idx_pool_t(tmp_idx, 1));
                y_i = idx_pool_t(tmp_idx, 2) * face(0, idx_pool_t(tmp_idx, 0)+annonum) + (1 - idx_pool_t(tmp_idx, 2)) * face(0, idx_pool_t(tmp_idx, 1)+annonum);
            }while(((x_af - x_i)*(x_af - x_i) + (y_af - y_i)*(y_af - y_i)) > ds);
            // cout << "xixi" << endl;
            son_pool_(idx_son, (i_anno+annonum)*2, n_stage) = tmp_idx;
            do{
                tmp_idx = floor(cyk.random(0,indexed_features.n_cols));
                x_i = idx_pool_t(tmp_idx, 2) * face(0, idx_pool_t(tmp_idx, 0)) + (1 - idx_pool_t(tmp_idx, 2)) * face(0, idx_pool_t(tmp_idx, 1));
                y_i = idx_pool_t(tmp_idx, 2) * face(0, idx_pool_t(tmp_idx, 0)+annonum) + (1 - idx_pool_t(tmp_idx, 2)) * face(0, idx_pool_t(tmp_idx, 1)+annonum);
            }while(((x_af - x_i)*(x_af - x_i) + (y_af - y_i)*(y_af - y_i)) > ds);
            son_pool_(idx_son, (i_anno+annonum)*2+1, n_stage) = tmp_idx;
        }
        // cout << "haha" << endl;
    }
    // cout << "xixi "<< endl;

    // // faster edition
    // mat cov_target_pixel = cov(dist_diff, indexed_features);
    // mat var_trainlabel = var(dist_diff);
    // mat cov_pixel_pixel = cov(indexed_features);
    // mat son_pool_corr = zeros(son_pool_size_, dist_diff.n_cols);
    // int tmp_idx;

    // cyktools cyk;
    // mat face = cyk.readMat("../../common/avr_face68.mat");
    // arma::mat idx_pool_t = idx_pool_.slice(n_stage);
    // double ds = sc_in_each_stage(0, n_stage) * ((face(0,39)-face(0,45))*(face(0,39)-face(0,45)) + 
    //             (face(0,39+annonum)-face(0,45+annonum))*(face(0,39+annonum)-face(0,45+annonum)));

    // for (int i_anno = 0; i_anno < annonum; ++i_anno)
    // {
    //     // cout << "# " << i_anno << ": " << endl;
    //     double x_af = face(0, i_anno);
    //     double y_af = face(0, i_anno + annonum);
    //     for (int i = 0; i < indexed_features.n_cols; ++i)
    //     {
    //         double x_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1));
    //         double y_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)+annonum) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1)+annonum);
    //         if (((x_af - x_i)*(x_af - x_i) + (y_af - y_i)*(y_af - y_i)) > ds)
    //             continue;
    //         // cout << "# " << i_anno << ": " << i << "/" << indexed_features.n_cols << endl;
    //         for (int j = i; j < indexed_features.n_cols; ++j)
    //         {
    //             double x_i = idx_pool_t(j, 2) * face(0, idx_pool_t(j, 0)) + (1 - idx_pool_t(j, 2)) * face(0, idx_pool_t(j, 1));
    //             double y_i = idx_pool_t(j, 2) * face(0, idx_pool_t(j, 0)+annonum) + (1 - idx_pool_t(j, 2)) * face(0, idx_pool_t(j, 1)+annonum);
    //             if (((x_af - x_i)*(x_af - x_i) + (y_af - y_i)*(y_af - y_i)) > ds)
    //                 continue;
    //             // mat diff_feature = indexed_features.col(i) - indexed_features.col(j);
    //             double corr = abs((cov_target_pixel(i_anno, i) - cov_target_pixel(i_anno, j))/
    //                 sqrt((cov_pixel_pixel(i,i)+cov_pixel_pixel(j,j)-2.0*cov_pixel_pixel(i,j))*
    //                     var_trainlabel(0,i_anno)));
    //             tmp_idx = 0;
    //             for (int k = 0; k < son_pool_size_; ++k)
    //             {
    //                 if (son_pool_corr(tmp_idx, i_anno) > son_pool_corr(k, i_anno))
    //                 {
    //                     tmp_idx = k;
    //                 }
    //             }
    //             if (son_pool_corr(tmp_idx, i_anno) < corr)
    //             {
    //                 son_pool_corr(tmp_idx, i_anno) = corr;
    //                 son_pool_(tmp_idx, i_anno*2, n_stage) = i;
    //                 son_pool_(tmp_idx, i_anno*2+1, n_stage) = j;
    //             }

    //             corr = abs((cov_target_pixel((i_anno+annonum), i) - cov_target_pixel((i_anno+annonum), j))/
    //                 sqrt((cov_pixel_pixel(i,i)+cov_pixel_pixel(j,j)-2.0*cov_pixel_pixel(i,j))*
    //                     var_trainlabel(0,(i_anno+annonum))));
    //             tmp_idx = 0;
    //             for (int k = 0; k < son_pool_size_; ++k)
    //             {
    //                 if (son_pool_corr(tmp_idx, (i_anno+annonum)) > son_pool_corr(k, (i_anno+annonum)))
    //                 {
    //                     tmp_idx = k;
    //                 }
    //             }
    //             if (son_pool_corr(tmp_idx, (i_anno+annonum)) < corr)
    //             {
    //                 son_pool_corr(tmp_idx, (i_anno+annonum)) = corr;
    //                 son_pool_(tmp_idx, (i_anno+annonum)*2, n_stage) = i;
    //                 son_pool_(tmp_idx, (i_anno+annonum)*2+1, n_stage) = j;
    //             }
    //         }
    //     }
    // }

    // mat mean_son_pool_corr = mean(son_pool_corr);
    // mean_son_pool_corr.print("mean_son_pool_corr");
}

void cykSIF::sdh_train(int n_stage){
    cout << "SDH training ..." << endl;
    gen_final_features(n_stage); // 串联所有的son_pool_ 得到全局特征
    
    // cykSDH cyk_sdh;
    // cyk_sdh.init(final_features, dist_diff, 32, 20, 1, 100, 0.0001);
    // cyk_sdh.train("SDH_model_");
}

void cykSIF::gen_final_features(int n_stage){ // 串联所有的son_pool_ 得到全局特征
    for (int i_anno = 0; i_anno < annonum*2; ++i_anno)
    {
        for (int i = 0; i < son_pool_size_; ++i)
        {
            final_features.col(son_pool_size_*i_anno+i) = 
                indexed_features.col(son_pool_(i, i_anno*2, n_stage)) - 
                indexed_features.col(son_pool_(i, i_anno*2+1, n_stage));
        }
    }
    final_features = sign(final_features);
}

void cykSIF::stage_train(int n_stage){
    cout << "stage training (LSR) ..." << n_stage << endl;
    cyktools cyk;
    // final_features = normalise(final_features, 2, 1);
    A.slice(n_stage) = cyk.LSR(final_features, dist_diff, lamda_final_train(0,n_stage));
    // A.save("A.dat");
}

// ========================  test ==========================
// ========================  test ==========================
// ========================  test ==========================
// ========================  test ==========================
// ========================  test ==========================
// ========================  test ==========================
// ========================  test ==========================

void cykSIF::test(int n_stage, bool if_training){
    // if (! if_training){
    //     load_model();
    //     indexed_features.load("indexed_features.dat");
    //     face_last_stage.load("face_last_stage.dat");
    // }
    update_face(n_stage);
    show_pre(face_last_stage, face_current_stage, if_training);
}

void cykSIF::update_face(int n_stage){ // update face using final indexed-diff-features
    gen_final_features(n_stage);
    // final_features = normalise(final_features, 2, 1);
    mat pre = final_features * A.slice(n_stage);
    mat eye_angle = -atan((face_last_stage.col(45+annonum)-face_last_stage.col(36+annonum))/
        (face_last_stage.col(45)-face_last_stage.col(36)));//弧度 逆时针方向
    for (int i = 0; i < annonum; ++i)
    {
        mat tmp1 = pre.col(i);
        mat tmp2 = pre.col(i+annonum);
        pre.col(i) = tmp1%cos(eye_angle) + tmp2%sin(eye_angle); // element-wise
        pre.col(i+annonum) = tmp2%cos(eye_angle) - tmp1%sin(eye_angle);
    }
    face_current_stage = face_last_stage + pre;
    // return current_face;
}

void cykSIF::show_pre(mat& lst_face, mat& crt_face, bool if_training){ // only for train
    cout << "show ! "<< endl;
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
    arma::mat anno = cyk.readMat((data_path+"train/anno.mat").c_str());

    double dis1 = 0;
    double dis2 = 0;
    mat err1 = lst_face;
    mat err2 = crt_face;
    for (int i = 0; i < anno.n_rows; ++i)
    {
        double sc = sqrt((anno(i,39)-anno(i,45))*(anno(i,39)-anno(i,45)) + 
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

    if ( ! if_training)
    {
        anno = cyk.readMat((data_path+"train/anno.mat").c_str());
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
            double sc = sqrt((anno(t,39)-anno(t,45))*(anno(t,39)-anno(t,45)) + 
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

                if_training = true;
                break;
                if(cv::waitKey(0)==27){
                    if_training = true; // end showing 
                    break;
                    // exit(1);// -1;
                }
            }
            if (if_training)
            {
                break;
            }
        }   
    }
}

void cykSIF::draw_face(cv::Mat& pano, mat face, cv::Scalar sc){
    for (int ccc=0; ccc<annonum; ccc++)
    {
        cv::circle(pano, cv::Point(face(0, ccc), face(0, ccc+annonum)), 2, sc);
    }

}

void cykSIF::genDesPool(){
    // cv::Mat img_;
    // arma::mat idx_pool_;

    cyktools cyk;
    // nfl_ = anno.n_cols / 2; // number of face landmards
    // std::cout << "idx_pool_size_: " << idx_pool_size_ << ", " << 3 << std::endl;
    idx_pool_ = arma::zeros(annonum * idx_pool_size_, 3, n_stages_); // feature pool (0:idx, 1:dx, 2:dy)
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
    A.save("A.dat");
    son_pool_.save("son_pool_.dat");
    idx_pool_.save("SIF_pool.dat");
    cout << "save ok." << endl;
}

void cykSIF::load_model(){
    A.load("A.dat");
    son_pool_.load("son_pool_.dat");
    idx_pool_.load("SIF_pool.dat");

    n_stages_ = A.n_slices;
    idx_pool_size_ = idx_pool_.n_rows / annonum;
    son_pool_size_ = son_pool_.n_rows;

    indexed_features = zeros(1, idx_pool_.n_rows);
    final_features = zeros(1, son_pool_size_*annonum*2);
    face_last_stage = zeros(1, annonum*2);
    dist_diff = face_last_stage;
}


// ========================  test dataset ==========================
// ========================  test dataset ==========================
// ========================  test dataset ==========================
// ========================  test dataset ==========================
// ========================  test dataset ==========================
// ========================  test dataset ==========================
// ========================  test dataset ==========================

void cykSIF::test_dataset(int imgn){
    cyktools cyk;

    imgnum = 330; // for helen test dataset
    nn = 1;
    // mat init_face;
    // init_face.load("face_init.dat");

    load_model();

    char buffer[256];
    ifstream in("conf.txt"); // read dataset file path
    if (! in.is_open())
    {
        cout << "Error opening conf.txt "<< endl;
        exit(1);
    }
    in.getline(buffer, 200);
    string data_path = buffer;

    arma::mat anno = cyk.readMat((data_path+"test/anno.mat").c_str());
    for (int t=imgn; t<imgnum; t++)
    {
        cout << "Dealing with "<< t+1 << "th image. "<<endl;
        stringstream ss;
        string c;
        ss << t+1;
        ss >> c;
        cv::Mat img = cv::imread(data_path+"test/"+c+".jpg");
        if (img.empty()) {
            cout << "Image loading error!"<<endl;
            exit(1);
        }
        double sc = sqrt((anno(t,39)-anno(t,45))*(anno(t,39)-anno(t,45)) + 
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
            mat face;
            face = cyk.readMat("../../common/avr_face68.mat");
            face *= 2.36 * cyk.random(0.92, 1.08) * train_scale; // scale random
            double theta = cyk.random(-13,13)/180*3.1415926; // rotation random
            for (int j=0;j<annonum;j++)
            {
                double tmp = face(0,j);
                face(0,j) = face(0,j)*cos(theta) + face(0,j+annonum)*sin(theta);
                face(0,j+annonum) = face(0,j+annonum)*cos(theta) - tmp*sin(theta);
            }
            mat cxg = mean(face.cols(0,annonum-1), 1); // find center of face
            mat cyg = mean(face.cols(annonum, 2*annonum-1), 1);
            mat cx = mean(anno.row(t).cols(0,annonum-1), 1);
            mat cy = mean(anno.row(t).cols(annonum, 2*annonum-1), 1);
            mat dx = cx - cxg;
            mat dy = cy - cyg;
            // dx += cyk.random(-8,8);
            // dy += cyk.random(-8,8);
            face.cols(0,annonum-1) += dx(0,0);
            face.cols(annonum, annonum*2-1) += dy(0,0);  

            // face = init_face.row(t*nn+k);

            if(pred_pic(gray, face, true) == true)
                return ;
        }
    }
            // arma::mat idx_pool_t = idx_pool_.slice(n_stage);
            // for (int i = 0; i < idx_pool_t.n_rows; ++i)
            // {
            //     double x_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1));
            //     double y_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)+annonum) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1)+annonum);
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

bool cykSIF::pred_pic(cv::Mat& gray, arma::mat face, bool if_show){
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

void cykSIF::gen_all_features(int n_stage, cv::Mat& gray, mat face){
    // // 以两眼连线作为水平线，将随机生成的初始脸放正（但并不对原始图像进行旋转，不然太慢）
    // // 计算两眼连线角度：
    // double eye_angle = -atan2(face(0,45+annonum)-face(0,36+annonum), face(0,45)-face(0,36));//弧度 逆时针方向
    // // eye_angle_all(t*nn+k,0)= eye_angle;
    // // 修正idx_pool_, 得到indexed_features:
    arma::mat idx_pool_t = idx_pool_.slice(n_stage);
    // idx_pool_t.col(1) = idx_pool_.slice(n_stage).col(1)*cos(eye_angle) + idx_pool_.slice(n_stage).col(2)*sin(eye_angle);
    // idx_pool_t.col(2) = idx_pool_.slice(n_stage).col(2)*cos(eye_angle) - idx_pool_.slice(n_stage).col(1)*sin(eye_angle);
    // // cout << eye_angle/3.1415926*180 << endl;
    for (int i = 0; i < idx_pool_t.n_rows; ++i)
    {
        double x_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1));
        double y_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)+annonum) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1)+annonum);
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











