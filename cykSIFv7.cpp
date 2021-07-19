#include "cykSIFv7.h"
#include "cykTools.h"
#include <cmath>
#include "cykFerns.h"
#include <time.h>

using namespace std;
using namespace arma;

cykSIF::cykSIF(){
    annonum = 68;
    nn = 20;
    imgnum = 3148;
    // idx_pool_size_ = 100;
    train_scale = 50.0;
    sc_in_each_stage = "0.5, 0.25, 0.20, 0.1, 0.08;";
    data_path = "/Users/ublack/Documents/ublack/Face/Datasets/300-W/";
}

cykSIF::~cykSIF() {
}

float cykSIF::show_err(fmat& total_err, bool if_show){
    float dis1 = 0;
    for (int i = 0; i < total_err.n_rows; ++i)
    {
        for (int j = 0; j < annonum; ++j)
        {
            dis1 += sqrt(total_err(i,j)*total_err(i,j)+total_err(i,j+annonum)*total_err(i,j+annonum));
        }
    }
    dis1 /= total_err.n_rows * annonum;
    cout << "total err: " << dis1*100/train_scale << "%" << endl;

    if (!if_show)
        return dis1*100/train_scale;

    if (total_err.n_rows/nn == 3148)
        data_path = "/Users/ublack/Documents/ublack/Face/Datasets/300-W/train";
    else
        data_path = "/Users/ublack/Documents/ublack/Face/Datasets/300-W/test";
    arma::fmat anno;
    anno.load(data_path+"/annof.dat");
    for (int t=0; t<imgnum; t++)
    {
        // cout << "Dealing with "<< t+1 << "th image. "<<endl;
        stringstream ss;
        string c;
        ss << t+1;
        ss >> c;
        cv::Mat img = cv::imread(data_path+"/"+c+".jpg");
        if (img.empty()) {
            cout << "Image loading error!"<<endl;
            exit(1);
        }
        float sc = sqrt((anno(t,39)-anno(t,45))*(anno(t,39)-anno(t,45)) + 
            (anno(t,39+annonum)-anno(t,45+annonum))*(anno(t,39+annonum)-anno(t,45+annonum)))/train_scale;
        cv::resize(img, img, cv::Size(img.cols/sc, img.rows/sc));
        anno.row(t) /= sc;

        for (int k=0; k<nn; k++) {
            // cout << k << endl;
            fmat face;
            face = anno.row(t) - total_err.row(t*nn+k);
            cv::Mat pano = img.clone();
            draw_face(pano, face, cv::Scalar(255,0,0));
            cv::imshow("pano", pano);
            if (cv::waitKey() == 27)
                return dis1*100/train_scale;
        }
    }
    return dis1*100/train_scale;
}

void cykSIF::draw_face(cv::Mat& pano, fmat face, cv::Scalar sc){
    for (int ccc=0; ccc<annonum; ccc++)
    {
        cv::circle(pano, cv::Point(face(0, ccc), face(0, ccc+annonum)), 2, sc);
    }
}

void cykSIF::save_model(){
    cout << "saving model ... " << endl;
    // A.save("A_f.dat");
    son_pool_.save("son_pool_f.dat");
    idx_pool_.save("SIF_pool_f.dat");
    cout << "save ok." << endl;
}

void cykSIF::load_model(bool if_final_use){
    // cout << "Loading model ... "<<endl;
    // A.load("A_f.dat");
    son_pool_.load("son_pool_f.dat");
    idx_pool_.load("SIF_pool_f.dat");
    // cout << "model load OK."<<endl;

    // n_stages_ = A.n_slices;
    idx_pool_size_ = idx_pool_.n_rows / annonum;
    son_pool_size_ = son_pool_.n_rows;
}

void cykSIF::gen_final_features(int n_stage){ // 串联所有的son_pool_ 得到全局特征 && sdh prediction
    cout << "generating final_features ... "<< endl;
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


void cykSIF::prepare_data(int ni, bool if_train, bool if_save){

    data_path = "/Users/ublack/Documents/ublack/Face/Datasets/300-W/";

    cyktools cyk;
    stringstream ss;
    string save_path;
    // string last_path;
    ss << ni;
    ss >> save_path;
    // ss.clear();
    // ss << ni - 1;
    // ss >> last_path;

    if (!if_train)
    {
        imgnum = 554;
        data_path += "test";
        save_path = "test_stage_"+save_path+"_f";
        // last_path = "test_stage_"+last_path+"_f";
    }
    else
    {
        imgnum = 3148; // 3148 / 2 = 1574
        data_path += "train";
        save_path = "train_stage_"+save_path+"_f";
        // last_path = "train_stage_"+last_path+"_f";
    }

    load_model(false);
    if (ni != 0){
        if (if_save)
        {
            cout << "Loading : " << save_path+".label" <<endl;
            face_current_stage.load(save_path+".label");
        }
        // else
        //     face_current_stage = face_last_stage;
    }
    else
    {
        face_current_stage = zeros<fmat>(imgnum*nn, annonum*2);
    }

    indexed_features = zeros<fmat>(imgnum*nn, annonum * idx_pool_size_);  // pre malloc idx-f for all traindata
    final_features = ones<fmat>(indexed_features.n_rows, son_pool_size_*annonum*2 + 1);

    arma::fmat anno;
    anno.load(data_path+"/annof.dat");
    int cd;
    // if (ni == 0)
    // {
        cout << "Prepare data ..." << endl;
        cout << "|---------------------------------------------------|"<< endl;
        cout << "|>|" << flush;
        cd = imgnum / 50;
    // }
    for (int t=0; t<imgnum; t++)
    {
        // if (ni == 0)
        // {
            if (t == cd)
            {
                // printf("\b\b->|");
                cout << "\b\b->|" << flush;
                cd = t + imgnum/50;
            }
        // }
        // cout << "Dealing with "<< t+1 << "th image. "<<endl;
        stringstream ss;
        string c;
        ss << t+1;
        ss >> c;
        cv::Mat img = cv::imread(data_path+"/"+c+".jpg");
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
            // cout << k << endl;
            fmat face;
            if (ni == 0){
                face.load("../../common/avr_face68_f.dat");
                // face *= 2.36 * cyk.random(0.92, 1.08) * train_scale; // scale random
                // float theta = cyk.random(-13,13)/180*3.1415926; // rotation random
                face *= 2.36 * train_scale; // scale random
                float theta = 0; // rotation random
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
            }
            else{
                face = anno.row(t) - face_current_stage.row(t*nn+k);
            }

            // cout << "hah "<< endl;

            arma::fmat idx_pool_t = idx_pool_.slice(ni);
            for (int i = 0; i < idx_pool_t.n_rows; ++i)
            {
                float x_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1));
                float y_i = idx_pool_t(i, 2) * face(0, idx_pool_t(i, 0)+annonum) + (1 - idx_pool_t(i, 2)) * face(0, idx_pool_t(i, 1)+annonum);
                if (0>x_i || 0>y_i || y_i>=gray.rows || x_i>=gray.cols)
                    indexed_features(t*nn+k, i) = 128;
                else
                    indexed_features(t*nn+k, i) = gray.ptr(y_i)[(int)x_i];
            }
            if (ni == 0)
                face_current_stage.row(t*nn+k) = anno.row(t) - face; // 直接保存整个脸坐标
        }
    }
    cout << endl;
    gen_final_features(ni);
    if (if_save)
    {
        cout << "save data ..." << endl;
        if (ni == 0)
            face_current_stage.save(save_path+".label");
        final_features.save(save_path+".data");
    }
    cout << "prepare data ok." << endl;
}

void cykSIF::test_dataset(int ni, bool if_train, bool if_save){
    cykFerns cykf;
    stringstream ss;
    string c, save_path;
    for (int i = 0; i < ni; ++i)
    {
        cout << "========================== stage "<<i+1<<" ===========================" <<endl;
        prepare_data(i, if_train, false);
        face_last_stage = face_current_stage;
        // if (i == 0)
        // {
            cout << "Last stage ";
            show_err(face_last_stage, false);
        // }
        // else
            cout << "Stage " << i+1 << " ";
        ss.clear();
        ss << i+1;
        ss >> c;
        face_current_stage = face_last_stage - cykf.fernsRegApply(final_features, "model/stage"+c);
        show_err(face_current_stage, true);
        if (if_save)
        {
            cout << "saving data ..." << endl;
            if (if_train)
                save_path = "train_stage_"+c+"_f";
            else
                save_path = "test_stage_"+c+"_f";
            // final_features.save(save_path+".data");
            face_current_stage.save(save_path+".label");
        }
        // show_face();
    }
}

fmat cykSIF::test_one(cv::Mat& gray, fmat& face, bool if_show, int ni){

    float start_t = clock();
    face_last_stage = face;
    fmat pre;
    // float eye_angle;
    float x_i;
    float y_i;
    // for (int si = 0; si < 3; ++si)
    for (int si = 0; si < 4; ++si)
    {
        // arma::fmat idx_pool_.slice(si) = idx_pool_.slice(si);
        // fmat idx_pool_t = idx_pool_.slice(si);
        // cout << 1 << endl;
        for (int i = 0; i < idx_pool_.slice(si).n_rows; ++i) // gen idx-features
        {
            // cout << i << endl;
            x_i = idx_pool_.slice(si)(i, 2) * face_last_stage(0, idx_pool_.slice(si)(i, 0)) + (1 - idx_pool_.slice(si)(i, 2)) * face_last_stage(0, idx_pool_.slice(si)(i, 1));
            y_i = idx_pool_.slice(si)(i, 2) * face_last_stage(0, idx_pool_.slice(si)(i, 0)+annonum) + (1 - idx_pool_.slice(si)(i, 2)) * face_last_stage(0, idx_pool_.slice(si)(i, 1)+annonum);
            if (0>x_i || 0>y_i || y_i>=gray.rows || x_i>=gray.cols)
                indexed_features(0, i) = 128;
            else
                indexed_features(0, i) = gray.ptr(y_i)[(int)x_i];
        }
        // cout << "cyk" << endl;
        for (int i_anno = 0; i_anno < 136; ++i_anno) // cascade all features
        {
            for (int i = 0; i < son_pool_size_; ++i)
            {
                final_features(0, son_pool_size_*i_anno+i) = 
                    indexed_features(0, son_pool_(i, i_anno+i_anno, si)) - 
                    indexed_features(0, son_pool_(i, i_anno+i_anno+1, si));
            }
        }

        pre = fern_func(si, final_features);

        face_last_stage = face_last_stage + pre;
    }


    float end_t = clock();

    if (if_show)
    {
        cv::Mat pano = gray.clone();
        cv::cvtColor(pano, pano, CV_GRAY2RGB);
        draw_face(pano, face_last_stage, cv::Scalar(100,0,255));
        cv::imshow("pano", pano);
        cv::waitKey();
    cout << "reg. cost: "<< (float)(end_t - start_t) /  CLOCKS_PER_SEC * 1000 << "ms"<<endl;
    }
    return face_last_stage;
}

fmat cykSIF::fern_func(int ni, fmat& data){
    fmat pre = zeros<fmat>(1,136);
    if (ni == 0)
    {
        int M_ = fids_0.n_rows;
        int S_ = fids_0.n_cols;
        // int N_ = data.n_rows;
        arma::umat inds = arma::zeros<arma::umat>(1, M_);
        for (int m = 0; m < M_; ++m)
        {
            for (int s = 0; s < S_; ++s)
            {
                inds(0, m) = inds(0, m) << 1; // * 2
                if (data(0, fids_0(m,s)) < thrs_0(m,s))
                {
                    inds(0, m) ++;
                }
            }
        }
        for (int m = 0; m < M_; ++m)
        {
            pre += ysFern_0.slice(m).row(inds(0,m));
        }
        return pre;
    }
    if (ni == 1)
    {
        int M_ = fids_1.n_rows;
        int S_ = fids_1.n_cols;
        // int N_ = data.n_rows;
        arma::umat inds = arma::zeros<arma::umat>(1, M_);
        for (int m = 0; m < M_; ++m)
        {
            for (int s = 0; s < S_; ++s)
            {
                inds(0, m) = inds(0, m) << 1; // * 2
                if (data(0, fids_1(m,s)) < thrs_1(m,s))
                {
                    inds(0, m) ++;
                }
            }
        }
        for (int m = 0; m < M_; ++m)
        {
            pre += ysFern_1.slice(m).row(inds(0,m));
        }
        return pre;
    }
    if (ni == 2)
    {
        int M_ = fids_2.n_rows;
        int S_ = fids_2.n_cols;
        // int N_ = data.n_rows;
        arma::umat inds = arma::zeros<arma::umat>(1, M_);
        for (int m = 0; m < M_; ++m)
        {
            for (int s = 0; s < S_; ++s)
            {
                inds(0, m) = inds(0, m) << 1; // * 2
                if (data(0, fids_2(m,s)) < thrs_2(m,s))
                {
                    inds(0, m) ++;
                }
            }
        }
        for (int m = 0; m < M_; ++m)
        {
            pre += ysFern_2.slice(m).row(inds(0,m));
        }
        return pre;
    }
    if (ni == 3)
    {
        int M_ = fids_3.n_rows;
        int S_ = fids_3.n_cols;
        // int N_ = data.n_rows;
        arma::umat inds = arma::zeros<arma::umat>(1, M_);
        for (int m = 0; m < M_; ++m)
        {
            for (int s = 0; s < S_; ++s)
            {
                inds(0, m) = inds(0, m) << 1; // * 2
                if (data(0, fids_3(m,s)) < thrs_3(m,s))
                {
                    inds(0, m) ++;
                }
            }
        }
        for (int m = 0; m < M_; ++m)
        {
            pre += ysFern_3.slice(m).row(inds(0,m));
        }
        return pre;
    }
    return pre;
}

void cykSIF::load4Model(){
    fids_0.load("model/stage1_fids.dat");
    thrs_0.load("model/stage1_thrs.dat");
    ysFern_0.load("model/stage1_ysFern.dat");

    fids_1.load("model/stage2_fids.dat");
    thrs_1.load("model/stage2_thrs.dat");
    ysFern_1.load("model/stage2_ysFern.dat");

    fids_2.load("model/stage3_fids.dat");
    thrs_2.load("model/stage3_thrs.dat");
    ysFern_2.load("model/stage3_ysFern.dat");

    fids_3.load("model/stage4_fids.dat");
    thrs_3.load("model/stage4_thrs.dat");
    ysFern_3.load("model/stage4_ysFern.dat");

    face_current_stage = zeros<fmat>(1, annonum*2);
    face_last_stage = zeros<fmat>(1, annonum*2);
    indexed_features = zeros<fmat>(1, annonum * idx_pool_size_);  // pre malloc idx-f for all traindata
    final_features = ones<fmat>(1, son_pool_size_*annonum*2 + 1);

}

void cykSIF::test_dataset_onebyone(int imgn, bool if_show, int ni){
    cyktools cyk;

    imgnum = 554; // all 689, common 554, train 3148
    nn = 1;
    // fmat init_face;
    // init_face.load("face_init.dat");

    // load_model();
    fmat total_err = zeros<fmat>((imgnum-imgn)*nn, annonum*2);
    float failure_count = 0;
    fmat diff_one_landmark_distribution = zeros<fmat>(1, 100);
    fmat diff_one_face_distribution = zeros<fmat>(1, 100);

    data_path = "/Users/ublack/Documents/ublack/Face/Datasets/300-W/";

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
            // cout << 1 << endl;
            face_current_stage = test_one(gray, face, if_show, ni);
            // cout << 2 << endl;
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
    cout << "testing ..." << endl;
    cyktools cyk;

    imgnum = 554; // all 689, common 554, train 3148
    nn = 1;
    // fmat init_face;
    // init_face.load("face_init.dat");

    // load_model();
    fmat total_err = zeros<fmat>((imgnum-imgn)*nn, annonum*2);
    float failure_count = 0;
    fmat diff_one_landmark_distribution = zeros<fmat>(1, 100);
    fmat diff_one_face_distribution = zeros<fmat>(1, 100);

    data_path = "/Users/ublack/Documents/ublack/Face/Datasets/300-W/";

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

                tmp_face += test_one(gray, face, if_show, 4);
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


