#include "cykSIFv8.h"
#include "cykTools.h"
#include <cmath>
#include "cykFerns.h"
#include <time.h>

using namespace std;
using namespace arma;

cykSIF::cykSIF(){
	#ifdef _WIN32
    path_dataset = "E:/datasets/300-W/";
    path_avr_face = "E:/VS_WS/common/avr_face68_f.dat";
    #else
    path_avr_face = "/Users/ublack/Documents/c_ws/common/avr_face68_f.dat";
    path_dataset = "/Users/ublack/Documents/ublack/Face/Datasets/300-W/";
    #endif
}

cykSIF::~cykSIF() {
}

//----- setters 
void cykSIF::set_annonum(int na){
    annonum = na;
}
void cykSIF::set_feature_size(int fs){
    feature_size = fs;
}
void cykSIF::set_n_stages(int ns){
    n_stages = ns;
}
void cykSIF::set_nn(int n){
    nn = n;
}
void cykSIF::set_face_scale(float fs){
    face_scale = fs;
}

void cykSIF::gen_pool(int ns, float distance_from_landmark){
    // make sure ns is within the max number of stages
    if (ns >= n_stages)
    {
        cout << "ns too large !" << endl;
        exit(1);
    }

    // pre melloc for pool if pool.dat not exists
    if (!pool.load("model/pool.dat"))
    {
        // one feature needs two points, every point needs idx1 idx2 & ratio; total 6
        pool = zeros<fcube>(annonum, 6*feature_size, n_stages);
    }

    // if pool doesnt have enough memory, expends it
    if (pool.n_slices < ns + 1)
    {
        fcube tmp_pool = pool;
        pool = zeros<fcube>(annonum, 6*feature_size, n_stages);
        for (int i = 0; i < tmp_pool.n_slices; ++i)
        {
            pool.slice(i) = tmp_pool.slice(i);
        }
    }
    
    // generate random pool according to distance_from_landmark
    cykTools cyk;

    fmat avr_face;

    avr_face.load(path_avr_face); // load standard face

    float ds = sqrt(distance_from_landmark) * ((avr_face(0,39)-avr_face(0,45))*(avr_face(0,39)-avr_face(0,45)) + 
            (avr_face(0,39+annonum)-avr_face(0,45+annonum))*(avr_face(0,39+annonum)-avr_face(0,45+annonum)));

    for (int i = 0; i < annonum; ++i)
    {
        float xl, yl; // landmark point
        xl = avr_face(0, i);
        yl = avr_face(0, i+annonum);
        for (int j = 0; j < feature_size; ++j)
        {
            float idx11, idx12, r1, x1, y1; // first point
            float idx21, idx22, r2, x2, y2; // second point

            do{
                // generate random points' index and ratio
                idx11 = floor(cyk.random(0, annonum));
                idx12 = floor(cyk.random(0, annonum));
                r1 = cyk.random(-0.5, 1.5);
                x1 = r1 * avr_face(0, idx11) + (1 - r1) * avr_face(0, idx12);
                y1 = r1 * avr_face(0, idx11 + annonum) + (1 - r1) * avr_face(0, idx12 + annonum);
            }while(((x1-xl)*(x1-xl)+(y1-yl)*(y1-yl))>ds);

            do{
                idx21 = floor(cyk.random(0,annonum));
                idx22 = floor(cyk.random(0,annonum));
                r2 = cyk.random(-0.5, 1.5);
                x2 = r2 * avr_face(0, idx21) + (1 - r2) * avr_face(0, idx22);
                y2 = r2 * avr_face(0, idx21 + annonum) + (1 - r2) * avr_face(0, idx22 + annonum);
            }while(((x2-xl)*(x2-xl)+(y2-yl)*(y2-yl))>ds);

            pool(i, 6 * j + 0, ns) = idx11;
            pool(i, 6 * j + 1, ns) = idx12;
            pool(i, 6 * j + 2, ns) = r1;
            pool(i, 6 * j + 3, ns) = idx21;
            pool(i, 6 * j + 4, ns) = idx22;
            pool(i, 6 * j + 5, ns) = r2;
        }
    }

    pool.save("model/pool.dat");
    cout << "pool.dat save OK! " << endl;
}

fmat cykSIF::rotate_label(fmat face_label, fmat face){
    fmat face_label_rotated = face_label;
    for (int n = 0; n < face_label.n_rows; ++n)
    {
        float eye_angle = atan2(face(n ,45+annonum)-face(n ,36+annonum),
            face(n ,45)-face(n ,36));//弧度 逆时针方向
        float cos_angle = cos(eye_angle);
        float sin_angle = sin(eye_angle);
        float tmp1, tmp2;
        for (int i = 0; i < annonum; ++i)
        {
            tmp1 = face_label(n, i);
            tmp2 = face_label(n, i+annonum);
            face_label_rotated(n, i) = tmp1*cos_angle + tmp2*sin_angle;
            face_label_rotated(n, i+annonum) = tmp2*cos_angle - tmp1*sin_angle;
        }
    }
    return face_label_rotated;
}

fmat cykSIF::rotate_label_back(fmat face_label, fmat face){
    fmat face_label_rotated = face_label;
    for (int n = 0; n < face_label.n_rows; ++n)
    {
        float eye_angle = atan2(face(n ,45+annonum)-face(n ,36+annonum),
            face(n ,45)-face(n ,36));//弧度 逆时针方向
        float cos_angle = cos(-eye_angle);
        float sin_angle = sin(-eye_angle);
        float tmp1, tmp2;
        for (int i = 0; i < annonum; ++i)
        {
            tmp1 = face_label(n, i);
            tmp2 = face_label(n, i+annonum);
            face_label_rotated(n, i) = tmp1*cos_angle + tmp2*sin_angle;
            face_label_rotated(n, i+annonum) = tmp2*cos_angle - tmp1*sin_angle;
        }
    }
    return face_label_rotated;
}

void cykSIF::prepare_data(int ns, bool is_train_set, bool if_save){

    // gen_pool shall not be here
    // gen_pool(ns, distance_from_landmark);
    pool.load("model/pool.dat");
    feature_size = pool.n_cols / 6;

    cykTools cyk;
    stringstream ss;
    string save_path;
    // string last_path;
    ss << ns;
    ss >> save_path;
    // ss.clear();
    // ss << ns - 1;
    // ss >> last_path;

    string data_path = path_dataset;

    if (!is_train_set)
    {
        imgnum = 554;
        data_path += "test";
        save_path = "testset_stage_"+save_path;
    }
    else
    {
        imgnum = 3148; // 3148 / 2 = 1574
        data_path += "train";
        save_path = "trainset_stage_"+save_path;
    }

    if (ns != 0){
        if (if_save)
        {
            cout << "Loading : " << save_path+".face" <<endl;
            face_current_stage.load(save_path+".face");
        }
        // else
        //     face_current_stage = face_last_stage;
    }
    else
    {
        face_current_stage = zeros<fmat>(imgnum*nn, annonum*2);
    }

    indexed_features = ones<fmat>(imgnum*nn, feature_size * annonum + 1); // let the last col of feature to be 1
    final_features = ones<fmat>(imgnum*nn, feature_size * annonum + 1); // let the last col of feature to be 1

    arma::fmat anno;
    anno.load(data_path+"/annof.dat");
    int cd;
    // if (ns == 0)
    // {
        cout << "Prepare data ..." << endl;
        cout << "|---------------------------------------------------|"<< endl;
        cout << "|>|" << flush;
        cd = imgnum / 50;
    // }
    for (int t=0; t<imgnum; t++)
    {
        // if (ns == 0)
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

        // read image
        cv::Mat img = cv::imread(data_path+"/"+c+".jpg");
        if (img.empty()) {
            cout << "Image loading error!"<<endl;
            exit(1);
        }

        // calculate the face scale: respect to face_scale 
        float sc = sqrt((anno(t,39)-anno(t,45))*(anno(t,39)-anno(t,45)) + 
            (anno(t,39+annonum)-anno(t,45+annonum))*(anno(t,39+annonum)-anno(t,45+annonum)))/face_scale;
        // float lfr = face_scale / 15 * sc; // local feature range

        // convert to gray scale
        cv::Mat gray;
        if (img.channels() == 3)
        {
            cv::cvtColor(img, gray, CV_RGB2GRAY);
        }else{
            gray = img.clone();
        }
        // cv::Mat img_show;// = gray.clone();

        // iterate all dataset
        for (int k=0; k<nn; k++) {
            // cv::cvtColor(gray, img_show, CV_GRAY2BGR);
            fmat face; // face location, not error
            if (ns == 0){
                // randomly projecting normal face
                face.load(path_avr_face);
                face *= 2.36 * cyk.random(0.92, 1.08) * face_scale * sc; // scale random
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
                // dx += cyk.random(-lfr, lfr);
                // dy += cyk.random(-lfr, lfr);
                face.cols(0,annonum-1) += dx(0,0);
                face.cols(annonum, annonum*2-1) += dy(0,0);   
            }
            else{
                face = face_current_stage.row(t*nn+k);
            }

            // extracting features: 
            for (int i = 0; i < annonum; ++i)
            {
                for (int j = 0; j < feature_size; ++j)
                {
                    float idx11, idx12, r1, x1, y1; // first point
                    float idx21, idx22, r2, x2, y2; // second point

                    idx11 = pool(i, 6 * j + 0, ns);
                    idx12 = pool(i, 6 * j + 1, ns);
                    r1 =    pool(i, 6 * j + 2, ns);
                    idx21 = pool(i, 6 * j + 3, ns);
                    idx22 = pool(i, 6 * j + 4, ns);
                    r2 =    pool(i, 6 * j + 5, ns);

                    // locate point on image
                    x1 = r1 * face(0, idx11) + (1 - r1) * face(0, idx12);
                    y1 = r1 * face(0, idx11 + annonum) + (1 - r1) * face(0, idx12 + annonum);
                    x2 = r2 * face(0, idx21) + (1 - r2) * face(0, idx22);
                    y2 = r2 * face(0, idx21 + annonum) + (1 - r2) * face(0, idx22 + annonum);

                    float tmp_f;
                    // as long as one point is out of image size, feature should be set to zero
                    if (0>x1 || 0>y1 || y1>=gray.rows || x1>=gray.cols || 0>x2 || 0>y2 || y2>=gray.rows || x2>=gray.cols)
                        tmp_f = 0;
                    else{
                        tmp_f = gray.ptr(y1)[(int)x1] - gray.ptr(y2)[(int)x2];
                        // img_show.ptr(y1)[3*(int)x1+0] = 255;
                        // img_show.ptr(y1)[3*(int)x1+1] = 0;
                        // img_show.ptr(y1)[3*(int)x1+2] = 0;
                        // img_show.ptr(y2)[3*(int)x2+0] = 255;
                        // img_show.ptr(y2)[3*(int)x2+1] = 0;
                        // img_show.ptr(y2)[3*(int)x2+2] = 0;
                    }

                    indexed_features(t*nn+k, i*feature_size+j) = tmp_f;
                }
            }
            // cv::imshow("show", img_show);
            // cv::waitKey();
            if (ns == 0)
            {
                face_current_stage.row(t*nn+k) = face;
            }
        }
    }
    cout << endl;
    // generate final_features: now just using sign() function
    final_features = sign(indexed_features);
    // final_features = indexed_features;
    if (if_save)
    {
        cout << "save data ..." << endl;
        if (ns == 0) // if ns!=0, then *.label shall be generate from regression stage, not here
            face_current_stage.save(save_path+".face");
        final_features.save(save_path+".feature");
    }
    cout << "prepare data ok." << endl;
}

/*
void cykSIF::train_one_stage_ferns(int ns, cyk_fern_prms fpa){

    cykFerns cykf;

    fmat td, tl;
    stringstream ss;
    string c1;
    ss << ns;
    ss >> c1;
    td.load("trainset_stage_"+c1+".data");
    tl.load("trainset_stage_"+c1+".label");
    ofstream flog("cykFerns_train_log.txt", ios::app);
    flog << "|====================================stage "<<c1<<"============================================" <<endl;
    flog << "fern_param.eta : " << fpa.eta << endl; // = 0.12;      // learning rate in [0,1] I just feel it is as same functionality as reg
    flog << "fern_param.thrr : " << fpa.thrr ;      // "-1,1;";     // range for randomly generated thresholds || NOTE!!! this range should be within minmax(xs0)!! ?????? but what if xs0 has multi-dimentions?
    flog << "fern_param.S : " << fpa.S << endl;     // 7;           // %  S - fern depth
    flog << "fern_param.M : " << fpa.M << endl;     // 500;         // %  M - number ferns
    flog << "fern_param.R : " << fpa.R << endl;     // 6;           // %  R - number repeats, set larger if always get "best not found!"
    flog << "fern_param.if_show : " << fpa.if_show << endl; // false;
    fmat pre = cykf.fernsRegTrain(td, tl, fpa, "model/stage"+c1);
    fmat err = tl - pre;
    flog << "train tl : "<< show_err(tl, false) << endl;
    flog << "train err : "<< show_err(err, false) << endl;
    // err.save("train_stage_"+c1+"_f.label");
    cout << "train end." << endl;
    td.load("testset_stage_"+c1+".data");
    tl.load("testset_stage_"+c1+".label");
    cykf.loadFerns("model/stage" + c1);
    pre = cykf.fernsRegApply(td);
    err = tl - pre;
    flog << "test tl : " << show_err(tl, false) << endl;
    flog << "test err : " << show_err(err, false) << endl;
    // err.save("test_stage_"+c1+"_f.label");
    cout << "test end." << endl;
    flog.close();
}*/

void cykSIF::forward_rotation(fmat& face, fmat& anno, fmat& tl){
    tl = face;
    int n = face.n_rows / anno.n_rows;
    for (int i = 0; i < anno.n_rows; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            fmat face_tmp = face.row(i*n+j);
            float sc = sqrt((face_tmp(0,39)-face_tmp(0,45))*(face_tmp(0,39)-face_tmp(0,45)) + 
                (face_tmp(0,39+annonum)-face_tmp(0,45+annonum))*(face_tmp(0,39+annonum)-face_tmp(0,45+annonum)))/face_scale;
            fmat err_raw = anno.row(i) - face_tmp;
            tl.row(i*n+j) = rotate_label(err_raw, face_tmp) / sc;
        }
    }
}

void cykSIF::back_rotation(fmat& face, fmat& err, fmat& face_new){
    face_new = rotate_label_back(err, face);
    for (int i = 0; i < face_new.n_rows; ++i)
    {
        float sc = sqrt((face(i,39)-face(i,45))*(face(i,39)-face(i,45)) + 
            (face(i,39+annonum)-face(i,45+annonum))*(face(i,39+annonum)-face(i,45+annonum)))/face_scale;
        face_new.row(i) *= sc;
        face_new.row(i) += face.row(i);
    }
}

float cykSIF::train_one_stage_LSR(int ns, float lamda){
    cykTools cyk;

    fmat td, tl, face;
    stringstream ss;
    string c1;
    ss << ns;
    ss >> c1;
    td.load("trainset_stage_"+c1+".feature");
    face.load("trainset_stage_"+c1+".face");
    arma::fmat anno_train, anno_test;
    anno_train.load(path_dataset+"/train/annof.dat");
    anno_test.load(path_dataset+"/test/annof.dat");
    // cout << "td size: " << td.n_rows << ", " << td.n_cols << endl;
    // cout << "tl size: " << tl.n_rows << ", " << tl.n_cols << endl;
    ofstream flog("cykFace_train_log.txt", ios::app);
    flog << "|================ LSR ===============stage "<<c1<<"============================================" <<endl;
    flog << "lamda : " << lamda << endl; // = 0.12;      // learning rate in [0,1] I just feel it is as same functionality as reg
    forward_rotation(face, anno_train, tl);
    mat A_d = cyk.LSR(conv_to<mat>::from(td), conv_to<mat>::from(tl), lamda);
    fmat A = conv_to<fmat>::from(A_d);
    A.save("model/stage"+c1+"_A.dat");
    fmat pre = td * A;
    fmat face_new;
    back_rotation(face, pre, face_new);
    flog << "train tl : "<< show_err(face, false) << endl;
    flog << "train err : "<< show_err(face_new, false) << endl;
    cout << "train end." << endl;

    td.load("testset_stage_"+c1+".feature");
    face.load("testset_stage_"+c1+".face");
    forward_rotation(face, anno_test, tl);
    pre = td * A;
    back_rotation(face, pre, face_new);
    flog << "test tl : " << show_err(face, false) << endl;
    float err_test = show_err(face_new, false);
    flog << "test err : " << err_test << endl;
    cout << "test end." << endl;
    flog.close();    
    return err_test;
}

float cykSIF::show_err(fmat& face, bool if_show){
    string data_path;
    imgnum = face.n_rows/nn;
    if (imgnum == 3148)
        data_path = path_dataset + "train/";
    else
        data_path = path_dataset + "test/";
    arma::fmat anno, error_face;
    anno.load(data_path+"annof.dat");
    float dis1 = 0;
    cv::Mat img;
    for (int in = 0; in < imgnum; ++in)
    {
        if (if_show)
        {
            stringstream ss;
            string c;
            ss << in+1;
            ss >> c;
            img = cv::imread(data_path+c+".jpg");
            if (img.empty()) {
                cout << "Image loading error!"<<endl;
                exit(1);
            }
        }

        float sc = sqrt((anno(in,39)-anno(in,45))*(anno(in,39)-anno(in,45)) + 
            (anno(in,39+annonum)-anno(in,45+annonum))*(anno(in,39+annonum)-anno(in,45+annonum)))/face_scale;

        for (int n = 0; n < nn; ++n)
        {
            if (!if_show)
            {
                error_face = anno.row(in) - face.row(in*nn+n);
                for (int i = 0; i < annonum; ++i)
                {
                    dis1 += sqrt(error_face(0,i)*error_face(0,i)+error_face(0,i+annonum)*error_face(0,i+annonum)) / sc;
                }
            }
            else
            {
                cv::resize(img, img, cv::Size(img.cols/sc, img.rows/sc));
                fmat face_draw = face.row(in*nn+n) / sc;
                cv::Mat pano = img.clone();
                draw_face(pano, face_draw, cv::Scalar(255,0,0));
                cv::imshow("pano", pano);
                if (cv::waitKey() == 27)
                    return -1;
            }
        }
    }
    dis1 /= face.n_rows * annonum;
    return dis1*100/face_scale;
}

void cykSIF::draw_face(cv::Mat& pano, fmat face, cv::Scalar sc){
    for (int ccc=0; ccc<annonum; ccc++)
    {
        cv::circle(pano, cv::Point(face(0, ccc), face(0, ccc+annonum)), 2, sc);
    }
}

/*
void cykSIF::test_one_stage_ferns(int ns, bool is_train_set, bool if_save){
    cykFerns cykf;
    stringstream ss;
    string c, save_path;


    for (int i = 0; i < ns+1; ++i)
    {
        cout << "========================== stage "<<i<<" ===========================" <<endl;
        prepare_data(i, is_train_set, false);
        face_last_stage = face_current_stage;
        // if (i == 0)
        // {
            cout << "Last stage ";
            show_err(face_last_stage, false);
        // }
        // else
            cout << "Stage " << i << " ";
        ss.clear();
        ss << i;
        ss >> c;
        cykf.loadFerns("model/stage" + c);
        face_current_stage = face_last_stage - cykf.fernsRegApply(final_features);

        // face_current_stage = rotate_label_back(face_current_stage, )

        if (if_save)
        {
            show_err(face_current_stage, false);
            cout << "saving data ..." << endl;
            ss.clear();
            ss << i+1;
            ss >> c;
            if (is_train_set)
                save_path = "trainset_stage_"+c;
            else
                save_path = "testset_stage_"+c;
            // final_features.save(save_path+".data");
            face_current_stage.save(save_path+".label");
        }
        else
            show_err(face_current_stage, true);
    }
}*/

void cykSIF::test_one_stage_LSR(int ns, bool is_train_set, bool if_save){
    fmat A;
    stringstream ss;
    string c, save_path;

    arma::fmat anno;
    if (is_train_set)
        anno.load(path_dataset+"/train/annof.dat");
    else
        anno.load(path_dataset+"/test/annof.dat");

    for (int i = 0; i < ns+1; ++i)
    {
        cout << "========================== stage "<<i<<" ===========================" <<endl;
        prepare_data(i, is_train_set, false);
        face_last_stage = face_current_stage;
        cout << "Last stage: " << show_err(face_last_stage, false) << endl;
    
        ss.clear();
        ss << i;
        ss >> c;
        A.load("model/stage"+c+"_A.dat");
        fmat err_pre = final_features * A;

        back_rotation(face_last_stage, err_pre, face_current_stage);

        if (if_save)
        {
            cout << "Stage " << i << ": " << show_err(face_current_stage, false) << endl;
            cout << "saving data ..." << endl;
            ss.clear();
            ss << i+1;
            ss >> c;
            if (is_train_set)
                save_path = "trainset_stage_"+c;
            else
                save_path = "testset_stage_"+c;
            // final_features.save(save_path+".data");
            face_current_stage.save(save_path+".face");
        }
    }
    if (!if_save)
    {
        show_err(face_current_stage, true);
    }
}

void cykSIF::merge_stage_model(int ns){
    fmat A_tmp; 
    A_tmp.load("model/stage0_A.dat");
    set_feature_size((A_tmp.n_rows-1)/annonum);
    model_lsr = zeros<fcube>(feature_size * annonum + 1, annonum*2, ns);
    for (int i = 0; i < ns; ++i)
    {
        stringstream ss;
        string c;
        ss << i;
        ss >> c;
        A_tmp.load("model/stage"+c+"_A.dat");
        model_lsr.slice(i) = A_tmp;
    }
    model_lsr.save("model/model_lsr.dat");
    cout << "merge_stage_model OK!" << endl;
}


// core part: for real time regression
void cykSIF::load_model_lsr(){
    model_lsr.load("model/model_lsr.dat");
    pool.load("model/pool.dat");
    cout << "Model load done, total stages: " << model_lsr.n_slices << endl;
    indexed_features = zeros<fmat>(1, feature_size * annonum + 1);
    final_features = zeros<fmat>(1, feature_size * annonum + 1);
    err_pre = zeros<fmat>(1, annonum*2);
}

void cykSIF::reg_one_face(cv::Mat& gray, arma::fmat& face, bool if_show){
    float idx11, idx12, r1, x1, y1; // first point
    float idx21, idx22, r2, x2, y2; // second point
    float sc;
    float eye_angle, cos_angle, sin_angle;
    fmat tmp1, tmp2;
    float start_t, end_t;

    if (if_show)
        start_t = clock();

    for (int ns = 0; ns < n_stages; ++ns)
    {
        for (int i = 0; i < annonum; ++i)
        {
            for (int j = 0; j < feature_size; ++j)
            {
                idx11 = pool(i, 6 * j + 0, ns);
                idx12 = pool(i, 6 * j + 1, ns);
                r1 =    pool(i, 6 * j + 2, ns);
                idx21 = pool(i, 6 * j + 3, ns);
                idx22 = pool(i, 6 * j + 4, ns);
                r2 =    pool(i, 6 * j + 5, ns);

                // locate point on image
                x1 = r1 * face(0, idx11) + (1 - r1) * face(0, idx12);
                y1 = r1 * face(0, idx11 + annonum) + (1 - r1) * face(0, idx12 + annonum);
                x2 = r2 * face(0, idx21) + (1 - r2) * face(0, idx22);
                y2 = r2 * face(0, idx21 + annonum) + (1 - r2) * face(0, idx22 + annonum);

                // as long as one point is out of image size, feature should be set to zero
                if (0>x1 || 0>y1 || y1>=gray.rows || x1>=gray.cols || 0>x2 || 0>y2 || y2>=gray.rows || x2>=gray.cols)
                    indexed_features(0, i*feature_size+j) = 0;
                else
                    indexed_features(0, i*feature_size+j) = gray.ptr(y1)[(int)x1] - gray.ptr(y2)[(int)x2];
            }
        }
        final_features = sign(indexed_features);
        err_pre = final_features * model_lsr.slice(ns);

        // rotation part:
        eye_angle = atan2(face(0, 45+annonum) - face(0, 36+annonum),
            face(0, 45) - face(0, 36));//弧度 逆时针方向
        cos_angle = cos(-eye_angle);
        sin_angle = sin(-eye_angle);
        tmp1 = err_pre.cols(0, annonum-1);
        tmp2 = err_pre.cols(annonum, err_pre.n_cols-1);
        err_pre.cols(0, annonum-1) = tmp1 * cos_angle + tmp2 * sin_angle;
        err_pre.cols(annonum, err_pre.n_cols-1) = tmp2 * cos_angle - tmp1 * sin_angle;

        // scale part: 
        sc = sqrt((face(0, 39)-face(0, 45))*(face(0, 39)-face(0, 45)) + 
            (face(0, 39+annonum)-face(0, 45+annonum))*(face(0, 39+annonum)-face(0, 45+annonum)))/face_scale;
        err_pre *= sc;
        face += err_pre;
    }
    if (if_show)
    {
        end_t = clock();
        cout << "reg. cost: "<< (float)(end_t - start_t) /  CLOCKS_PER_SEC * 1000 << "ms"<<endl;
    }
}

void cykSIF::test_dataset(int imgn, bool if_show){
    cykTools cyk;

    imgnum = 554; // all 689, common 554, train 3148
    nn = 1;

    load_model_lsr();
    fmat total_err = zeros<fmat>((imgnum-imgn)*nn, annonum*2);
    float failure_count = 0;
    fmat diff_one_landmark_distribution = zeros<fmat>(1, 100);
    fmat diff_one_face_distribution = zeros<fmat>(1, 100);

    string data_path = path_dataset;

    arma::fmat anno;
    anno.load(data_path+"test/annof.dat");// = cyk.readMat((data_path+"test/anno.mat").c_str());

    cout << "Testing ..." << endl;

    for (int t=imgn; t<imgnum; t++)
    {
        if (if_show)
        {
            cout << "Dealing with "<< t+1 << "th image. "<<endl;
        }
        stringstream ss;
        string c;
        ss << t+1;
        ss >> c;
        cv::Mat img = cv::imread(data_path+"test/"+c+".jpg");
        if (img.empty()) {
            cout << "Image loading error!"<<endl;
            exit(1);
        }

        // calculate the face scale: respect to face_scale 
        float sc = sqrt((anno(t,39)-anno(t,45))*(anno(t,39)-anno(t,45)) + 
            (anno(t,39+annonum)-anno(t,45+annonum))*(anno(t,39+annonum)-anno(t,45+annonum)))/face_scale;

        cv::Mat gray;
        if (img.channels() == 3)
            cv::cvtColor(img, gray, CV_RGB2GRAY);
        else
            gray = img.clone();

        for (int k=0; k<nn; k++) {
            fmat face;
            face.load(path_avr_face);
            face *= 2.36  * face_scale * sc; // scale random
            // float theta = cyk.random(-13,13)/180*3.1415926; // rotation random
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

            reg_one_face(gray, face, if_show);

            if (if_show)
            {
                cv::Mat pano = gray.clone();
                cv::cvtColor(pano, pano, CV_GRAY2RGB);
                cv::resize(pano, pano, cv::Size(pano.cols/sc, pano.rows/sc));
                draw_face(pano, face/sc, cv::Scalar(100,0,255));
                cv::imshow("pano", pano);
                if (cv::waitKey() == 27)
                    return;
            }

            float diff_one_face = 0;

            total_err.row((t-imgn)*nn+k) = (anno.row(t) - face) / sc;
            for (int j = 0; j < annonum; ++j)
            {
                float diff_one_landmark = sqrt(total_err((t-imgn)*nn+k,j)*total_err((t-imgn)*nn+k,j)+
                    total_err((t-imgn)*nn+k,j+annonum)*total_err((t-imgn)*nn+k,j+annonum));

                diff_one_face += diff_one_landmark;

                if (diff_one_landmark >= 99)
                    diff_one_landmark_distribution(0,99)++;
                else
                    diff_one_landmark_distribution(0, floor(diff_one_landmark))++;
                if ( diff_one_landmark > 0.1 * face_scale)
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
    cout << "total err: " << dis1*100/face_scale << "%" << endl;
    ofstream flog("cykFace_train_log.txt", ios::app);
    flog << "|                     >>>>     total err on testset: " << dis1*100/face_scale << "%" << endl;
    flog << "|==================================================================================================" <<endl;
    flog.close();
}

void cykSIF::auto_train(){

    set_annonum(68);
    set_face_scale(80);
    set_feature_size(40); // feature size for one landmark
    set_n_stages(5); 
    set_nn(20);

    ofstream flog("cykFace_autotrain_log.txt", ios::app);
    flog << "|==================================================================================================" <<endl;

    for (int ns = 0; ns < n_stages; ++ns)
    {
        flog << "| ============ ns: " << ns << " =============" << endl;
        int train_count = 0;
        float delta_dfl = 1;
        float dfl_left = 0;
        float dfl_right = 1;

        float mid_dfl_1, mid_dfl_2;
        float err_mid_dfl_1, err_mid_dfl_2;
        float lamda;

        // 三分法： http://blog.csdn.net/eastmoon502136/article/details/7706479
        while (delta_dfl >= 0.1)
        {
            // flog << "# " << train_count++ << ": --------------" << endl;
            float lamda_1, lamda_2;

            // --------------- 1 --------------------
            mid_dfl_1 = (dfl_right - dfl_left) / 3 + dfl_left;
            flog << "dfl: " << mid_dfl_1 << endl;
            gen_pool(ns, mid_dfl_1);
            sleep(2);
            prepare_data(ns, true, true);
            prepare_data(ns, false, true);
            sleep(2);

            // float delta_lamda = 1e10;
            // float lamda_left = 30000;
            // float lamda_right = 120000;
            // float mid_lamda_1, mid_lamda_2;
            // float err_mid_lamda_1, err_mid_lamda_2;
            // while (delta_lamda >= 10000)
            // {
            //     mid_lamda_1 = (lamda_right - lamda_left) / 3 + lamda_left;
            //     err_mid_lamda_1 = train_one_stage_LSR(ns, mid_lamda_1);
            //     flog << "       lamda_1: " << mid_lamda_1 << ", err: " << err_mid_lamda_1 << endl;

            //     mid_lamda_2 = lamda_right - (lamda_right - lamda_left) / 3;
            //     err_mid_lamda_2 = train_one_stage_LSR(ns, mid_lamda_2);
            //     flog << "       lamda_2: " << mid_lamda_2 << ", err: " << err_mid_lamda_2 << endl;

            //     if (err_mid_lamda_1 < err_mid_lamda_2)
            //         lamda_right = mid_lamda_2;
            //     else
            //         lamda_left = mid_lamda_1;

            //     delta_lamda = lamda_right - lamda_left;
            // }
            // err_mid_dfl_1 = err_mid_lamda_1;
            lamda_1 = 50000;
            err_mid_dfl_1 = train_one_stage_LSR(ns, lamda_1);
            flog << "   final lamda: " << lamda_1 << ", final_err: " << err_mid_dfl_1 << endl;
 
            // --------------- 2 --------------------
            mid_dfl_2 = dfl_right - (dfl_right - dfl_left) / 3;
            flog << "dfl: " << mid_dfl_2 << endl;
            gen_pool(ns, mid_dfl_2);
            sleep(2);
            prepare_data(ns, true, true);
            prepare_data(ns, false, true);
            sleep(2);

            // delta_lamda = 1e10;
            // lamda_left = 30000;
            // lamda_right = 120000;
            // while (delta_lamda >= 10000)
            // {
            //     mid_lamda_1 = (lamda_right - lamda_left) / 3 + lamda_left;
            //     err_mid_lamda_1 = train_one_stage_LSR(ns, mid_lamda_1);
            //     flog << "       lamda_1: " << mid_lamda_1 << ", err: " << err_mid_lamda_1 << endl;

            //     mid_lamda_2 = lamda_right - (lamda_right - lamda_left) / 3;
            //     err_mid_lamda_2 = train_one_stage_LSR(ns, mid_lamda_2);
            //     flog << "       lamda_2: " << mid_lamda_2 << ", err: " << err_mid_lamda_2 << endl;

            //     if (err_mid_lamda_1 < err_mid_lamda_2)
            //         lamda_right = mid_lamda_2;
            //     else
            //         lamda_left = mid_lamda_1;

            //     delta_lamda = lamda_right - lamda_left;
            // }
            // err_mid_dfl_2 = err_mid_lamda_1;
            lamda_2 = 50000;
            err_mid_dfl_2 = train_one_stage_LSR(ns, lamda_2);
            flog << "   final lamda: " << lamda_2 << ", final_err: " << err_mid_dfl_2 << endl;


            if (err_mid_dfl_1 < err_mid_dfl_2)
            {
                dfl_right = mid_dfl_2;
                lamda = lamda_1;
            }
            else
            {
                dfl_left = mid_dfl_1;
                lamda = lamda_2;
            }
            delta_dfl = dfl_right - dfl_left;
        }
        flog << "final params: " << endl;
        flog << "   dfl: " << dfl_right << endl;
        flog << "   lamda: " << lamda << endl;

        gen_pool(ns, dfl_right);
        sleep(2);
        prepare_data(ns, true, true);
        prepare_data(ns, false, true);
        sleep(2);
        train_one_stage_LSR(ns, lamda);
        // ---------------- testing part & save prediction results -----------------
        cout << "######## testing part & save prediction results ########" << endl;
        cout << "------ TRAIN SET ------" << endl;
        test_one_stage_LSR(ns, true, true); // trainset
        cout << "------ TEST SET -------" << endl;
        test_one_stage_LSR(ns, false, true); // testset
        cout << "######## ALL DONE ########" << endl;
        sleep(2);

    }

    flog.close();
}

void cykSIF::pool_selection(string path_model_origin, string path_model_new, int size_feature_new){
    // int size_feature_new = atoi(argv[1]);

    fcube pool;
    pool.load(path_model_origin + "pool.dat");
    cout << pool.n_rows << ", " << pool.n_cols << ", " << pool.n_slices << endl;

    fmat A;
    A.load(path_model_origin + "stage0_A.dat");
    cout << A.n_rows << ", " << A.n_cols << endl;

    int f_size = A.n_rows / pool.n_rows;
    fcube pool_tmp = zeros<fcube>(68, 6*size_feature_new, 5);
    fmat A_tmp = ones<fmat>(68*size_feature_new+1, 136);

    for (int ns = 0; ns < 5; ++ns)
    {
        cout << "ns: " << ns << endl;
        stringstream ss;
        string c;
        ss << ns;
        ss >> c;
        A.load(path_model_origin + "stage"+c+"_A.dat");

        for (int na = 0; na < 68; ++na)
        {
            fmat A_col = A.cols(na*2, na*2+1); // 取两列
            for (int ni = 0; ni < 68; ++ni)
            {
                fmat A_son = A_col.rows(ni*f_size, ni*f_size+f_size-1); // 取一个landmark的特征所对应的A, f_size行
                fmat A_son_col_sum =  sum(abs(A_son), 1); // 把两列相加, 每行的和表示该feature的贡献率
                fmat A_son_sorted = sort(A_son_col_sum, "descend", 0); // 对列从大到小排序
                // A_son_sorted.print("A_son_sorted");

                float min_selected = A_son_sorted(size_feature_new-1, 0); // 取要截断的中间A值大小，作为A的最小值，取所有大于等于该值的A

                int fi = 0;
                int i = 0;
                while(fi < size_feature_new)
                {
                    if (i == A_son_col_sum.n_rows)
                        cout << "Error!" << endl;
                    if (A_son_col_sum(i, 0) >= min_selected)
                    {
                        pool_tmp.slice(ns).row(ni).cols(6*fi, 6*fi+5) = pool.slice(ns).row(ni).cols(6*i, 6*i+5);
                        A_tmp(size_feature_new*ni+fi, na*2) = A_son(i, 0);
                        A_tmp(size_feature_new*ni+fi, na*2+1) = A_son(i, 1);
                        fi++;
                    }
                    i++;
                }
            }
        }
        A_tmp.row(A_tmp.n_rows-1) = A.row(A.n_rows-1);
        A_tmp.save(path_model_new + "stage"+c+"_A.dat");
    }
    pool_tmp.save(path_model_new + "pool.dat");  
    merge_stage_model(5);
}

void cykSIF::train_one_stage_LSR_auto_lamda(int ns){
    float delta_lamda = 1e10;
    float lamda_left = 30000;
    float lamda_right = 120000;
    float mid_lamda_1, mid_lamda_2;
    float err_mid_lamda_1, err_mid_lamda_2;
    while (delta_lamda >= 10000)
    {
        mid_lamda_1 = (lamda_right - lamda_left) / 3 + lamda_left;
        err_mid_lamda_1 = train_one_stage_LSR(ns, mid_lamda_1);
        cout << "       lamda_1: " << mid_lamda_1 << ", err: " << err_mid_lamda_1 << endl;
        mid_lamda_2 = lamda_right - (lamda_right - lamda_left) / 3;
        err_mid_lamda_2 = train_one_stage_LSR(ns, mid_lamda_2);
        cout << "       lamda_2: " << mid_lamda_2 << ", err: " << err_mid_lamda_2 << endl;

        if (err_mid_lamda_1 < err_mid_lamda_2)
            lamda_right = mid_lamda_2;
        else
            lamda_left = mid_lamda_1;
        delta_lamda = lamda_right - lamda_left;
    }
    train_one_stage_LSR(ns, mid_lamda_1);
}








