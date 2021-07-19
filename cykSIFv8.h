#ifndef CYK_SIF_
#define CYK_SIF_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <armadillo>
#include "cykFerns.h"

// #define CYK_ON_WINDOWS

class cykSIF{
public:

    cykSIF(); 
    ~cykSIF();

    //----- setter:
    void set_annonum(int na);
    void set_feature_size(int fs);
    void set_n_stages(int ns);
    void set_nn(int n);
    void set_face_scale(float fs);

    //----- training part functions:

    // generate pool for later feature extraction
    void gen_pool(int ns, float distance_from_landmark); 
    // generate features and labels for training 
    void prepare_data(int ns, bool is_train_set, bool if_save);
    // train random ferns using .data && .label ----- ferns
    // void train_one_stage_ferns(int ns, cyk_fern_prms fpa);
    // train random ferns using .data && .label ----- LSR
    float train_one_stage_LSR(int ns, float lamda);
    // tool for show err
    float show_err(arma::fmat& total_err, bool if_show);
    // tool for draw face
    void draw_face(cv::Mat& pano, arma::fmat face, cv::Scalar sc);
    // test dataset and generate prediction results ----- ferns
    // void test_one_stage_ferns(int ns, bool is_train_set, bool if_save);
    // test dataset and generate prediction results ----- LSR
    void test_one_stage_LSR(int ns, bool is_train_set, bool if_save);
    // rotate face err to normal direction
    arma::fmat rotate_label(arma::fmat face_label, arma::fmat face);
    // rotate face err back to normal direction
    arma::fmat rotate_label_back(arma::fmat face_label, arma::fmat face);
    // rotate face err for the whole data set
    void forward_rotation(arma::fmat& face, arma::fmat& anno, arma::fmat& tl);
    void back_rotation(arma::fmat& face, arma::fmat& err, arma::fmat& face_new);

    //----- core part functions:
    void merge_stage_model(int ns);
    void load_model_lsr();
    void reg_one_face(cv::Mat& gray, arma::fmat& face, bool if_show);
    void test_dataset(int imgn, bool if_show);

    // auto training 
    void train_one_stage_LSR_auto_lamda(int ns);
    void auto_train();

    void pool_selection(std::string path_model_origin, std::string path_model_new, int size_feature_new);

private:

    // training part variables:
    std::string path_dataset; // path of dataset
    std::string path_avr_face;
    int nn; // number of samples yeilded from one face picture
    int imgnum; // number of images in dataset
    float face_scale; // scale of face when training: pixel distance between two eyes
    // arma::fmat sc_in_each_stage; // feature points range when extracting features, respect to train_scale

    // core part:
    int annonum; // number of landmarks on face
    int n_stages; // number of regression

    arma::fcube pool; // index pool to generate features
    float feature_size; // feature size for one landmark

    arma::fmat indexed_features; // shape-indexed features
    arma::fmat final_features;   // cascade all selected features

    arma::fmat face_last_stage;      // full-face coordinates && without resize
    arma::fmat face_current_stage;      // full-face coordinates && without resize

    // model:
    arma::fcube model_lsr;

    // only for regression speed:
    arma::fmat err_pre;

};


#endif