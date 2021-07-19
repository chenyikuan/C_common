#ifndef CYK_SIF_
#define CYK_SIF_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <armadillo>


class cykSIF{
public:
     // float a[5][5441][136];
    // float ***a;

    cykSIF(); 
    ~cykSIF();
    // cykSIF(float (*a_)[5441][136]); 
    void prepare_data(int n_stage);
    void select_son_pool(int n_stage);
    void sdh_train(int n_stage);
    void stage_train(int n_stage);
    void init(int n_stages, int idx_pool_size, int son_pool_size, arma::fmat lft, arma::fmat sies, float ts); 
    void train(int n_stages, int idx_pool_size, int son_pool_size, arma::fmat lft, arma::fmat sies, float ts);
    void test(int n_stage, bool if_training);
    void save_model();
    void gen_final_features(int n_stage);
    void show_pre(arma::fmat& in_label, arma::fmat& in_pre, bool if_training);
    void gen_dist_diff();
    void update_face(int n_stages);
    void draw_face(cv::Mat& pano, arma::fmat face, cv::Scalar sc);
    void genDesPool();   
    void load_model(bool if_final_use = true);
    void test_dataset(int imgn, bool if_show, int ni = -1);
    void test_dataset_multi_init(int imgn, bool if_show, int ni);
    arma::fmat test_one(cv::Mat& gray, arma::fmat& face, bool if_show, int ni = -1);
    bool pred_pic(cv::Mat& gray, arma::fmat face, bool if_show);
    void gen_all_features(int n_stage, cv::Mat& gray, arma::fmat face);
    void train_only_stage(int n_stage, int idx_pool_size, int son_pool_size, arma::fmat lft, arma::fmat sies, float ts);
    // void train_last_stage(double lmd);
   

private:
    int n_stages_;

    int idx_pool_size_;
    int son_pool_size_;
    arma::fcube idx_pool_;
    // float ip[][][];
    arma::fcube son_pool_;
    // float sp[][][5];
    arma::fcube A; // final reg. model
    // arma::fcube sdh_model;

    int annonum;
    int nn;
    int imgnum;
    arma::fmat lamda_final_train;
    arma::fmat sc_in_each_stage;
    float train_scale;         // witch the training is 
    // float scale_shrink;        
    arma::fmat idx_pool_tube;    // pool delta dx & dy

    arma::fmat indexed_features; // shape-indexed features
    arma::fmat final_features;   // cascade all selected features

    arma::fmat face_last_stage;      // total face coordinates 
    arma::fmat face_current_stage;

    arma::fmat dist_diff;        // face alignment error OR so-called label
    // int nfl_;
    // float (*a)[5441][136];

public:
    // cv::Mat img_;

};


#endif