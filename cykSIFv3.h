#ifndef CYK_SIF_
#define CYK_SIF_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <armadillo>


class cykSIF{
public:

	cykSIF(); 
    void prepare_data(int n_stage);
    void select_son_pool(int n_stage);
    void sdh_train(int n_stage);
    void stage_train(int n_stage);
    void init(int n_stages, int idx_pool_size, int son_pool_size); 
    void train(int n_stages, int idx_pool_size, int son_pool_size);
    void test(int n_stage, bool if_training);
    void load_model();
    void save_model();
    void gen_final_features(int n_stage);
    void show_pre(arma::mat& in_label, arma::mat& in_pre, bool if_training);
    void gen_dist_diff();
    void update_face(int n_stages);
    void draw_face(cv::Mat& pano, arma::mat face, cv::Scalar sc);
    void genDesPool();   
    void test_dataset(int imgn);
    bool pred_pic(cv::Mat& gray, arma::mat face, bool if_show);
    void gen_all_features(int n_stage, cv::Mat& gray, arma::mat face);

private:
    int n_stages_;

    int idx_pool_size_;
    int son_pool_size_;
    arma::cube idx_pool_;
    arma::cube son_pool_;
    arma::cube A; // final reg. model


    int annonum;
    int nn;
    int imgnum;
    arma::mat lamda_final_train;
    arma::mat sc_in_each_stage;
    double train_scale;         // witch the training is 
    // double scale_shrink;        
    arma::mat idx_pool_tube;    // pool delta dx & dy

    arma::mat indexed_features; // shape-indexed features
    arma::mat final_features;   // cascade all selected features

    arma::mat face_last_stage;      // total face coordinates 
    arma::mat face_current_stage;

    arma::mat dist_diff;        // face alignment error OR so-called label
    // int nfl_;

    // cv::Mat img_;

};


#endif