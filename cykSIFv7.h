#ifndef CYK_SIF_
#define CYK_SIF_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <armadillo>


class cykSIF{
public:

    cykSIF(); 
    ~cykSIF();

    void save_model();
    void load_model(bool if_final_use);
    void gen_final_features(int ni);
    void prepare_data(int ni, bool if_train, bool if_save);
    void test_dataset(int ni, bool if_train, bool if_save);
    void draw_face(cv::Mat& pano, arma::fmat face, cv::Scalar sc);
    float show_err(arma::fmat& total_err, bool if_show);

    arma::fmat test_one(cv::Mat& gray, arma::fmat& face, bool if_show, int ni);
    arma::fmat fern_func(int ni, arma::fmat& data);
    void load4Model();
    void test_dataset_onebyone(int imgn, bool if_show, int ni);
    void test_dataset_multi_init(int imgn, bool if_show, int ni);


private:

    std::string data_path;

    int annonum;
    int nn;
    int imgnum;
    int n_stages_;
    float train_scale;         // witch the training is 
    arma::fmat sc_in_each_stage;

    int idx_pool_size_;
    int son_pool_size_;
    arma::fcube idx_pool_;
    arma::fcube son_pool_;

    arma::fmat indexed_features; // shape-indexed features
    arma::fmat final_features;   // cascade all selected features

    arma::fmat face_last_stage;      // total face coordinates 
    arma::fmat face_current_stage;      // total face coordinates 

    arma::umat fids_0;// = arma::zeros<arma::umat>(pa.M, pa.S);
    arma::fmat thrs_0;// = arma::zeros<arma::fmat>(pa.M, pa.S);
    arma::fcube ysFern_0;// = arma::zeros<arma::fmat>(pow(2, pa.S), pa.M);
    arma::umat fids_1;// = arma::zeros<arma::umat>(pa.M, pa.S);
    arma::fmat thrs_1;// = arma::zeros<arma::fmat>(pa.M, pa.S);
    arma::fcube ysFern_1;// = arma::zeros<arma::fmat>(pow(2, pa.S), pa.M);
    arma::umat fids_2;// = arma::zeros<arma::umat>(pa.M, pa.S);
    arma::fmat thrs_2;// = arma::zeros<arma::fmat>(pa.M, pa.S);
    arma::fcube ysFern_2;// = arma::zeros<arma::fmat>(pow(2, pa.S), pa.M);
    arma::umat fids_3;// = arma::zeros<arma::umat>(pa.M, pa.S);
    arma::fmat thrs_3;// = arma::zeros<arma::fmat>(pa.M, pa.S);
    arma::fcube ysFern_3;// = arma::zeros<arma::fmat>(pow(2, pa.S), pa.M);

};


#endif