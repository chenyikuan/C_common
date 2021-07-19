#ifndef CYK_FERNS_H_
#define CYK_FERNS_H_

#include <iostream>
// #include <opencv2/opencv.hpp>
#include <armadillo>

#ifndef CYK_FERNS_PRMS_
#define CYK_FERNS_PRMS_
struct cyk_fern_prms
{
	cyk_fern_prms() : 
		eta(0.1),
		S(3),
		M(100),
		R(100),
		validation_ratio(0.f),
		if_show(true)
	{}
	float eta;
     int S;
     int M;
     int R;
	 float validation_ratio;
     bool if_show;
};
#endif

class cykFerns{
public:
    cykFerns();

    // train ferns with auto model saved
     virtual arma::fmat fernsRegTrain(arma::fmat data, arma::fmat ys, cyk_fern_prms pa, std::string fn);

    // apply ferns: model needed to be loaded alone
     virtual arma::fmat fernsRegApply(arma::fmat data);

	 // get inds as feature
	 virtual arma::umat get_inds(arma::fmat data);
     
     bool loadFerns(std::string fn);

protected:
     void trainFern(arma::fmat& data, arma::fmat ys);
     void init_params(cyk_fern_prms pa);
    
     arma::umat fernsInds(arma::fmat& data, arma::umat& fids_, arma::fmat& thrs_);
     void saveFerns(std::string fn);

	 void normalise_features_train(arma::fmat& data);
	 void normalise_features_apply(arma::fmat& data);
	 void normalise_labels_train(arma::fmat& label);
	 void normalise_labels_apply(arma::fmat& label);

	 void train_state_analysis(arma::fmat ys, arma::fmat ysSum, std::string fn);

protected:
     float eta;
     int S;
     int M;
     int R;
	 float validation_ratio;
     bool if_show;
     bool model_loaded;

     int N;
     int F;
     int K;         // label dimentions

	 arma::fmat normalise_params_data;
	 arma::fmat normalise_params_label;

    arma::umat fids;// = arma::zeros<arma::umat>(pa.M, pa.S);
    arma::fmat thrs;// = arma::zeros<arma::fmat>(pa.M, pa.S);
    arma::fmat ysSum;// = arma::zeros<arma::fmat>(N, 1);
    arma::fcube ysFern;// = arma::zeros<arma::fmat>(pow(2, pa.S), pa.M);

    arma::umat fids1;
    arma::fmat thrs1;
    arma::fmat ysFern1;
    arma::fmat ys1;

    arma::umat best_fids1;
    arma::fmat best_thrs1;
    arma::fmat best_ysFern1;
    arma::fmat best_ys1;

};


#endif