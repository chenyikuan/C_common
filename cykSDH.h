//


#ifndef CYK_SDH_H_
#define CYK_SDH_H_

#include <armadillo>

#ifndef CYK_SDH_PRMS_
#define CYK_SDH_PRMS_
struct sdh_prms
{
	sdh_prms():	na(1000), 
				sigma(1),
				nb(32),
				ni(15),
				gl(1),
				fl(100),
				nu_(0.01),
				if_classification(true),
				if_using_anchor(true)
				{}
	arma::uword na;
	float sigma;
	int nb;
	int ni;
	float gl;
	float fl;
	float nu_;
	bool if_classification;
	bool if_using_anchor;
};


#endif

class SDH {
    
    
private:
    arma::uword Ntrain;
    int leng;
    
    int maxItr;
    float g_lambda;
    float f_lambda;
    float nu;
    int nbits;

    arma::fmat X;
    arma::fmat label;

    arma::fmat Phix;
    arma::fmat zinit;

    arma::uword n_anchors;
    float sigma;
    
	arma::fmat anchor;
    arma::fmat Wg;
    arma::fmat Wf;
    
	bool if_classification;
	bool if_using_anchor;
	arma::fmat normalise_params_data;
	arma::fmat normalise_params_label;

public:
    SDH();
	void init(sdh_prms prms);
	// training SDH
	//  ============= notes =============
	// argmin: |y - B*wg| + gl*|Wf| + nu*|sign(F(x)) - B|
	// s.t. F(x) = X * wf
	// s.t. sign(F(x)) = B --> argmin: |B - X*wf| + fl*|wf|
	//
	// but countable result is:
	//      min: { sign(X * wf) <===> y }
	//  in practice: we can always set gl small enough to let wg overfit to let B be close
	//              to y, so we should mainly focus on diff of X & sign(X*wf), that's to say
	//              don't let gl be too small, but fl small to minimize err between X & X*wf
	// X ---> B <----> label(y)
	bool train(arma::fmat td, arma::fmat tl, std::string fn);
	// testing SDH
	arma::fmat predict(arma::fmat td);
	// loading model
	void load(std::string fn);
	// get B
	arma::fmat get_B(arma::fmat td);

private:
	// saving model
	void save(std::string fn);
	void normalise_features_train(arma::fmat& data);
	void normalise_features_apply(arma::fmat& data);
	void normalise_labels_train(arma::fmat& label);
	void normalise_labels_apply(arma::fmat& label);

    bool calc( void );
    arma::fmat getWf(arma::fmat B, arma::fmat Px);

};


#endif
