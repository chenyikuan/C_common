//


#ifndef CYK_SDH_WITH_FERNS_H_
#define CYK_SDH_WITH_FERNS_H_

#include <armadillo>
#include "cykFerns.h"

#ifndef CYK_SDH_with_ferns_WITH_FERNS_PRMS_
#define CYK_SDH_with_ferns_WITH_FERNS_PRMS_
struct sdh_with_ferns_prms
{
	sdh_with_ferns_prms():	nb(32),
				ni(15),
				gl(1),
				nu_(0.01),
				if_classification(true)
				{}
	int nb;
	int ni;
	float gl;
	float nu_;
	bool if_classification;
	cyk_fern_prms fern_prm;
};


#endif

class SDH_with_ferns {
    
    
private:
    arma::uword Ntrain;
    int leng;
    
    int maxItr;
    float g_lambda;
    float nu;
    int nbits;
	cyk_fern_prms fern_prm;

    arma::fmat X;
    arma::fmat label;

    arma::fmat zinit;

    arma::fmat Wg;
    // arma::fmat Wf;
    cykFerns Wf;

    
	bool if_classification;
	arma::fmat normalise_params_label;

public:
    SDH_with_ferns();
	void init(sdh_with_ferns_prms prms);
	// training SDH_with_ferns
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
	// testing SDH_with_ferns
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

    bool calc( std::string fn );

};


#endif
