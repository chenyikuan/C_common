#ifndef CYK_FERNS_FEATURE_H_
#define CYK_FERNS_FEATURE_H_

#include <iostream>
#include <armadillo>
#include "cykFerns.h"

class cykFerns_feature: public cykFerns{
public:
	arma::fmat fernsRegTrain(arma::fmat data, arma::fmat ys, cyk_fern_prms pa, std::string fn);

	arma::fmat fernsRegApply(arma::fmat data);

	bool loadFerns(std::string fn);

private:
	void saveFerns(std::string fn);
};


#endif