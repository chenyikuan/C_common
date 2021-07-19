#ifndef CYK_FERNS_WITH_BF_
#define CYK_FERNS_WITH_BF_

#include <iostream>
#include <armadillo>
#include "cykFerns.h"

class cykFerns_with_bf: public cykFerns{
public:
	arma::fmat fernsRegTrain(arma::fmat data, arma::fmat ys, arma::fmat bf, cyk_fern_prms pa, std::string fn);

     arma::fmat fernsRegApply(arma::fmat data, arma::fmat bf);

private:
     void trainFern(arma::fmat& data, arma::fmat ys, arma::fmat bf);

	arma::fmat mu_all_layer;
	arma::fmat mu;
};


#endif