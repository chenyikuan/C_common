//
//  main.cpp
//  GBPI
//
//  Created by ³ÂÒÒ¿í on 15/9/16.
//  Copyright (c) 2015Äê ³ÂÒÒ¿í. All rights reserved.
//

#include <iostream>
#include "cykTools.h"
#include "cnpy.h"


using namespace std;
using namespace arma;

int main(int argc, const char * argv[]) {

	cyk_numpy_tools numpy_tools;

	arma::mat arma_double;
	numpy_tools.loadNPY("np_double.npy", arma_double);
	arma_double.print("arma_double");

	// np_float = np.zeros([6,6], np.float32) in python
	arma::fmat arma_float;
	numpy_tools.loadNPY("np_float.npy", arma_float);
	arma_float.print("arma_float");

	arma::arma_rng::set_seed_random();
	arma_double.randn(4, 5);
	arma_double.print("arma_double");
	arma_float.randn(3, 3);
	arma_float.print("arma_float");

	numpy_tools.save2NPY("arma_double.npy", arma_double);
	numpy_tools.save2NPY("arma_float.npy", arma_float);

	////const unsigned int shape1[] = { arma_double.n_rows, arma_double.n_cols };
	//umat shape;
	//shape << arma_double.n_rows << arma_double.n_cols << endr;
	//cnpy::npy_save("arma_double.npy", arma_double.memptr(), shape);
	////const unsigned int shape2[] = { arma_float.n_rows, arma_float.n_cols };
	//umat shape2;
	//shape2 << arma_float.n_rows << arma_float.n_cols << endr;
	//cnpy::npy_save("arma_float.npy", arma_float.memptr(), shape2);

	
	CYK_TOOLS.pause();


	return 0;
}


















