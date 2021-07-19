//
//  main.cpp
//  GBPI
//
//  Created by ³ÂÒÒ¿í on 15/9/16.
//  Copyright (c) 2015Äê ³ÂÒÒ¿í. All rights reserved.
//

#include <iostream>
#include "cykTools.h"
#include "cykFerns.h"
#include "cykFerns_feature.h"
#include "cykSDH.h"

using namespace std;
using namespace arma;

int main(int argc, const char * argv[]) {



	string mnist_path = "H:/mnist/";
	fmat td_train, td_test;
	fmat tl_train, tl_test;
	td_train.load(mnist_path + "td_train.dat");
	td_test.load(mnist_path + "td_test.dat");
	tl_train.load(mnist_path + "tl_train.dat");
	tl_test.load(mnist_path + "tl_test.dat");
	cout_arma_size(td_train);
	cout_arma_size(tl_train);
	cout_arma_size(td_test);
	cout_arma_size(tl_test);

//#define Train_stage
	SDH fr;
	sdh_prms fr_prms;
	fr_prms.if_classification = true;
	fr_prms.if_using_anchor = true;
#ifdef Train_stage
	fr_prms.fl = 1;
	fr_prms.gl = 1;
	fr_prms.na = 1000;
	fr_prms.nb = 32;
	fr_prms.ni = 10;
	fr_prms.nu_ = 1e-4;
	fr_prms.sigma = 1; // roughly given, refined in training stage
	fr.init(fr_prms);
	fr.train(td_train, tl_train, "mnist");
	for (uword i = 0; i < td_train.n_rows; i++)
	{
		fmat pre = fr.predict(td_train.row(i));
		cout << "Prediction: " << pre(0, 0) << ", " << "label: " << tl_train(i, 0) << endl;
		CYK_TOOLS.pause();
	}
#else
	fr.init(fr_prms);
	fr.load("mnist");
	uword wrong_num = 0;
	for (uword i = 0; i < td_test.n_rows; i++)
	{
		fmat pre = fr.predict(td_test.row(i));
		if (pre(0, 0) != tl_test(i, 0))
		{
			cout << "Prediction: " << pre(0, 0) << ", " << "label: " << tl_test(i, 0)  << " --- #" << i << endl;
			wrong_num++;
		}
	}
	cout << "Precision: " << 100-100.f*wrong_num / td_test.n_rows << "% with " << wrong_num << "/" << td_test.n_rows << " wrong predictions."<< endl;
	CYK_TOOLS.pause();
#endif



	return 0;
}


















