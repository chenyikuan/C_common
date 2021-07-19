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

using namespace std;
using namespace arma;

int main(int argc, const char * argv[]) {

	cykFerns reg;
	cykFerns_feature fea;

	int N = 1000;
	float sig = 0.5;

	fmat xs0 = randu<fmat>(N, 1);
	fmat ys0 = zeros<fmat>(xs0.n_rows, 1);
	for (int i = 0; i < ys0.n_cols; ++i)
	{
		fmat tmp = cos(xs0*3.1415*4) + (xs0 + 1) % (xs0 + 1);
		ys0.col(i) = tmp;
	}
	xs0 += randu<fmat>(N, 1) * 0.1;

	fmat xs1 = randu<fmat>(N, 1);
	//fmat ys1 = cos(xs1*3.1415 * 4) + (xs1 + 1) % (xs1 + 1) + randn<fmat>(N, 1)*sig;

//#define USE_FEATURE
#ifdef USE_FEATURE
	cyk_fern_prms fern_param_feature;
	fern_param_feature.eta = 0.1;      // learning rate in [0,1] I just feel it is as same functionality as reg
	fern_param_feature.S = 2;           // %  S - fern depth
	fern_param_feature.M = 100;        // %  M - number ferns
	fern_param_feature.R = 100;           // %  R - number repeats, set larger if always get "best not found!"
	fern_param_feature.if_show = false;
	cyk_fern_prms fern_param;
	fern_param.eta = 0.1;      // learning rate in [0,1] I just feel it is as same functionality as reg
	fern_param.S = 2;           // %  S - fern depth
	fern_param.M = 100;        // %  M - number ferns
	fern_param.R = 100;           // %  R - number repeats, set larger if always get "best not found!"
	fern_param.if_show = false;
	// use features;
	fea.fernsRegTrain(xs0, ys0, fern_param_feature, "feature_extraction");
	fmat features = fea.fernsRegApply(xs0);
	reg.fernsRegTrain(features, ys0, fern_param, "reg");
	features = fea.fernsRegApply(xs1);
	fmat ys_pre1 = reg.fernsRegApply(features);
#else
	cyk_fern_prms fern_param;
	fern_param.eta = 0.1;      // learning rate in [0,1] I just feel it is as same functionality as reg
	fern_param.S = 2;           // %  S - fern depth
	fern_param.M = 100;        // %  M - number ferns
	fern_param.R = 100;           // %  R - number repeats, set larger if always get "best not found!"
	fern_param.if_show = false;
	// directly:
	reg.fernsRegTrain(xs0, ys0, fern_param, "reg");
	fmat ys_pre1 = reg.fernsRegApply(xs1);
#endif

	cv::Mat pano = cv::Mat::ones(800, 500, CV_8UC1) * 255;
	cv::Mat train_pano;
	cv::cvtColor(pano, train_pano, CV_GRAY2RGB);
	for (int j = 0; j < ys0.n_cols; ++j)
	{
		int b = CYK_TOOLS.random(0, 255);
		int g = CYK_TOOLS.random(0, 255);
		int r = CYK_TOOLS.random(0, 255);
		for (int i = 0; i < xs0.n_rows; ++i)
		{
			cv::circle(train_pano, cv::Point(xs0(i, 0) * 500, 350 - ys0(i, j) * 50), 1, cv::Scalar(b, g, r), 2);
		}
	}
	for (int j = 0; j < ys_pre1.n_cols; ++j)
	{
		int b = CYK_TOOLS.random(0, 255);
		int g = CYK_TOOLS.random(0, 255);
		int r = CYK_TOOLS.random(0, 255);
		for (int i = 0; i < xs1.n_rows; ++i)
		{
			cv::circle(train_pano, cv::Point(xs1(i, 0) * 500, 350 - ys_pre1(i, j) * 50), 1, cv::Scalar(b, g, r), 2);
		}
	}
	cv::imshow("train_pano", train_pano);
	cv::waitKey();
	return 0;
}


















