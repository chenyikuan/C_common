#include "cykFerns_feature.h"
#include <math.h>
#include "cykTools.h"

using namespace std;
using namespace arma;

arma::fmat cykFerns_feature::fernsRegTrain(arma::fmat data, arma::fmat ys,
	cyk_fern_prms pa, string fn){

	cout << "< -------- cykFerns version: 2 ------- >" << endl;

	normalise_features_train(data);
	normalise_labels_train(ys);

	init_params(pa);
	N = data.n_rows;
	F = data.n_cols;
	K = ys.n_cols;
	fids = arma::zeros<arma::umat>(M, S);
	thrs = arma::zeros<arma::fmat>(M, S);
	ysSum = arma::zeros<arma::fmat>(N, K);
	ysFern = arma::zeros<arma::fcube>(pow(2, S), K, M);

	double e = norm(conv_to<mat>::from(ys), "fro");
	if (if_show)
	{
		cout << "phase original error = " << e / ys.n_rows << endl;
	}

	bool if_best_exits;
	if (!if_show)
	{
		cout << "|-------------------Train-ferns--------------------|" << endl;
		cout << "|>|" << flush;
	}
	int cd = M / 50;

	for (int m = 0; m < M; ++m)
	{
		// cout <<  << "%" << endl;
		arma::fmat ysTar = ys - ysSum;
		e = norm(conv_to<mat>::from(ysTar), "fro");
		if (m == cd && !if_show)
		{
			// printf("\b\b->|");
			cout << "\b\b->|" << flush;
			cd = m + M / 50;
		}
		if (if_show)
		{
			cout << e << ", ";
		}
		if_best_exits = false;
		for (int r = 0; r < R; ++r)
		{
			trainFern(data, ysTar);
			double e1 = norm(conv_to<mat>::from(ysTar - ys1), "fro");
			// cout << ", "<< e1 - e;
			if (e >= e1)
			{
				e = e1;
				best_fids1 = fids1;
				best_ys1 = ys1;
				best_ysFern1 = ysFern1;
				best_thrs1 = thrs1;
				if_best_exits = true;
			}
		}
		// cout << endl;
		if (!if_best_exits)
		{
			cout << "best not exits!" << endl;
			CYK_TOOLS.pause();
			exit(-1);
		}
		fids.row(m) = best_fids1;
		thrs.row(m) = best_thrs1;
		ysFern.slice(m) = best_ysFern1 * eta;
		ysSum += best_ys1 * eta;
		if (if_show)
		{
			cout << "phase = " << m << ", one sample error = " << e / ysTar.n_rows << endl;
		}
	}
	cout << endl;

	this->saveFerns(fn);

	return conv_to<fmat>::from(fids);
}

fmat cykFerns_feature::fernsRegApply(arma::fmat data){
	if (!model_loaded)
	{
		cout << "Ferns not loaded!" << endl;
		exit(-1);
	}

	normalise_features_apply(data);

	umat inds = fernsInds(data, fids, thrs);
	return conv_to<fmat>::from(inds);
}

void cykFerns_feature::saveFerns(string fn){
	cout << "Son's saving" << endl;
	fids.save(fn + "_fids.dat");
	thrs.save(fn + "_thrs.dat");
	normalise_params_data.save(fn + "_norm_params_data.dat");
	model_loaded = true;
}

bool cykFerns_feature::loadFerns(string fn){
	model_loaded = true;
	if (
		fids.load(fn + "_fids.dat") &&
		thrs.load(fn + "_thrs.dat") &&
		normalise_params_data.load(fn + "_norm_params_data.dat")
		)
		return true;
	else
		return false;
}
