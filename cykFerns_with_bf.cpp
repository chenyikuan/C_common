#include "cykFerns_with_bf.h"
#include <math.h>
#include <iomanip>
#include "cykTools.h"

using namespace std;
using namespace arma;

arma::fmat cykFerns_with_bf::fernsRegTrain(arma::fmat data, arma::fmat ys, fmat bf, cyk_fern_prms pa, string fn)
{
	
	init_params(pa);

	string fifn = fn + "_log_analysis.txt";
	ofstream flog(fifn.c_str(), ios::app);
	flog << "---------- analysis -----------------------------------------" << endl;
	flog << "eta: " << pa.eta << endl;
	flog << "M: " << pa.M << endl;
	flog << "R: " << pa.R << endl;
	flog << "S: " << pa.S << endl;
	flog.close();
	cout << "< -------- cykFerns with bf ------- >" << endl;

	uword train_num = (1.f - validation_ratio) * data.n_rows;
	fmat data_validation, ys_validation, bf_validation;
	if (validation_ratio > 0)
	{
		data_validation = data.rows(train_num, data.n_rows - 1);
		ys_validation = ys.rows(train_num, data.n_rows - 1);
		bf_validation = bf.rows(train_num, data.n_rows - 1);
	}
	data = data.rows(0, train_num - 1);
	ys = ys.rows(0, train_num - 1);
	bf = bf.rows(0, train_num - 1);
	cout << "  Using " << validation_ratio * 100 << "% data as validation set." << endl;
	cout << "    Train set num: " << data.n_rows << endl;
	cout << "    Validation set num: " << data_validation.n_rows << endl;
	cout << "  --------- Trianing start ---------" << endl;

	normalise_features_train(data);
	normalise_labels_train(ys);

	if (data.n_cols != bf.n_cols || data.n_rows != bf.n_rows)
	{
		cout << "data error!" << endl;
		CYK_TOOLS.pause();
		exit(-1);
	}
    N = data.n_rows;
    F = data.n_cols;
    K = ys.n_cols;
    fids = arma::zeros<arma::umat>(M, S);
    thrs = arma::zeros<arma::fmat>(M, S);
    ysSum = arma::zeros<arma::fmat>(N, K); // bf的长度和data应是一样的（F），用来衡量data的可信度
    ysFern = arma::zeros<arma::fcube>(pow(2, S), K, M);

	mu_all_layer = zeros<fmat>(M, K);

	fmat e_data = ys - ysSum;
    double e_data_show = norm(conv_to<mat>::from(e_data), "fro");
	float e_all = sqrtf(e_data_show*e_data_show / e_data.n_elem);
	e_all = e_all == 0 ? 1 : e_all;
	float e_tmp;
    if (if_show)
    {
		cout << "| Phase original error: 100% | " << e_all << endl;
	}

    bool if_best_exits;
    if (!if_show)
    {
        cout << "|-------------------Train-ferns-bf-----------------|"<< endl;
        cout << "|>|" << flush;
    }
    int cd = M / 50;

    for (int m = 0; m < M; ++m)
    {
        if (m == cd && !if_show)
        {
            // printf("\b\b->|");
            cout << "\b\b->|" << flush;
            cd = m + M/50;
        }
        // cout <<  << "%" << endl;
        arma::fmat ysTar = ys - ysSum;
        e_data_show = norm(conv_to<mat>::from(ysTar), "fro");
		if (if_show)
		{
			e_tmp = sqrtf(e_data_show*e_data_show / e_data.n_elem);
			cout << "| " << setw(8) << e_tmp / e_all * 100 << "% | " << setw(8) << e_tmp << " -> " << flush;
		}
		if_best_exits = false;
		mu = sum(ysTar) / N;
		mu_all_layer.row(m) = mu;
		// cout << mu << endl;
		for (int r = 0; r < R; ++r)
        {
            trainFern(data, ysTar, bf);
			e_data = ysTar - ys1;
            double e1 = norm(conv_to<mat>::from(e_data), "fro");
            // cout << ", "<< e1 - e;
			if (e_data_show >= e1)
            {
				e_data_show = e1;
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
		ysFern.slice(m) = best_ysFern1 * ((m == (M - 1)) ? 1 : eta);
		ysSum += best_ys1 * ((m == (M - 1)) ? 1 : eta);
        if (if_show)
        {
			e_tmp = sqrtf(e_data_show*e_data_show / e_data.n_elem);
			cout << "phase " << m << ": " << e_tmp / e_all * 100 << "% | " << e_tmp << endl;
			train_state_analysis(ys, ysSum, fn);
		}

		// update belief mask for this phase

    }
    cout << endl;

    saveFerns(fn);
	normalise_labels_apply(ysSum);

	if (validation_ratio > 0)
	{
		fmat validation_pred = fernsRegApply(data_validation, bf_validation);
		e_all = norm(conv_to<mat>::from(ys_validation), "fro");
		e_all = (e_all == 0) ? 1 : e_all;
		e_data_show = norm(conv_to<mat>::from(ys_validation - validation_pred), "fro");
		cout << "  Validation set error: " << e_data_show / e_all * 100.f << "%" << endl;
	}
	else
	{
		e_data_show = 0;
		cout << "  No validation." << endl;
	}

	fifn = fn + "_log.txt";
	flog.open(fifn.c_str(), ios::app);
    flog << "----------------------------------------------" << endl;
    flog << "  eta: " << eta << endl;
    flog << "  S: " << S << endl;
    flog << "  M: " << M << endl;
    flog << "  R: " << R << endl;
	if (validation_ratio > 0)
		flog << "  Validation set error: " << e_data_show / e_all * 100.f << "%" << endl;
	else
		flog << "  No validation." << endl;
	flog.close();

	return ysSum;
}

void cykFerns_with_bf::trainFern(arma::fmat& data, arma::fmat ys, arma::fmat bf){
	for (int i = 0; i < ys.n_rows; ++i)
		ys.row(i).cols(0, K-1) -= mu;
	// ys = ys - mu;
    fids1 = arma::conv_to<arma::umat>::from(arma::randu(1,S) * F);
    // cout << fids1 ;
    thrs1 = arma::conv_to<arma::fmat>::from(arma::randu(1,S) * 2 - 1);
    arma::umat inds = fernsInds(data, fids1, thrs1);
    // inds.rows(0,10).print("inds");
    ysFern1 = arma::zeros<arma::fmat>(pow(2,S), ys.n_cols);
    arma::fmat cnts = arma::zeros<arma::fmat>(pow(2,S), 1); 

    for (int n = 0; n < N; ++n)
    {
        arma::uword ind = inds(n,0);
		float bf_f = 0;
		for (int i = 0; i < S; i++)
		{
// 			cout << fids1(0, i) + K << ", " << ys(n, fids1(0, i) + K) << endl;
			bf_f += bf(n, fids1(0, i));
		}
		bf_f /= S;
		ysFern1.row(ind) += ys.row(n) * bf_f;
        // ysFern1(ind, 0) += ys(n, 0);
		cnts(ind, 0) += bf_f;
    }
    for (int i = 0; i < ysFern1.n_rows; ++i)
    {
        ysFern1.row(i) = ysFern1.row(i) / (cnts(i,0)>2.2204e-16?cnts(i,0):2.2204e-16);
    }
    // ysFern1 = ysFern1 / arma::max(cnts+reg*N, 
    //     arma::ones<arma::fmat>(ysFern1.n_rows, ysFern1.n_cols)*2.2204e-16) + mu;
    ys1 = arma::zeros<arma::fmat>(inds.n_rows, K);
    for (int i = 0; i < inds.n_rows; ++i)
    {
		float bf_f = 0;
		for (int j = 0; j < S; j++)
		{
			bf_f += bf(i, fids1(0, j));
		}
		bf_f /= S;
		ys1.row(i) = ysFern1.row(inds(i, 0));
		ys1.row(i) *= bf_f;
		ys1.row(i) += mu;
    }
}

fmat cykFerns_with_bf::fernsRegApply(arma::fmat data, fmat bf){
    if (!model_loaded)
    {
        cout << "Ferns not loaded!" << endl;
        exit(-1);
    }

	normalise_features_apply(data);

	umat inds = fernsInds(data, fids, thrs);
    N = inds.n_rows;
    M = inds.n_cols;
    K = ysFern.n_cols;
	S = fids.n_cols;
    ysSum = zeros<fmat>(N,K);
    for (int m = 0; m < M; ++m)
    {
        for (int i = 0; i < N; ++i)
        {
			float bf_f = 0;
			for (int j = 0; j < S; j++)
			{
				bf_f += bf(i, fids(m, j));
			}
			bf_f /= S;
			ysSum.row(i) += ysFern.slice(m).row(inds(i, m)) * bf_f;
// 			ysSum.row(i) += mu_all_layer.row(m);
        }
    }

	normalise_labels_apply(ysSum);

    return ysSum;
}



