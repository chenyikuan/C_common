#include "cykFerns.h"
#include "cykTools.h"
#include <math.h>
#include <iomanip>

using namespace std;
using namespace arma;

cykFerns::cykFerns(){
    arma_rng::set_seed_random();
	validation_ratio = 0;
    model_loaded = false;
}

void cykFerns::init_params(cyk_fern_prms pa){
    eta = pa.eta;
    S = pa.S;
    M = pa.M;
    R = pa.R;
	validation_ratio = pa.validation_ratio;
    if_show = pa.if_show;    
}

void cykFerns::normalise_features_train(fmat& data)
{
	normalise_params_data = zeros<fmat>(2, data.n_cols);

	normalise_params_data.row(0) = mean(data);
	for (uword i = 0; i < data.n_rows; ++i)
	{
		data.row(i) -= normalise_params_data.row(0);
	}

	normalise_params_data.row(1) = max(abs(data));
	for (uword i = 0; i < data.n_cols; i++)
	{
		normalise_params_data(1, i) = normalise_params_data(1, i) < 1e-5 ? 1 : normalise_params_data(1, i);
	}
	for (uword i = 0; i < data.n_rows; ++i)
	{
		data.row(i) /= normalise_params_data.row(1);
	}
}

void cykFerns::normalise_features_apply(fmat& data)
{
	for (uword i = 0; i < data.n_rows; ++i)
	{
		data.row(i) -= normalise_params_data.row(0);
		data.row(i) /= normalise_params_data.row(1);
	}
}

void cykFerns::normalise_labels_train(fmat& label)
{
	normalise_params_label = zeros<fmat>(2, label.n_cols);

	normalise_params_label.row(0) = mean(label);
	for (uword i = 0; i < label.n_rows; ++i)
	{
		label.row(i) -= normalise_params_label.row(0);
	}

	normalise_params_label.row(1) = max(abs(label));
	for (uword i = 0; i < label.n_cols; i++)
	{
		normalise_params_label(1, i) = normalise_params_label(1, i) < 1e-5 ? 1 : normalise_params_label(1, i);
	}
	for (uword i = 0; i < label.n_rows; ++i)
	{
		label.row(i) /= normalise_params_label.row(1);
	}
}

void cykFerns::normalise_labels_apply(fmat& label)
{
	for (uword i = 0; i < label.n_rows; ++i)
	{
		label.row(i) %= normalise_params_label.row(1); // element-wise multiplication
		label.row(i) += normalise_params_label.row(0);
	}
}

arma::fmat cykFerns::fernsRegTrain(arma::fmat data, arma::fmat ys,
    cyk_fern_prms pa, string fn){
	init_params(pa);

	string fifn = fn + "_log_analysis.txt";
	ofstream flog(fifn.c_str(), ios::app);
	flog << "---------- analysis -----------------------------------------" << endl;
	flog << "eta: " << pa.eta << endl;
	flog << "M: " << pa.M << endl;
	flog << "R: " << pa.R << endl;
	flog << "S: " << pa.S << endl;
	flog.close();
	cout << "< -------- cykFerns version: 2 ------- >" << endl;

	//float validation_ratio = 0.1;
	uword train_num = (1 - validation_ratio) * data.n_rows;
	fmat data_validation, ys_validation;
	if (validation_ratio > 0)
	{
		data_validation = data.rows(train_num, data.n_rows - 1);
		ys_validation = ys.rows(train_num, data.n_rows - 1);
	}
	data = data.rows(0, train_num - 1);
	ys = ys.rows(0, train_num - 1);
	cout << "  Using " << validation_ratio * 100 << "% data as validation set." << endl;
	cout << "    Train set num: " << data.n_rows << endl;
	cout << "    Validation set num: " << data_validation.n_rows << endl;
	cout << "  --------- Trianing start ---------" << endl;

    normalise_features_train(data);
    normalise_labels_train(ys);

    N = data.n_rows;
    F = data.n_cols;
    K = ys.n_cols;
    fids = arma::zeros<arma::umat>(M, S);
    thrs = arma::zeros<arma::fmat>(M, S);
    ysSum = arma::zeros<arma::fmat>(N, K);
    ysFern = arma::zeros<arma::fcube>(pow(2, S), K, M);

    float e = norm(conv_to<mat>::from(ys), "fro");
	float e_all = sqrtf(e*e / ys.n_elem);
	e_all = (e_all == 0) ? 1 : e_all;
	float e_tmp;
    if (if_show)
    {
        cout << "| Phase original error: 100% | " << e_all << endl;
    }

    bool if_best_exits;
    if (!if_show)
    {
        cout << "|-------------------Train-ferns--------------------|"<< endl;
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
            cd = m + M/50;
        }
        if (if_show)
        {
			e_tmp = sqrtf(e*e / ys.n_elem);
			//cout << "-------------------------------------------------------------" << endl;
			cout << "| Phase " << setw(3) << m << ":" << setw(9) << e_tmp / e_all * 100 << "% | " << setw(8) << e_tmp << " -> " << flush;
        }
		if_best_exits = false;
        for (int r = 0; r < R; ++r)
        {
            trainFern(data, ysTar);
            float e1 = norm(conv_to<mat>::from(ysTar-ys1), "fro");
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
        ysFern.slice(m) = best_ysFern1 * ((m==(M-1))?1:eta);
		ysSum += best_ys1 * ((m == (M - 1)) ? 1 : eta);
        if (if_show)
        {
			e_tmp = sqrtf(e*e / ys.n_elem);
			cout << setw(8) << e_tmp / e_all * 100 << "% | " << e_tmp << endl;
			train_state_analysis(ys, ysSum, fn);
        }
    }
    cout << endl;

	saveFerns(fn);
	normalise_labels_apply(ysSum);

	if (validation_ratio != 0)
	{
		fmat validation_pred = fernsRegApply(data_validation);
		e_all = norm(conv_to<mat>::from(ys_validation), "fro");
		e_all = (e_all == 0) ? 1 : e_all;
		e = norm(conv_to<mat>::from(ys_validation - validation_pred), "fro");
		cout << "  Validation set error: " << e / e_all * 100.f << "%" << endl;
	}
	else
	{
		e = 0;
	}

	fifn = fn + "_log.txt";
	flog.open(fifn.c_str(), ios::app);
	flog << "----------------------------------------------" << endl;
    flog << "  eta: " << eta << endl;
    flog << "  S: " << S << endl;
    flog << "  M: " << M << endl;
	flog << "  R: " << R << endl;
	if (validation_ratio > 0)
		flog << "  Validation set error: " << e / e_all * 100.f << "%" << endl;
	else
		flog << "  No validation." << endl;
    flog.close();

    return ysSum;
}

void cykFerns::train_state_analysis(arma::fmat ys, arma::fmat ysSum, string fn)
{
	string fifn = fn + "_log_analysis.txt";
	ofstream flog(fifn.c_str(), ios::app);
	fmat e_show = zeros<fmat>(2, ys.n_cols);
	uword print_count = 0;
	for (uword i = 0; i < ys.n_cols; i++)
	{
		float e1 = norm(conv_to<mat>::from(ys.col(i)), "fro");
		float e2 = norm(conv_to<mat>::from(ys.col(i) - ysSum.col(i)), "fro");
		e1 = sqrtf(e1*e1 / ys.n_rows);
		e1 = (e1 == 0) ? 1 : e1;
		e2 = sqrtf(e2*e2 / ys.n_rows);
		flog << "|" << setw(4) << i << ":" << setw(8) << e2 / e1 * 100 << "%";
		e_show(0, i) = e2 / e1 * 100;
		if (print_count++ == 7)
		{
			flog << "|" << endl;
			print_count = 0;
		}
	}
	flog << "|" << endl;
	flog.close();
	//CYK_TOOLS.plot_arma("training ferns analysis", e_show);

}


void cykFerns::trainFern(arma::fmat& data, arma::fmat ys){
    arma::fmat tmp = sum(ys);
    // cout << tmp << endl;
    fmat mu = tmp/N;
    for (int i = 0; i < ys.n_rows; ++i)
        ys.row(i) -= mu;
    // ys = ys - mu;
    fids1 = arma::conv_to<arma::umat>::from(arma::randu(1,S) * F);
    // cout << fids1 ;
    thrs1 = arma::conv_to<arma::fmat>::from(arma::randu(1,S)) * 2 - 1;
    arma::umat inds = fernsInds(data, fids1, thrs1);
    // inds.rows(0,10).print("inds");
    ysFern1 = arma::zeros<arma::fmat>(pow(2,S), ys.n_cols);
    arma::fmat cnts = arma::zeros<arma::fmat>(pow(2,S), 1); 

    for (int n = 0; n < N; ++n)
    {
        arma::uword ind = inds(n,0);
        ysFern1.row(ind) += ys.row(n);
        // ysFern1(ind, 0) += ys(n, 0);
        cnts(ind, 0) ++;
    }
    for (int i = 0; i < ysFern1.n_rows; ++i)
    {
        ysFern1.row(i) = ysFern1.row(i) / (cnts(i,0)>2.2204e-16?cnts(i,0):2.2204e-16) + mu;
    }
    // ysFern1 = ysFern1 / arma::max(cnts+reg*N, 
    //     arma::ones<arma::fmat>(ysFern1.n_rows, ysFern1.n_cols)*2.2204e-16) + mu;
    ys1 = arma::zeros<arma::fmat>(inds.n_rows, K);
    for (int i = 0; i < inds.n_rows; ++i)
    {
        ys1.row(i) = ysFern1.row(inds(i,0));//, 0);
    }
}

arma::umat cykFerns::fernsInds(arma::fmat& data, arma::umat& fids_, arma::fmat& thrs_){
    int M_ = fids_.n_rows;
    int S_ = fids_.n_cols;
    int N_ = data.n_rows;
    arma::umat inds = arma::zeros<arma::umat>(N_, M_);
    for (int n = 0; n < N_; ++n)
    {
        for (int m = 0; m < M_; ++m)
        {
            for (int s = 0; s < S_; ++s)
            {
                inds(n, m)  = inds(n, m) << 1; // * 2
                if (data(n, fids_(m,s)) < thrs_(m,s))
                {
                    inds(n, m) ++;
                }
            }
        }
    }
    // inds = inds + 1;
    return inds;
}

fmat cykFerns::fernsRegApply(arma::fmat data){
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
    ysSum = zeros<fmat>(N,K);
    for (int m = 0; m < M; ++m)
    {
        for (int i = 0; i < N; ++i)
        {
            ysSum.row(i) += ysFern.slice(m).row(inds(i,m));
        }
    }

    normalise_labels_apply(ysSum);
    
    return ysSum;
}

umat cykFerns::get_inds(arma::fmat data)
{
	if (!model_loaded)
	{
		cout << "Ferns not loaded!" << endl;
		exit(-1);
	}

	normalise_features_apply(data);

	return fernsInds(data, fids, thrs);
}
// float cykFerns::fernsRegApplyOne(float)

void cykFerns::saveFerns(string fn){
    cout << "Saving model ..." << endl;
    fids.save(fn+"_fids.dat");
    thrs.save(fn+"_thrs.dat");
    ysFern.save(fn+"_ysFern.dat");
	normalise_params_data.save(fn + "_norm_params_data.dat");
	normalise_params_label.save(fn + "_norm_params_label.dat");
    model_loaded = true;
    cout << "Model saved." << endl;
}

bool cykFerns::loadFerns(string fn){
    model_loaded = true;
    if (
        fids.load(fn + "_fids.dat") &&
        thrs.load(fn + "_thrs.dat") &&
        ysFern.load(fn + "_ysFern.dat") &&
		normalise_params_data.load(fn + "_norm_params_data.dat") &&
		normalise_params_label.load(fn + "_norm_params_label.dat")
        )
        return true;
	else
	{
		cout << "Ferns not loaded, please check the model path!" << endl;
		CYK_TOOLS.pause();
		return false;
	}
}





