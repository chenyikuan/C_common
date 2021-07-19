#include "cykSDH.h"
#include "cykTools.h"

#include<iomanip>
#include <cmath>

using namespace std;
using namespace arma;

SDH::SDH(){
    n_anchors = 1000;
    sigma = 1;
    maxItr = 15;
    g_lambda = 1; // original 1.0
    nu = 0.0001;
    nbits = 32;
    f_lambda = 100; // original 1e-5
	if_classification = true;
	if_using_anchor = true;
}

void SDH::init(sdh_prms prms){

    nbits = prms.nb;
	maxItr = prms.ni;

    n_anchors = prms.na;
	sigma = prms.sigma;

	g_lambda = prms.gl;  //control total err.
	f_lambda = prms.fl;  //control diff. of X & label; total err
                    //ex: 0.0001 should be small
	nu = prms.nu_;       //penalty of sign(F(X)) & hash-code: hash-code longer, nu should be smaller
                    //ex: nu = 64./nb

	if_classification = prms.if_classification;
	if_using_anchor = prms.if_using_anchor;
}

void SDH::normalise_features_train(fmat& data)
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

void SDH::normalise_features_apply(fmat& data)
{
	for (uword i = 0; i < data.n_rows; ++i)
	{
		data.row(i) -= normalise_params_data.row(0);
		data.row(i) /= normalise_params_data.row(1);
	}
}

void SDH::normalise_labels_train(fmat& label)
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

void SDH::normalise_labels_apply(fmat& label)
{
	for (uword i = 0; i < label.n_rows; ++i)
	{
		label.row(i) %= normalise_params_label.row(1); // element-wise multiplication
		label.row(i) += normalise_params_label.row(0);
	}
}

fmat SDH::predict(fmat td){
	normalise_features_apply(td);
	//fmat Phix;// = td;
	if (if_using_anchor)
	{
		X = td;
		int Ntest = X.n_rows;
		n_anchors = anchor.n_rows;
		//std::cout << "Generating Phix ..."<< std::endl;
		Phix.ones(Ntest, n_anchors + 1);
		float si = 2.0* sigma* sigma;
		fmat xt = sum(square(X), 1);// 60000 * 1
		fmat yt = sum(square(anchor), 1).t();//1000 * 1
		fmat pt(Ntest, n_anchors);
		for (uword i = 0; i < n_anchors; i++)
			pt.col(i) = xt;
		for (uword i = 0; i < Ntest; i++)
			pt.row(i) += yt;
		pt -= 2.* X * anchor.t();
		Phix.cols(0, Phix.n_cols - 2) = pt;
		Phix /= si;
		Phix = arma::exp(-Phix);
	}
	else
		Phix = td;
    
	//std::cout << "predicting ..." <<std::endl;
    fmat B = arma::sign(Phix * Wf);
    fmat pred = B*Wg;

	if (if_classification)
	{
		uword idx;
		for (uword i = 0; i < pred.n_rows; i++)
		{
			pred.row(i).max(idx);
			pred(i, 0) = idx;
		}
		pred = pred.col(0);
	}
	else
		normalise_labels_apply(pred);
    
    return pred;
}

bool SDH::train(fmat td, fmat tl, std::string fn){

	cout << "< --------------- cykSDH ---------------- >" << endl;
	float validation_ratio = 0.3;
	uword train_num = (1 - validation_ratio) * td.n_rows;
	fmat td_validation = td.rows(train_num, td.n_rows-1);
	fmat tl_validation = tl.rows(train_num, td.n_rows-1);
	td = td.rows(0, train_num);
	tl = tl.rows(0, train_num);
	cout << "  Using " << validation_ratio * 100 << "% td as validation set." << endl;
	cout << "    Train set num: " << td.n_rows << endl;
	cout << "    Validation set num: " << td_validation.n_rows << endl;
	cout << "  --------- Trianing start ---------" << endl;
    
	X = td;
	label = tl;
	Ntrain = X.n_rows;
	leng = X.n_cols;
	normalise_features_train(X);
	if (!if_classification)
		normalise_labels_train(label);
    
	if (if_using_anchor)
	{
		std::cout << "Generating anchors ..." << std::endl;
		anchor.set_size(n_anchors, X.n_cols);
		for (uword i = 0; i < n_anchors; i++) {
			uword k = round(CYK_TOOLS.random(0, Ntrain - 1));
			anchor.row(i) = X.row(k);
		}
		std::cout << "Generating Phix ..." << std::endl;
		Phix.ones(Ntrain, n_anchors + 1);
		float si = 2.0*sigma*sigma;
		fmat xt = sum(square(X), 1);// 60000 * 1
		fmat yt = sum(square(anchor), 1);//1000 * 1
		yt = yt.t();
		fmat pt(Ntrain, n_anchors);
		for (uword i = 0; i < n_anchors; i++)
			pt.col(i) = xt;
		for (uword i = 0; i < Ntrain; i++)
			pt.row(i) += yt;
		pt -= 2.* X * anchor.t();
		Phix.cols(0, Phix.n_cols - 2) = pt;
		Phix /= si;
		cout << "Original sigma: " << sigma << endl;
		float ad_sigma = 1.f;
		fmat mean_phix;
		cout << "mean_phix: ";
		do
		{
			Phix /= ad_sigma;
			sigma *= sqrtf(ad_sigma);
			mean_phix = mean(mean(Phix, 0), 1);
			ad_sigma = 2.f;
			cout << mean_phix(0, 0) << "->";
		} while (mean_phix(0, 0) > 3);
		cout << "\b\b  \nAfter refinement sigma: " << sigma << endl;
		Phix = arma::exp<fmat>(-Phix);
		mean_phix = mean(mean(Phix, 0), 1);
		if (mean_phix(0, 0) < 0.1)
		{
			cout << "[ WARNING ]: bad sigma!" << endl;
			CYK_TOOLS.pause();
		}
	}
	else
		Phix = X;

    // learn G and F
    zinit.randu(Ntrain, nbits);
    zinit = sign(zinit-0.5);

	//Wg = RRC(zinit, label, g_lambda);
	Wg = CYK_TOOLS.LSRf(zinit, label, g_lambda);

	float me_bias = norm(conv_to<mat>::from(zinit*Wg - label), "fro");
	// float me_bias = norm(zinit*Wg - label, "fro");
	std::cout << "### random bias ### = "<<me_bias<<std::endl;
	me_bias = norm(conv_to<mat>::from(label), "fro");
	// me_bias = norm(label, "fro");
	std::cout << "Label fro: "<< me_bias <<std::endl;

    if(!calc())
        return false;
    


    std::cout << "Saving data ..."<<std::endl;
    
    save(fn);
    
    std::cout << "Done!"<<std::endl;
	fmat validation_pred = predict(td_validation);
	float e_all = norm(conv_to<mat>::from(tl_validation), "fro");
	e_all = (e_all == 0) ? 1 : e_all;
	float e = norm(conv_to<mat>::from(tl_validation - validation_pred), "fro");
	cout << "  Validation set error: " << e / e_all * 100.f << "%" << endl;


	return true;
    
}

bool SDH::calc(void ){

    std::cout << "Calculating ... "<<std::endl;
	//float tol = FLT_MIN;
    
//     double delta = 1.0/nu;
    
	if (if_classification)
	{
 		// ============== FOR CLASSIFICATION !!! ===============
		// =========== project label to a N * c matrix =========
		if (label.n_cols == 1) {
			label = round(label);
			fmat tmp = label;
			std::cout <<"label minimun: "<< arma::min(label);
			std::cout <<"label maxmun:  "<< arma::max(label);
			float m = arma::max(arma::max(label));
			if (label.min() > 0)
				cout << "[ WARNING ]: label minimum greater then 0 !" << endl;
         
			label.set_size(Ntrain, m+1);
			label.fill(0.0);
			for (int i=0; i<Ntrain; i++) {
				if (tmp(i, 0)<0) {
					std::cout << "label less then zero! " <<std::endl;
					return false;
				}
				label(i, tmp(i, 0)) = 1;
			}
		}
 		// ======================================================
	}

    std::cout << "G-step init."<<std::endl;
	Wg = CYK_TOOLS.LSRf(zinit, label, g_lambda);
    std::cout << "F-step init."<<std::endl;
	mat pm, tdx;
	tdx = conv_to<mat>::from(Phix);
	if (Phix.n_rows > Phix.n_cols) {
		tdx = tdx.t()*tdx;
		for (uword i = 0; i < Phix.n_cols; i++)
			tdx(i, i) += f_lambda;
		pm = solve(tdx, conv_to<mat>::from(Phix.t()), solve_opts::fast);
	}
	else{
		tdx = tdx*tdx.t();
		for (uword i = 0; i < Phix.n_rows; i++)
			tdx(i, i) += f_lambda;
		pm = solve(tdx, conv_to<mat>::from(Phix), solve_opts::fast);
		pm = pm.t();
	}
	tdx.clear();
	Wf = conv_to<fmat>::from(pm * conv_to<mat>::from(zinit));
	//Wf = CYK_TOOLS.LSRf(Phix, zinit, f_lambda);
	float total_bias = 0;

	fmat XF, Q;
    for (int i=0; i<maxItr; i++)
	{
        
        std::cout << "Iteration: "<< i << std::endl;
        XF = Phix * Wf;
        Q = nu * XF + label * Wg.t();
        
		//zinit.fill(0.0); // why?????
        for (int j=0; j<10; j++) {
           std::cout << j <<"-";
            fmat Z0 = zinit;
            for (int t=0; t<zinit.n_cols; t++) {
                fmat Zk(zinit.n_rows, zinit.n_cols-1);
                if (t == 0) {
                    Zk = zinit.cols(1, zinit.n_cols-1);
                }
                else if (t == Zk.n_cols){
                    Zk = zinit.cols(0, Zk.n_cols-1);
                }
                else{
                    Zk.cols(0,t-1) = zinit.cols(0, t-1);
                    Zk.cols(t,Zk.n_cols-1) = zinit.cols(t+1, Zk.n_cols);
                }
                fmat Wkk = Wg.row(t);
                fmat Wk(Wg.n_rows-1, Wg.n_cols);
                if (t == 0) {
                    Wk = Wg.rows(1, Wg.n_rows-1);
                }
                else if (t == Wk.n_rows){
                    Wk = Wg.rows(0, Wk.n_rows-1);
                }
                else{
                    Wk.rows(0,t-1) = Wg.rows(0, t-1);
                    Wk.rows(t,Wk.n_rows-1) = Wg.rows(t+1, Wk.n_rows);
                }
                
                zinit.col(t) = arma::sign(Q.col(t) - Zk*Wk*Wkk.t());
            }
            if (arma::norm(conv_to<mat>::from(zinit - Z0), "fro") < 1e-6 * arma::norm(conv_to<mat>::from(Z0), "fro")) {
               std::cout << "break" << std::endl;
                break;
            }
        }
        
		Wg = CYK_TOOLS.LSRf(zinit, label, g_lambda);
		fmat W0 = Wf;
		Wf = conv_to<fmat>::from(pm * conv_to<mat>::from(zinit));
		//Wf = pm * zinit;
		//Wf = CYK_TOOLS.LSRf(Phix, zinit, f_lambda);
		//Wf = conv_to<fmat>::from(CYK_TOOLS.LSR(conv_to<mat>::from(Phix), conv_to<mat>::from(zinit), f_lambda));

        float P_bias = norm(conv_to<mat>::from(zinit - Phix*Wf), "fro") / norm(conv_to<mat>::from(zinit), "fro");
        // float P_bias = norm(zinit - Phix*Wf, "fro") / norm(zinit, "fro");
		std::cout << "P_bias(fisrt) = " << P_bias * 100 << "%" << std::endl;

		float me_bias = norm(conv_to<mat>::from(zinit*Wg - label), "fro") / norm(conv_to<mat>::from(label), "fro");
 		std::cout << "Reg(second) bias = " << me_bias * 100 << "%" << std::endl;

		float old_bias = total_bias;
		total_bias = me_bias + nu * P_bias;
		std::cout << "Total bisa(weighted) = " << total_bias * 100 << "%"<< std::endl;
		
		float tol = 1e-3;
		if (P_bias < tol)
		{
			cout << "[ P_bias break ]" << endl;
			break;
		}

		if (norm(conv_to<mat>::from(Wf-W0), "fro") < tol * norm(conv_to<mat>::from(W0), "fro"))
		{
			cout << "[ Wf break ]" << endl;
			break;
		}
		cout << endl;

    }
    
    return true;
}

void SDH::save(std::string fn){
	normalise_params_data.save(fn + "_normalise_prms.dat");
	if (if_using_anchor)
		anchor.save(fn + "_anchor.dat");
    Wf.save(fn+"_Wf.dat");
    Wg.save((fn+"_Wg.dat").c_str());
	fmat sigma_fmat;
	sigma_fmat << sigma << endr;
	sigma_fmat.save(fn + "_sigma.dat");
	if (!if_classification)
		normalise_params_label.save(fn + "_normalise_prms_tl.dat");
//    std::cout << anchor.n_rows <<"#"<< anchor.n_cols <<"#"<< Wf.n_rows <<"#"<< Wf.n_cols <<"#"<< Wg.n_rows <<"#"<< Wg.n_cols <<std::endl;
}

void SDH::load(std::string fn){
	normalise_params_data.load(fn + "_normalise_prms.dat");
	if (if_using_anchor)
		anchor.load(fn + "_anchor.dat");
    Wf.load(fn+"_Wf.dat");
    Wg.load((fn+"_Wg.dat").c_str());
	fmat sigma_fmat;
	if (!sigma_fmat.load(fn + "_sigma.dat"))
	{
		cout << "Sigma load failed, init to 1." << endl;
		sigma = 1;
	}
	else
		sigma = sigma_fmat(0,0);
	if (!if_classification)
		normalise_params_label.load(fn + "_normalise_prms_tl.dat");
}


fmat SDH::get_B(fmat td)
{
	normalise_features_apply(td);
	if (if_using_anchor)
	{
		X = td;
		int Ntest = X.n_rows;
		n_anchors = anchor.n_rows;
		//std::cout << "Generating Phix ..."<< std::endl;
		fmat Phix = td;
		Phix.ones(Ntest, n_anchors + 1);
		float si = 2.0* sigma* sigma;
		fmat xt = sum(square(X), 1);// 60000 * 1
		fmat yt = sum(square(anchor), 1).t();//1000 * 1
		fmat pt(Ntest, n_anchors);
		for (uword i = 0; i < n_anchors; i++)
			pt.col(i) = xt;
		for (uword i = 0; i < Ntest; i++)
			pt.row(i) += yt;
		pt -= 2.* X * anchor.t();
		Phix.cols(0, Phix.n_cols - 2) = pt;
		Phix /= si;
		Phix = arma::exp(-Phix);
	}
	else
		Phix = td;

	//    std::cout << "predicting ..." <<std::endl;
	fmat B = Phix * Wf;
	return B;
}

// old testing, not used anymore
fmat SDH::getWf(fmat B, fmat Px){
	fmat wf;
	wf = zeros<fmat>(Px.n_cols, B.n_cols);
	
// 	Px.row(100).print("dd");
	for (uword i=0; i<B.n_cols; i++)
	{
		fmat bt = B.col(i);
		uword idx = 0;
		float mi = FLT_MAX;
		for (uword j=0; j<Px.n_cols; j++)
		{
			fmat xt = Px.col(j);
			float mt = norm(conv_to<mat>::from(bt-sign(xt)), 2);
			if (mt < mi)
			{
				idx = j;
				mi = mt;
			}
// 			xt.print("xt");
		}
		wf(idx, i) = 1.f;
	}
	return wf;
}














