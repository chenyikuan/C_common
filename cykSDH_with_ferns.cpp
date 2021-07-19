#include "cykSDH_with_ferns.h"
#include "cykTools.h"

#include<iomanip>
#include <cmath>

using namespace std;
using namespace arma;

SDH_with_ferns::SDH_with_ferns(){
    maxItr = 15;
    g_lambda = 1; // original 1.0
    nu = 0.0001;
    nbits = 32;
	if_classification = true;

}

void SDH_with_ferns::init(sdh_with_ferns_prms prms){

    nbits = prms.nb;
	maxItr = prms.ni;

	g_lambda = prms.gl;  //control total err.
	fern_prm = prms.fern_prm;
	nu = prms.nu_;       //penalty of sign(F(X)) & hash-code: hash-code longer, nu should be smaller
                    //ex: nu = 64./nb

	if_classification = prms.if_classification;
}

void SDH_with_ferns::normalise_labels_train(fmat& label)
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

void SDH_with_ferns::normalise_labels_apply(fmat& label)
{
	for (uword i = 0; i < label.n_rows; ++i)
	{
		label.row(i) %= normalise_params_label.row(1); // element-wise multiplication
		label.row(i) += normalise_params_label.row(0);
	}
}

fmat SDH_with_ferns::predict(fmat td){

	//std::cout << "predicting ..." <<std::endl;
    fmat B = arma::sign(Wf.fernsRegApply(td));
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

bool SDH_with_ferns::train(fmat td, fmat tl, std::string fn){

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
	if (!if_classification)
		normalise_labels_train(label);
    

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

    if(!calc(fn))
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

bool SDH_with_ferns::calc(string fn){

    std::cout << "Calculating ... "<<std::endl;
    
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
    Wf.fernsRegTrain(X, zinit, fern_prm, fn);
	float total_bias = 0;

	fmat XF, Q;
    for (int i=0; i<maxItr; i++)
	{
        
        std::cout << "Iteration: "<< i << std::endl;
        XF = Wf.fernsRegApply(X);
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
            if (i==maxItr-1)
            	cout << "end" << endl;
        }
        
		Wg = CYK_TOOLS.LSRf(zinit, label, g_lambda);
	    Wf.fernsRegTrain(X, zinit, fern_prm, fn);

        float P_bias = norm(conv_to<mat>::from(zinit - Wf.fernsRegApply(X)), "fro") / norm(conv_to<mat>::from(zinit), "fro");
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

		cout << endl;

    }
    
    return true;
}

void SDH_with_ferns::save(std::string fn){
    Wg.save((fn+"_Wg.dat").c_str());
	if (!if_classification)
		normalise_params_label.save(fn + "_normalise_prms_tl.dat");
//    std::cout << anchor.n_rows <<"#"<< anchor.n_cols <<"#"<< Wf.n_rows <<"#"<< Wf.n_cols <<"#"<< Wg.n_rows <<"#"<< Wg.n_cols <<std::endl;
}

void SDH_with_ferns::load(std::string fn){
	Wf.loadFerns(fn);
    Wg.load((fn+"_Wg.dat").c_str());
	if (!if_classification)
		normalise_params_label.load(fn + "_normalise_prms_tl.dat");
}


fmat SDH_with_ferns::get_B(fmat td)
{
	return Wf.fernsRegApply(td);
}












