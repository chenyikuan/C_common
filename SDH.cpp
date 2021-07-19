#include "SDH.h"
#include "cykTools.h"

#include<iomanip>
#include <cmath>

SDH::SDH(){
//     n_anchors = 1000;
//     sigma = 1;
    maxItr = 15;
    g_lambda = 1; // original 1.0
    nu = 0.0001;
    nbits = 32;
    f_lambda = 100; // original 1e-5
}

//SDH::SDH(int nb, int itr){
//    nbits = nb;
//    maxItr = itr;
//}

//void SDH::init(const char* td, const char* tl){
//    cyktools cyk;
    
//    std::cout << "Loading data ..."<<std::endl;
    
    //    std::cout << td <<std::endl;
//    X = cyk.readMat(td);
//    label = cyk.readMat(tl);
    
//    std::cout << "Loaded."<<std::endl;
void SDH::init(arma::mat td, arma::mat tl, int nb, int ni, double gl, double fl, double nu_){

    nbits = nb;
//     n_anchors = na;
    maxItr = ni;
    g_lambda = gl;
	f_lambda = fl;
	nu = nu_;
    X = td;
    label = tl;
    
    Ntrain = X.n_rows;
    leng = X.n_cols;

}

arma::mat SDH::predict(arma::mat td, arma::mat tl, bool ifshow){
	cyktools cyk;
    X = td;
    int Ntest = X.n_rows;
//     n_anchors = anchor.n_rows;
    
//    std::cout << "Generating Phix ..."<< std::endl;
    arma::mat Phix = td;
//     Phix.set_size(Ntest, n_anchors + 1);
//     Phix.fill(0.0); // set the end cols to zero
//     double si = 2.0* sigma* sigma;
//     arma::mat xt = sum(square(X), 1);// 60000 * 1
//     arma::mat yt = sum(square(anchor), 1);//1000 * 1
//     arma::mat pt(Ntest, n_anchors);
//     for (int i=0; i<Ntest; i++) {
//         for (int j=0; j< n_anchors; j++) {
//             pt(i, j) = xt(i,0) + yt(j,0);
//         }
//     }
//     pt -= 2.* X * anchor.t();
//     Phix.cols(0,Phix.n_cols-2) = pt;
//     Phix /= si;
//     Phix = arma::exp(Phix);
    
//    std::cout << "predicting ..." <<std::endl;
    arma::mat B = arma::sign(Phix * Wf);
//     B = (B+square(B))/2.0;//////////////////////////////////////////////////////////////////////////FUCK!!!!!
    arma::mat pred = B*Wg;
//     arma::mat c = ( Wg.t() * B.t()).t();
    
	if (ifshow)
	{
		arma::mat err = pred - tl;
        std::cout << std::setprecision(3);                         // 等价于 cout <<setprecision(4) ;
        std::cout << "err  " << "\t" << "pred  " << "\t" << "tl  " << std::endl;
		for (int i=0; i<Ntest; i++)
		{
			std::cout << err(i,0) << "\t" << pred(i,0) << "\t" << tl(i,0) << std::endl;
		}
		std::cout << "err  " << "\t" << "pred  " << "\t" << "tl  " << std::endl;
        std::cout << "tl min: " << arma::min(tl);
        std::cout << "tl max: " << arma::max(tl);
		std::cout << "label mean: "<<arma::mean(arma::abs(tl));
		std::cout << "mean error: "<<arma::mean(arma::abs(err));
//         std::cout << "win ratio: " << 1.0*ccc/err.n_rows <<std::endl;
	}

//     arma::mat pred = arma::ones(Ntest, 1);
//     for (int i=0; i<Ntest; i++) {
//         double t = c(i,0);
//         for (int j=1; j<c.n_cols; j++) {
//             if (t < c(i,j)) {
//                 t = c(i,j);
//             }
//         }
//         for (int j=0; j<c.n_cols; j++) {
//             if (t == c(i,j)) {
//                 pred(i,0) = j;
//                 break;
//             }
//         }
//     }
//     
//     if (ifshow) {
//         arma::mat err = pred - tl;
//         int ccc = 0;
//         for (int i=0; i<err.n_rows; i++) {
//             if (err(i,0) == 0) {
//                 ccc++;
//             }
//         }
// //         std::cout << std::setprecision(3);                         // 等价于 cout <<setprecision(4) ;
//         std::cout << "err  " << "\t" << "pred  " << "\t" << "tl  " << std::endl;
//         for (int i=0; i<err.n_rows; i++) {
// 			if (abs(err(i,0)) > 1)
// 			{
// 				std::cout << err(i,0) << "\t" << pred(i,0) << "\t" << tl(i,0) << std::endl;
// 			}
//         }
//         std::cout << "err  " << "\t" << "pred  " << "\t" << "tl  " << std::endl<<std::endl;
//         std::cout << "tl min: " << arma::min(tl);
//         std::cout << "tl max: " << arma::max(tl);
// 		arma::mat mm = arma::mean(tl);
// 		std::cout << "original error: "<<arma::mean(arma::abs(tl-mm(0,0)));
//         std::cout << "mean error: "<<arma::mean(arma::abs(err));
//         std::cout << "win ratio: " << 1.0*ccc/err.n_rows <<std::endl;
// 
// 		std::cout << "original cov: "<< arma::cov(tl.t()) << std::endl;
// 		std::cout << "pred cov: "<< arma::cov(err.t()) << std::endl;
//     }
    
    return pred;
}

void SDH::train(std::string fn){
    
    cyktools cyk;
    
//     std::cout << "Generating anchors ..."<< std::endl;
//     anchor.set_size(n_anchors,X.n_cols);
//     for (int i=0; i<n_anchors; i++) {
//        int k = arma::round(cyk.random(0,Ntrain-1));
//         int k = cvRound(cyk.random(0,Ntrain-1));
//         anchor.row(i) = X.row(k);
//     }
//     
//     std::cout << "Generating Phix ..."<< std::endl;
//     Phix.set_size(Ntrain, n_anchors + 1);
//     Phix.fill(0.0); // set the end cols to zero
//     double si = 2.0*sigma*sigma;
//     arma::mat xt = sum(square(X), 1);// 60000 * 1
//     arma::mat yt = sum(square(anchor), 1);//1000 * 1
//     arma::mat pt(Ntrain, n_anchors);
//     for (int i=0; i<Ntrain; i++) {
//         for (int j=0; j<n_anchors; j++) {
//             pt(i, j) = xt(i,0) + yt(j,0);
//         }
//     }
//     pt -= 2.* X * anchor.t();
//     Phix.cols(0,Phix.n_cols-2) = pt;
//     Phix /= si;
//     Phix = arma::exp(Phix);
//     
//     Phix.row(10).print("px");
    
//    for (int i=0; i<Ntrain; i++) {
//        for (int j=0; j<n_anchors; j++) {
//            double t = -norm(X.row(i) - anchor.row(j))/si;
//            Phix(i, j) = exp(t);
//        }
////        Phix(i, n_anchors) = 0;
////        std::cout << i <<std::endl;
//    }
    Phix = X;
    // learn G and F
//    zinit.set_size(Ntrain, nbits);
    zinit.randu(Ntrain, nbits);
    zinit = arma::sign(zinit-0.5);

	Wg = RRC(zinit, label, g_lambda);
	double me_bias = arma::norm(zinit*Wg - label, "fro");
	std::cout <<"### random bias ### = "<<me_bias<<std::endl;
	me_bias = arma::norm(label, "fro");
	std::cout << "Label fro: "<< me_bias <<std::endl;

    if(!calc())
        return ;
    
    std::cout << "Saving data ..."<<std::endl;
    
    save(fn);
    
    std::cout << "Done!"<<std::endl;
    
}

bool SDH::calc(void ){

	arma::mat pm;
	std::cout << "PM ing..." <<std::endl;
	pm = solve(Phix.t()*Phix + f_lambda * arma::eye<arma::mat>(Phix.n_cols, Phix.n_cols), Phix.t());
	cyktools cyk;
    std::cout << "Calculating ... "<<std::endl;
    double tol = 1e-5;
    
//     double delta = 1.0/nu;
    
// 	// ============== FOR CLASSIFICATION !!! ===============
//     // =========== project label to a N * c matrix =========
//     if (label.n_cols == 1) {
//         label = round(label);
//         arma::mat tmp = label;
//         std::cout <<"label minimun: "<< arma::min(label)<<std::endl;
//         std::cout <<"label maxmun:  "<< arma::max(label)<<std::endl;
// //        int ml = arma::min(arma::min(label));
//         double m = arma::max(arma::max(label));
// //        std::cout << "Ntrain:"<<Ntrain<<std::endl;
//         
//         label.set_size(Ntrain, m+1);
// //        std::cout << "r:"<<label.n_rows<<std::endl;
//         label.fill(0.0);
//         for (int i=0; i<Ntrain; i++) {
//             if (tmp(i, 0)<0) {
//                 std::cout << "label less then zero! " <<std::endl;
//                 return false;
//             }
//             label(i, tmp(i, 0)) = 1;
//         }
//     }
// 	// ======================================================

    std::cout << "G-step init."<<std::endl;
//    std::cout << zinit.n_cols <<"; "<< label.n_cols <<std::endl;
    Wg = RRC(zinit, label, g_lambda);
    std::cout << "F-step init."<<std::endl;
//     Wf = RRC(Phix, zinit, f_lambda);
	Wf = pm * zinit;
    
    arma::mat XF, Q;
    for (int i=0; i<maxItr; i++) {
        
        std::cout << "Iteration: "<< i << std::endl;
        XF = Phix * Wf;
        Q = nu * XF + label * Wg.t();
        
//         zinit.fill(0.0);
        for (int j=0; j<32; j++) {
           std::cout << j <<"-";
            arma::mat Z0 = zinit;
            for (int t=0; t<zinit.n_cols; t++) {
                arma::mat Zk(zinit.n_rows, zinit.n_cols-1);
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
                arma::mat Wkk = Wg.row(t);
                arma::mat Wk(Wg.n_rows-1, Wg.n_cols);
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
            if (arma::norm(zinit - Z0, "fro") < 1e-6 * arma::norm(Z0, "fro")) {
               std::cout << "break" << std::endl;
                break;
            }
        }
//         double bias = arma::norm(zinit - Phix*Wf, "fro");
//         std::cout <<"B bias = "<<bias<<std::endl;

//        std::cout << "-#"<<std::endl;
        
        Wg = RRC(zinit, label, g_lambda);
        arma::mat Wf0 = Wf;
		Wf = pm * zinit;
//         Wf = RRC(Phix, zinit, f_lambda); //P
//        std::cout << Wf.n_rows <<std::endl;
        double bias = arma::norm(zinit - Phix*Wf, "fro");
        std::cout <<"P bias = "<<bias<<std::endl;

		double To_bias = arma::norm(arma::sign(Phix * Wf)*Wg - label, "fro");
		std::cout <<"To bias = "<<To_bias<<std::endl;

		double me_bias = arma::norm(zinit*Wg - label, "fro");
		std::cout <<"me bias = "<<me_bias<<std::endl;

		double o_bias = me_bias+nu*bias;
		std::cout << "O bisa = "<<o_bias<<std::endl;


        if (bias < tol * arma::norm(zinit, "fro")) {
            break;
        }
        if (arma::norm(Wf - Wf0, "fro") < tol * arma::norm(Wf0)) {
            break;
        }
    }
    
    return true;
}

arma::mat SDH::RRC(arma::mat td, arma::mat tl, double lambda){
    // faster way;
    arma::mat pm;
    if (td.n_rows > td.n_cols) {
        pm = solve(td.t()*td + lambda * arma::eye<arma::mat>(td.n_cols, td.n_cols), td.t());
    }else{
        pm = (solve((td*td.t() + lambda * arma::eye<arma::mat>(td.n_rows, td.n_rows)).t(), td)).t();
    }
    return pm * tl;
////    std::cout << "lambda: "<<lambda <<std::endl;
////    std::cout << "td:" <<td.n_rows<<", "<<td.n_cols<<std::endl;
//    std::cout << "123" <<std::endl;
//    arma::mat tdt = td * td.t();
//    std::cout << "123" <<std::endl;
//    tdt += lambda * arma::eye<arma::mat>(td.n_rows, td.n_rows);
////    cyktools cyk,
//
//    return arma::solve(tdt, tl);
}

void SDH::save(std::string fn){
//     anchor.save((fn+"_anchor.dat").c_str());
    Wf.save((fn+"_Wf.dat").c_str());
    Wg.save((fn+"_Wg.dat").c_str());
    
//    std::cout << anchor.n_rows <<"#"<< anchor.n_cols <<"#"<< Wf.n_rows <<"#"<< Wf.n_cols <<"#"<< Wg.n_rows <<"#"<< Wg.n_cols <<std::endl;
}

void SDH::load(std::string fn){
//     anchor.load((fn+"_anchor.dat").c_str());
    Wf.load((fn+"_Wf.dat").c_str());
    Wg.load((fn+"_Wg.dat").c_str());
}














