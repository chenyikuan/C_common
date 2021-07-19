//


#ifndef SDH_H_
#define SDH_H_

#include <armadillo>

class SDH {
    
    
    int Ntrain;
    int leng;
    
    int maxItr;
    double g_lambda;
    double f_lambda;
    double nu;
    int nbits;

    arma::mat X;
    arma::mat label;
    arma::mat Phix;
    arma::mat zinit;

public:
    double n_anchors;
    double sigma;
    
    arma::mat anchor;
    arma::mat Wg;
    arma::mat Wf;
    
public:
    SDH();
//    SDH(int nb = 32, int itr = 15);
    arma::mat gen_not_B(arma::mat td);
    void train(std::string fn);
    bool calc( void );
    arma::mat RRC(arma::mat td, arma::mat tl, double lambda);
    
    void save(std::string fn);
    void load(std::string fn);
    void init(arma::mat td, arma::mat tl, int nb=32, int ni=15, double gl=1, double fl=100, double nu_=0.01, int na = 1000);
    arma::mat predict(arma::mat td, arma::mat tl = arma::mat(), bool ifshow = false);
};


#endif
