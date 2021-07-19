#ifndef CYK_TOOLS_
#define CYK_TOOLS_

#include <iostream>
#include <string>
#include <fstream>
#ifndef _WIN32
#include <sys/time.h>
#else
#include <time.h>
#endif
#include <opencv2/opencv.hpp>
#include <armadillo>
#include "cyk_singleton.h"

#define CYK_NUM_PI 3.1415926

#define CYK_TOOLS (*(cykTools::get_instance()))
#define cout_arma_size(arma_mat){\
	std::cout << arma_mat.n_rows << ", " << arma_mat.n_cols << std::endl;\
}
#define cout_cube_size(arma_mat){\
	std::cout << arma_mat.n_rows << ", " << arma_mat.n_cols << ", " << arma_mat.n_slices << std::endl;\
}

class cykTools : public cyk_singleton<cykTools>
{
//  public:
// 	 std::ofstream cyklog;
// 

public:
    int test_number;

    void initRand(void);
    double random(double start, double end);

    // zero colume not added in the end of x!
    // arma::vec LSR(arma::mat x, arma::mat y);

    // ========== LSR for X*A=Y get A ============
	arma::mat LSR(arma::mat x, arma::mat y, double lamda, const arma::solve_opts::opts& opts = arma::solve_opts::fast);
	arma::fmat LSRf(arma::fmat x, arma::fmat y, float lamda, const arma::solve_opts::opts& opts = arma::solve_opts::fast);

    // Y = FILTER2(B, X) filters the data in X with the 2 - D FIR
    //	filter in the matrix B.The result, Y, is computed
    //	using 2 - D correlation and is the same size as X.
    // NOW: simply used for simple filter like medianblur
    arma::fmat filter2(arma::fmat b, arma::fmat x);

    // .mat file should be double scale!!!
    // read matlab .mat file
    // arma::mat readMat(const char* fn);
    // void writeMat(const char *filename, arma::mat &armaMatrix, const char *name);

    // convert arma::mat to cv::Mat, CV_64F is default format
	cv::Mat mat2opencv(arma::mat a);
	cv::Mat fmat2opencv(arma::fmat a);

    // convert cv::Mat to arma::mat, double scale
    arma::mat opencv2arma(cv::Mat a);

    // read .mat file to cv::Mat
    // cv::Mat readMat2opencv(const char* fn);

    // Mat.pushback
    // void addSample(cv::Mat& a, cv::Mat& b);
    // 	void matfile2Mat(const char* matfilename, const char* matfieldname, cv::Mat& dst, int type = CV_64F);
    	

    // Plot arma::mat in opencv window:
    // Demo:
    // cykTools cyk;
    // int data_lenght = 800;
    // int draw_height = 500;
    // int draw_width = data_lenght;
    // mat v1, v2;
    // v1 = linspace(1, 100, data_lenght).t() + 50 * sin(linspace(1, 10, data_lenght).t()*datum::pi);
    // v2 = linspace(5, 50, data_lenght).t();
    // cyk.plot_arma("v1", join_cols(v1, v2), 0);
    void plot_arma(std::string win_name, arma::fmat data, int key_t = 1);

	// get R&T, transform A to B: A*R.t()+T.t() = B
	// point is a row vec
	void get_RT(arma::fmat& A, arma::fmat& B, arma::fmat& R, arma::fmat& T);
	// get R
	// abc: ZYX
	void get_R(arma::fmat& r, float a, float b, float c);
	// get Eulerian angle from Rotation matrix
	// abc: ZYX
	void get_euler_abc(arma::fmat r, float& a, float& b, float& c, const std::string code = "rad");
	// change from eular angle space to quaternion space
	void eular2quaternion_rad(float theta1, float theta2, float theta3, arma::fmat& q);
	// change from eular angle space to quaternion space
	void quaternion2eular_rad(arma::fmat q, float& theta1, float& theta2, float& theta3);
	// averaging two quaternions by weight
	bool average_quadernion(arma::fmat q1, arma::fmat q2, arma::fmat& avr_quad, float w1 = 0.5f, float w2 = 0.5f);
	// make quaternion normalised
	void normalize_quad(arma::fmat& quad);

	template<typename T> T rad2deg(T th){
		return th * 57.29578f;
	}
	template<typename T> T deg2rad(T th){
		return th * 0.0174533f;
	}

	void debugging(arma::uword idx);

    void tic();
    void toc();

    int pause();
#ifndef _WIN32
    int _kbhit(void);
#endif

private:
 	static bool rand_set_ok;
#ifndef _WIN32
    struct timeval t_start;
#else
	double t_start;
#endif

    cykTools(){};
    virtual ~cykTools(){};
friend class cyk_singleton<cykTools>;
};


#endif