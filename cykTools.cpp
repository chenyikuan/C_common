#ifdef _WIN32
	#include <conio.h>
#else
	#include <sys/select.h>
	#include <termios.h>
	#include <unistd.h>
	#include <fcntl.h>
	#include <stdio.h>
	// #include <curses.h>
#include <unistd.h>
#endif
#include "cykTools.h"
// #include "armaMex.hpp"
// #include <time.h>
#include <math.h>
//#include "mat.h"
//#include "matrix.h"

using namespace std;
using namespace arma;

bool cykTools::rand_set_ok = false;

//  cykTools::cykTools(){
// // 	cyklog.open("log.txt", ios::out);
//  	//rand_set_ok = false;
//  }
 
 void cykTools::initRand(){
	 // std::cout << "random seed init!!!" << std::endl;
 	srand(unsigned(time(0)));
 	rand_set_ok = true;
 }
 
 double cykTools::random(double start, double end){
     if (!rand_set_ok){
 		initRand();
         rand();    // ignore the first rand() becouse it's allways not random.
     }
 	return start+(end-start)*rand()/(RAND_MAX + 1.0);
 }

arma::mat cykTools::LSR(arma::mat td, arma::mat tl, double lambda, const solve_opts::opts& opts){
	 // ====== 3nd edition ===== //
	arma::mat pm, tdx;
	// std::cout << "  --- 1st stage ---"<<std::endl;
	if (td.n_rows > td.n_cols) {
		tdx = td.t()*td;
		// std::cout << "    --- xixi ---"<<std::endl;
		for (uword i=0;i<td.n_cols;i++)
			tdx(i,i) += lambda;
	pm = solve(tdx, td.t(), opts);
	}else{
		tdx = td*td.t();
		// std::cout << "    --- xixi ---"<<std::endl;
		for (uword i = 0; i<td.n_rows; i++)
			tdx(i,i) += lambda;
	pm = (solve(tdx, td, opts)).t();
	}
	// std::cout << "  --- 2nd stage ---"<<std::endl;
	return pm * tl;
}

arma::fmat cykTools::LSRf(arma::fmat td, arma::fmat tl, float lambda, const solve_opts::opts& opts){
	return conv_to<fmat>::from(LSR(conv_to<mat>::from(td), conv_to<mat>::from(tl), lambda, opts));
 	/*//std::cout << "LSRf not working well, please convert to double format and use LSR instead." << std::endl;
    arma::fmat pm, tdx;
    std::cout << "  --- 1st stage ---"<<std::endl;
    if (td.n_rows > td.n_cols) {
        tdx = td.t()*td;
        // std::cout << "    --- xixi ---"<<std::endl;
		for (uword i = 0; i<td.n_cols; i++)
        tdx(i,i) += lambda;
        pm = solve<fmat>(tdx, td.t(), arma::solve_opts::fast);
    }else{
        tdx = td*td.t();
        // std::cout << "    --- xixi ---"<<std::endl;
		for (uword i = 0; i<td.n_rows; i++)
            tdx(i,i) += lambda;
		pm = solve<fmat>(tdx, td, arma::solve_opts::fast);
		pm = pm.t();
    }
    std::cout << "  --- 2nd stage ---"<<std::endl;
	return pm * tl;*/
}

// arma::mat cykTools::readMat(const char* fn){
//     // mat file should be double scale!!!
//     mat fromFile = armaReadMatFromFile(fn);
//     return fromFile;
// }

// void cykTools::writeMat(const char *filename, arma::mat &armaMatrix, const char *name){
//     armaWriteMatToFile(filename, armaMatrix, name);
// }

cv::Mat cykTools::mat2opencv(arma::mat a){
	cv::Mat b(a.n_rows, a.n_cols, CV_64FC1);
	for (uword i = 0; i < a.n_rows; i++) {
		for (uword j = 0; j < a.n_cols; j++) {
			b.ptr<double>(i)[j] = a(i, j);
		}
	}
	return  b;
}

cv::Mat cykTools::fmat2opencv(arma::fmat a){
	cv::Mat b(a.n_rows, a.n_cols, CV_32FC1);
	for (uword i = 0; i < a.n_rows; i++) {
		for (uword j = 0; j < a.n_cols; j++) {
			b.ptr<float>(i)[j] = a(i, j);
		}
	}
	return  b;
}

arma::mat cykTools::opencv2arma(cv::Mat a){
    arma::mat b(a.rows, a.cols);
    for (int i=0; i<a.rows; i++) {
        for (int j=0; j<a.cols; j++) {
            b(i,j) = a.ptr<double>(i)[j];
        }
    }
    return b;
}

// cv::Mat cykTools::readMat2opencv(const char* fn){
//     return arma2opencv(readMat(fn));
// }

// void cykTools::addSample(cv::Mat& a, cv::Mat& b){
//     if (a.empty())
//         a = b.clone();
//     else
//         a.push_back(b);
//     return;
// }

arma::fmat cykTools::filter2(arma::fmat b, arma::fmat x)
{
	fmat stencil = fliplr(flipud(b)); // why?
	fmat U;
	fvec s;
	fmat V;
	svd_econ(U, s, V, stencil);
// 	U.print("U");
// 	s.print("s");
// 	V.print("V");
// 	cout << s(0) << ", " << s(1) << endl;
	fmat hcol = U.col(0) * sqrtf(s(0));
	fmat hrow = V.col(0) * sqrtf(s(0));
// 	hrow.print("hcol");
	fmat y = x;
	y = conv2(y, hrow, "same");
	y = conv2(y, hcol.t(), "same");
	return y;
}

void cykTools::get_RT(fmat& A, fmat& B, fmat& R, fmat& T)
{
	fmat centroid_A, centroid_B;
	centroid_A = mean(A);
	centroid_B = mean(B);
	fmat A_centroid = A;
	fmat B_centroid = B;
	for (uword i = 0; i < A.n_rows; ++i)
	{
		A_centroid.row(i) = A.row(i) - centroid_A;
		B_centroid.row(i) = B.row(i) - centroid_B;
	}
	fmat H = A_centroid.t() * B_centroid;
	//mat U, V;
	//vec s;
	//svd(U, s, V, conv_to<mat>::from(H));
	fmat U, V;
	fvec s;
	svd<fmat>(U, s, V, H);
	//R = conv_to<fmat>::from(V*U.t());
	R = V*U.t();
	T = -R*centroid_A.t() + centroid_B.t();
}

void cykTools::get_R(arma::fmat& r, float a, float b, float c)
{
	if (r.n_cols<3 || r.n_rows<3)
	{
		r.zeros(3, 3);
	}
	float sx = sin(c);
	float cx = cos(c);
	float sy = sin(b);
	float cy = cos(b);
	float sz = sin(a);
	float cz = cos(a);

	r(0, 0) = cy*cz;
	r(0, 1) = sx*sy*cz - cx*sz;
	r(0, 2) = cx*sy*cz + sx*sz;
	r(1, 0) = cy*sz;
	r(1, 1) = sx*sy*sz + cx*cz;
	r(1, 2) = cx*sy*sz - sx*cz;
	r(2, 0) = -sy;
	r(2, 1) = sx*cy;
	r(2, 2) = cx*cy;
}

void cykTools::get_euler_abc(arma::fmat r, float& a, float& b, float& c, const string code)
{
	a = atan2f(r(1, 0), r(0, 0));
	float cz = cos(a);
	float sz = sin(a);
	b = atan2f(-r(2, 0), r(0, 0)*cz + r(1, 0)*sz);
	c = atan2f(r(0, 2)*sz - r(1, 2)*cz, r(1, 1)*cz - r(0, 1)*sz);
	if (code != "rad")
	{
		a = this->rad2deg(a);
		b = this->rad2deg(b);
		c = this->rad2deg(c);
	}
}

// change from eular angle space to quaternion space
void cykTools::eular2quaternion_rad(float theta1, float theta2, float theta3, arma::fmat& q)
{
	q = zeros<fmat>(1, 4);
	float th1 = theta1 / 2;
	float th2 = theta2 / 2;
	float th3 = theta3 / 2;
	float s1 = sin(th1);
	float c1 = cos(th1);
	float s2 = sin(th2);
	float c2 = cos(th2);
	float s3 = sin(th3);
	float c3 = cos(th3);

	q(0, 0) = c1*c2*c3 + s1*s2*s3;
	q(0, 1) = s1*c2*c3 - c1*s2*s3;
	q(0, 2) = c1*s2*c3 + s1*c2*s3;
	q(0, 3) = c1*c2*s3 - s1*s2*c3;
}

// change from eular angle space to quaternion space
void cykTools::quaternion2eular_rad(arma::fmat q, float& theta1, float& theta2, float& theta3)
{
	float w = q(0, 0);
	float x = q(0, 1);
	float y = q(0, 2);
	float z = q(0, 3);

	theta1 = atan2f(2 * (w*x + y*z), 1 - 2 * (x*x + y*y));
	theta2 = asinf(2 * (w*y - z*x));
	theta3 = atan2f(2 * (w*z + y*x), 1 - 2 * (z*z + y*y));
}

bool cykTools::average_quadernion(arma::fmat q1, arma::fmat q2, arma::fmat& avr_quad, float w1, float w2)
{
	fmat dot_product_q1q2 = q1*q2.t();
	float dot_product_q1q2_val = dot_product_q1q2(0, 0);
	float z = sqrtf((w1 - w2)*(w1 - w2) + 4 * w1*w2*dot_product_q1q2_val*dot_product_q1q2_val);
	if (dot_product_q1q2_val == 0 && w1 == w2)
	{
		cout << "[ WARNING ]: Bad condition of two quaternions! " << endl;
		return false;
	}
	avr_quad = sqrtf(w1*(w1 - w2 + z) / z / (w1 + w2 + z)) * q1 +
		(dot_product_q1q2_val > 0 ? 1 : -1) *
		sqrtf(w2*(w2 - w1 + z) / z / (w1 + w2 + z)) * q2;
	return true;
}

void cykTools::normalize_quad(arma::fmat& quad)
{
	float n_q = sqrtf(norm<fmat>(quad, "fro"));
	n_q = n_q > 0 ? n_q : 1;
	quad /= n_q;
}
//float cykTools::rad2deg(float th)
//{
//	return th * 57.29578f;
//}
//
//float cykTools::deg2rad(float th)
//{
//	return th * 0.0174533f;
//}

void cykTools::plot_arma(std::string win_name, arma::fmat data, int key_t)
{
	int empty_width = 10;
	float max_height = 600;
	float min_height = 100;
	float max_width = 1000;

	float value_min = data.min();
	float value_max = data.max();
	cv::Mat pano;

	float plot_height = value_max - value_min;
	float plot_width = (float)data.n_cols;
	float plot_height_scale = 1;
	float plot_width_scale = 1;
	plot_height = plot_height > 0 ? plot_height : 1;

	if (plot_height > max_height)
	{
		// cout << 1 << endl;
		plot_height_scale = max_height / plot_height;
		plot_height = max_height;
	}
	else if (plot_height < min_height)
	{
		// cout << 2 << endl;
		plot_height_scale = min_height / plot_height;
		plot_height = min_height;
	}
	if (plot_width > max_width)
	{
		plot_width_scale = max_width / plot_width;
		plot_width = max_width;
	}

	pano = cv::Mat(plot_height+2*empty_width, plot_width, CV_8UC3, cv::Scalar::all(0));
	for (uword j = 0; j < data.n_rows; ++j)
	{
		// cout << j << endl;
		//int b = this->random(0, 255);
		//int g = this->random(0, 255);
		//int r = this->random(0, 255);
		uchar b, g, r;
		if (j == 0)
		{
			b = g = r = 20;
		}
		if (j==1)
		{
			r = 255;
			b = g = 0;
		}
		else if (j==2)
		{
			g = 255;
			r = b = 0;
		}
		else if (j==3)
		{
			b = 255;
			r = g = 0;
		}
		else
		{
			b = g = r = 255;
		}
		for (uword i = 0; i < data.n_cols; ++i)
		{
			int col = floor(i*plot_width_scale)*3;
			int row = plot_height + empty_width - plot_height_scale * (data(j, i) - value_min);
			// cout << plot_height_scale << ", " << data(j, i) << endl;
			// cout << row << ", " << col << endl;
			pano.ptr(row)[col + 0] = b;
			pano.ptr(row)[col + 1] = g;
			pano.ptr(row)[col + 2] = r;
		}
	}
	cv::imshow(win_name, pano);
	cv::waitKey(key_t);
	// cv::Mat pano()
}

void cykTools::debugging(uword idx)
{
	cout << "[ cyk_debug ] debug count: " << idx << endl;
	this->pause();
}

void cykTools::tic()
{
#ifndef _WIN32
	gettimeofday(&t_start, 0);
#else
	t_start = clock();
#endif
}

void cykTools::toc()
{
#ifndef _WIN32
	timeval c_time;
	gettimeofday(&c_time, 0);
	c_time.tv_sec -= t_start.tv_sec;
	c_time.tv_usec -= t_start.tv_usec;
	long long t_passed = 0;
	t_passed = 1000000LL * c_time.tv_sec + c_time.tv_usec;
	t_passed /= 1000;
	printf("Time used: %lldms\n", t_passed);
#else
	double t_end = clock();
	printf("Time used: %f s\n", (t_end - t_start)/CLOCKS_PER_SEC );
#endif
}

int cykTools::pause()
{
//#ifdef _WIN32
//	system("pause");
//#else
	cout << "Paused ... \t<press any key to continue, ESC to exit.>" << endl;
	while(!_kbhit());
#ifdef _WIN32
	int c = _getch();
	if (c == 224)
		c = _getch();
#else
	int c = getchar();
	// cout << "key: " << c  << "." << endl;
#endif
	if (c == 27)
	{
		exit(0);
	}
	return c;
}

#ifndef _WIN32
int cykTools::_kbhit(void)
{
	// // ---- 阻塞式：
	// struct timeval tv;
	// fd_set read_fd;

	// tv.tv_sec = 0;
	// tv.tv_usec = 0;
	// FD_ZERO(&read_fd);
	// FD_SET(0, &read_fd);

	// if (select(1, &read_fd, NULL, NULL, &tv) == -1)
	// 	return 0;
	// if (FD_ISSET(0, &read_fd))
	// 	return 1;

	// ---- 非阻塞式：
	struct termios oldt, newt;
	int ch, oldf;
	tcgetattr(STDIN_FILENO, &oldt);
	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);
	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
	ch = getchar();
	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);
	if (ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}

	return 0;
}
#endif











