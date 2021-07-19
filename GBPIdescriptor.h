// #ifndef GBPI_DESRCIPTOR_
// #define GBPI_DESRCIPTOR_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <armadillo>


class GBPI{
public:

	GBPI();
	GBPI(cv::Mat pic, int ds = 60);//, std::string gpname = "E:/VS_WS/common/gp.mat"
	void set_gp(std::string gpname, int ds = 60);
	void loadImg(cv::Mat& pic);
    void getDes(cv::Point2i pt, arma::mat& des)const;
    void getDes(cv::Point2i pt, cv::Mat& des)const;
    void set_des(int ds);

    int Nl;
	std::string gpn;
    
private:
	cv::Mat img;
//	cv::Mat des;
//	std::vector<cv::Mat> imgs;

	int des_size;
	cv::Mat gp;

	void init(int ds);

};


/*#endif*/