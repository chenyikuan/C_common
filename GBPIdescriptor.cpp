#include "GBPIdescriptor.h"
#include "cykTools.h"
//#include <armadillo>


GBPI::GBPI(){
	init(40);
}

GBPI::GBPI(cv::Mat pic, int ds){
	init(ds);
	loadImg(pic);
}

void GBPI::set_gp(std::string gpname, int ds){
	gpn = gpname;
	set_des(ds);
}

void GBPI::set_des(int ds){
	cyktools cyk;
	arma::mat gp_a = cyk.readMat(gpn.c_str());
	gp = cyk.arma2opencv(gp_a);
	for (int i=0; i<Nl; i++)
	{
		gp.ptr<double>(i)[0] = cvRound(gp.ptr<double>(i)[0] * ds);
		gp.ptr<double>(i)[1] = cvRound(gp.ptr<double>(i)[1] * ds);
	}
}

void GBPI::init(int ds){
    cyktools cyk;
	gpn = "../../common/gp_400.mat";
// 	gpn = "E:/VS_WS/common/gp.mat";
    arma::mat gp_a = cyk.readMat(gpn.c_str());

    gp = cyk.arma2opencv(gp_a);

//	matfile2Mat("E:\\Matlab_WS\\testGBPI\\gp.mat", "gp", gp);
	des_size = ds;
	Nl = gp.rows;
//    std::cout << Nl << std::endl;
	for (int i=0; i<Nl; i++)
	{
		gp.ptr<double>(i)[0] = cvRound(gp.ptr<double>(i)[0] * des_size);
		gp.ptr<double>(i)[1] = cvRound(gp.ptr<double>(i)[1] * des_size);
	}
}

void GBPI::loadImg(cv::Mat& pic){

	if (pic.empty())
	{
		std::cout<< "Empty image!" << std::endl;
		return ;
	}
	if (pic.channels()==3)
		cvtColor(pic,img, CV_RGB2GRAY);
	else
		img = pic.clone();

//	for (int sig=1; sig<=32; sig++)
//	{
//		cv::Mat B;
//		cv::GaussianBlur(img, B, cv::Size(2*sig+1,2*sig+1), sig*0.25);
//		imgs.push_back(B);
//	}
	return ;
}

void GBPI::getDes(cv::Point2i pt, arma::mat& des)const{
    
//    des = cv::Mat::zeros(1, Nl/2, CV_64FC1); // 1000/2 = 500
    des.set_size(1, Nl/2);
    
    if (img.empty())
    {
        std::cout<< "Empty image! ------> No descriptors!" << std::endl;
        //		return des;
    }
    
    cv::Mat pts = cv::Mat::zeros(Nl, 1, CV_32FC1);
    for (int i=0; i<Nl; i++)
    {
        cv::Point2i l1(pt.x+gp.ptr<double>(i)[0], pt.y+gp.ptr<double>(i)[1]);
        if ( l1.x>=img.cols || l1.y>=img.rows || l1.x<0 || l1.y<0 )
        {
//            std::cout << "Points range error."<< std::endl;
            continue;
        }
        // 		std::cout << gp.ptr<double>(i)[2] <<std::endl;
        pts.ptr<float>(i)[0] = img.ptr(l1.y)[l1.x];
        // 		pts.ptr<float>(i)[0] = imgs.at(0).ptr(l1.y)[l1.x];
    }
    for (int i=0; i<Nl/2; i++)
    {
        des(0,i) = (pts.ptr<float>(i)[0] - pts.ptr<float>(i+Nl/2)[0]);
    }
    // 	for (int i=0; i<Nl; i++)
    // 	{
    // 		for (int j=0; j<Nl; j++)
    // 		{
    // 			des.ptr<float>(0)[i*Nl+j] = pts.ptr<float>(i)[0] - pts.ptr<float>(j)[0];
    // 		}
    // 	}
    des = arma::normalise(des, 2, 1);
    //	return des;
}

void GBPI::getDes(cv::Point2i pt, cv::Mat& des)const{
    
    des = cv::Mat::zeros(1, Nl/2, CV_64FC1); // 1000/2 = 500
    
    if (img.empty())
    {
        std::cout<< "Empty image! ------> No descriptors!" << std::endl;
        //		return des;
    }
    
    cv::Mat pts = cv::Mat::zeros(Nl, 1, CV_32FC1);
    for (int i=0; i<Nl; i++)
    {
        cv::Point2i l1(pt.x+gp.ptr<double>(i)[0], pt.y+gp.ptr<double>(i)[1]);
        if ( l1.x>=img.cols || l1.y>=img.rows || l1.x<0 || l1.y<0 )
        {
            std::cout << "Points range error."<< std::endl;
            continue;
        }
        // 		std::cout << gp.ptr<double>(i)[2] <<std::endl;
        pts.ptr<float>(i)[0] = img.ptr(l1.y)[l1.x];
        // 		pts.ptr<float>(i)[0] = imgs.at(0).ptr(l1.y)[l1.x];
    }
    for (int i=0; i<Nl/2; i++)
    {
        des.ptr<double>(0)[i] = (pts.ptr<float>(i)[0] - pts.ptr<float>(i+Nl/2)[0]);
    }
    // 	for (int i=0; i<Nl; i++)
    // 	{
    // 		for (int j=0; j<Nl; j++)
    // 		{
    // 			des.ptr<float>(0)[i*Nl+j] = pts.ptr<float>(i)[0] - pts.ptr<float>(j)[0];
    // 		}
    // 	}
    normalize(des, des);
    //	return des;
}
