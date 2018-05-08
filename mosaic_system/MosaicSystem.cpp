#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;

const double pi = 3.1415926535897;
bool USE_CAMERA = false;
int thresh = 500;
int max_thresh = 1500;
void rgbtohsi(cv::Mat, cv::Mat&);
void featureDetection(cv::Mat& hsi, std::vector<std::vector<cv::KeyPoint>> &features);
void featureTrack(int, void*);
void featureDescription(std::vector<std::vector<cv::KeyPoint>> &features, std::vector<cv::Mat> &images, std::vector<cv::Mat> &descriptors);

cv::Mat src;
cv::Mat hsi;
std::vector<std::vector<cv::KeyPoint>> features;
const char* kp = "kp";

int main(int argc, char** argv)
{
	if (argc < 2) {
		printf("not enough args");
		return -1;
	}

	std::vector<cv::Mat> ims, imsk(2);
	ims.push_back((cv::Mat)cv::imread(argv[1], 0));
	ims.push_back((cv::Mat)cv::imread(argv[2], 0));

	cv::resize(ims[0], ims[0], cv::Size(400, 400));
	cv::resize(ims[1], ims[1], cv::Size(400, 400));
	
	std::vector<std::vector<cv::KeyPoint>> f(2);
	std::vector<cv::Mat> d(2);

	cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
	sift->detect(ims, f);
	sift->compute(ims, f, d);

	cv::drawKeypoints(ims[0], f[0], imsk[0], Scalar::all(-1));
	cv::drawKeypoints(ims[1], f[1], imsk[1], Scalar::all(-1));

	std::vector<cv::DMatch> dm, gm;
	cv::Ptr<cv::DescriptorMatcher> bfm = cv::DescriptorMatcher::create("FlannBased");
	bfm->match(d[1], d[0], dm);
	
	double max_dist = 0, min_dist = 100;
	for( int i = 0; i < d[1].rows; i++ ){ 
		double dist = dm[i].distance;
	    if( dist < min_dist ) min_dist = dist;
	    if( dist > max_dist ) max_dist = dist;
	}

  	for( int i = 0; i < d[1].rows; i++ ){ 
  		if( dm[i].distance <= max(2*min_dist, 0.02) ){ 
  			gm.push_back( dm[i]);
  		}
  	}

  	std::vector<cv::Point2f> sP, dP;
  	for (int i = 0; i < gm.size(); i++){
  		int q = gm[i].queryIdx;
  		int t = gm[i].trainIdx;
  		sP.push_back(f[1][q].pt);
  		dP.push_back(f[0][t].pt);
  	}

  	cv::Mat res;
	cv::drawMatches(ims[1], f[1], ims[0], f[0], gm, res, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), 2);

  	cv::Mat dst;
  	cv::Mat H = cv::findHomography(sP, dP, cv::RANSAC);
  	cv::warpPerspective(ims[1], dst, H, cv::Size(800, 400));

  	cv::Mat tmp = dst(cv::Rect(cv::Point(0, 0), cv::Point(400, 400)));
  	ims[0].copyTo(tmp);

  	printf("-- Max dist : %f \n", max_dist );
  	printf("-- Min dist : %f \n", min_dist );
  	printf("-- No of matches: %lu \n", gm.size());
	
	cv::imshow("dst", dst);
	cv::imshow("gm", res);
	cv::imshow("1", ims[0]);
	cv::imshow("2", ims[1]);
	cv::waitKey(0);
	return 0;

}
/*int main(int argc, char** argv)
{
	if (argc < 2){
		// printf("usage: MosaicSystem.out <Image_Path>\n");
		printf("loading camera...\n");
		USE_CAMERA = true;
        // return -1;
	}

	cv::namedWindow(kp, WINDOW_AUTOSIZE);
	cv::createTrackbar("Corners: ", kp, &thresh, max_thresh, featureTrack);

	if (!USE_CAMERA){
		src = cv::imread( argv[1], cv::IMREAD_COLOR );

		if ( !src.data ){
	        printf("No image data \n");
	        return -1;
	    }

	    rgbtohsi(src, hsi);
	    featureDetection(hsi, features);
	    // featureTrack(0, 0);

		
    	cv::waitKey(0);
	}else{
		cv::VideoCapture cap(0);
		while (true){
			if (!cap.isOpened()){
				printf("cannot locate camera\n");
				return -1;
			}

			cap >> src;
			if (src.empty()) return -1;

			rgbtohsi(src, hsi);
			featureDetection(hsi, features);
			//featureTrack(0, 0);

			cv::Mat temp;
			cv::drawKeypoints(src, features[1], temp);
			cv::imshow(kp, temp);
			if (waitKey(33) >= 0) break;

			}
		
	}
    
    return 0;
}*/

void rgbtohsi(cv::Mat src, cv::Mat& hsi)
{
	hsi.create(src.rows, src.cols, src.type());
    float r, b, g, h, s, in;

    for (int i = 0; i < src.rows; i++){
    	for (int j = 0; j < src.cols; j++){
    		b = src.at<cv::Vec3b>(i, j)[0];
    		g = src.at<cv::Vec3b>(i, j)[1];
    		r = src.at<cv::Vec3b>(i, j)[2];

    		in = (b + g + r) / 3;

    		int min_val = std::min(std::min(b, g), r);
    		s = 1 - (min_val/in);

    		r /= in; b /=in; g /=in;
    		h = (0.5 * ((r - g) + (r - b))) / std::sqrt(std::pow(r - g, 2) + ((r-b) * (g - b)));
    		h = std::acos(h);

    		if (b > g){
    			h = (2 * pi) - h;
    		}

    		hsi.at<cv::Vec3b>(i, j)[0] = (h * 180) / pi;
    		hsi.at<cv::Vec3b>(i, j)[1] = s * 100;
    		hsi.at<cv::Vec3b>(i, j)[2] = in;
    	}
    }

}

void featureDetection(cv::Mat& hsi, std::vector<std::vector<cv::KeyPoint>> &features)
{
	std::vector<cv::Mat> cmp;
    cv::split(hsi, cmp);

    //features = new std::vector<cv::KeyPoint>[3];

    std::vector<cv::Mat> imgcopy(3);
	features.clear();

    for (int k = 0; k < 3; k++){
    	std::vector<cv::Point2f> corners;
    	std::vector<cv::KeyPoint> kp;
	    cv::goodFeaturesToTrack(cmp.at(k), corners, thresh, .01, 10, cv::Mat(), 2, false, 0.04);
	    cv::KeyPoint::convert(corners, kp);
	    features.push_back(kp);

	   	std::cout << "Corners detected: " << corners.size() << std::endl;	
	}

	cv::Mat temp;
	cv::drawKeypoints(src, features[1], temp);
	cv::imshow(kp, temp);
}

void featureDescription(std::vector<std::vector<cv::KeyPoint>> &features, std::vector<cv::Mat> &images, std::vector<cv::Mat> &descriptors)
{
	cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
	sift->compute(images, features, descriptors);
}

void featureTrack(int, void*)
{
	printf("%d\n", thresh);
	featureDetection(hsi, features);

}