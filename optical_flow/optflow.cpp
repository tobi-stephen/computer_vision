#include <stdio.h>
#include "opencv2/opencv.hpp"

int main(int argc, char **argv)
{
	cv::Mat im1, im1g, im2, im2g, im3;
	std::vector<cv::Point2f> cornersA, cornersB;
	cv::VideoCapture cap(0);

	cv::Mat flow;
	int win_size = 10;
	// int flag = OPTFLOW_USE_INITAL_FLOW;
	
	cap >> im1;
	cv::cvtColor(im1, im1g, cv::COLOR_BGR2GRAY);
	
	cv::Mat mask = cv::Mat::zeros(im1.size(), im1.type());
	while (1){

		cv::goodFeaturesToTrack(
			im1g, 
			cornersA,
			500,
			0.01,
			5, 
			cv::noArray(),
			3,
			false,
			0.04
		);

		cv::cornerSubPix(
			im1g,
			cornersA,
			cv::Size(win_size, win_size),
			cv::Size(-1, -1),
			cv::TermCriteria(
				cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
				20,
				0.03
			)
		);

		cap >> im2;
		cv::cvtColor(im2, im2g, cv::COLOR_BGR2GRAY);
		
		std::vector<uchar> status;
		cv::calcOpticalFlowPyrLK(
			im1g,
			im2g,
			cornersA,
			cornersB,
			status,
			cv::noArray(),
			cv::Size(win_size*2+1, win_size*2+1),
			5,
			cv::TermCriteria(
				cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
				20,
				0.3
			)
		);
		
		for(int i = 0; i < (int)cornersA.size(); i++){
			if (status[i]){
				cv::line(mask, cornersB[i], cornersA[i], cv::Scalar(0,255,0), 2);
				cv::circle(im2, cornersB[i], 5, cv::Scalar(0, 255, 0), -1);
			}
		}
		cv::add(im2, mask, im2);
		// cv::calcOpticalFlowFarneback(
		// 	im1g,
		// 	im2g,
		// 	flow,
		// 	0.2,
		// 	4,
		// 	13,
		// 	10,
		// 	5,
		// 	1.1,
		// 	0);

		// for (int y = 0; y < im2.rows; y+=40){
		// 	for (int x = 0; x < im2.cols; x+=40){
		// 		int p = flow.at<cv::Vec2f>(y, x)[0];
		// 		int q = flow.at<cv::Vec2f>(y, x)[1];
		// 		cv::line(im3, cv::Point(y, x), cv::Point(p, q), cv::Scalar(0, 255, 0), 2);
		// 	}
		// }
		im1g = im2g.clone();
		//if (flow.empty()) flag = CV_OPTFLOW_FARNEBACK_GAUSSIAN;

		
		cv::imshow("ims", im2);
		if (cv::waitKey(33) == 27) break;
	}

	return 0;
}