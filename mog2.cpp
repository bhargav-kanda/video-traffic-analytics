#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>

struct Gauss {
	float* mean;
	float variance, weight;
}

class GMM {
	//const double PI = 3.141593;
	int nmixtures, history;
	float threshold;
	cv::Mat background, foreground;
	public :
	GMM() :nmixtures(5),threshold(9),history(100){}
	GMM(int nm, int th, int his) {nmixtures=nm;	threshold=th;	history=his;}
	void bgs(cv::Mat frame, cv::Mat bg) {
		
	}
	void getBgImage(cv::Mat bg) {
		bg=background;
	}
}; 

int main(int argc, char** argv)
{
	// Open the video file
	CvCapture* capture = cvCreateFileCapture( argv[1] );
	//cv::VideoCapture capture(argv[1]);
	// check if video successfully opened
	if (!capture) {
	printf("cannot open file");
	return 0;
	}
	// current video frame
	cv::Mat frame;
	// foreground binary image
	cv::Mat foreground;
	cv::Mat background;
	cv::Mat IBackground=cv::imread("bg.jpg");
	cv::namedWindow("Extracted Foreground1");
	cv::namedWindow("Extracted Foreground2");
	cv::namedWindow("Current background");
	cv::namedWindow("Original frame");
	// The Mixture of Gaussian object
	// used with all default parameters
	GMM gmm(10000, 25, true);
	//mog.setBackgroundImage(IBackground);
	bool stop(false);
	// for all frames in video
	frame=IBackground;
	
	while (!stop) {
		// read next frame if any
		if (!cvQueryFrame( capture ))
		break;
		// update the background
		// and return the foreground
		gmm.bgs(frame,foreground);
		gmm.getBgImage(background);
		cv::imshow("Current background",background);
		// Complement the image
		cv::imshow("Extracted Foreground1",foreground);
		cv::threshold(foreground,foreground,128,255,cv::THRESH_BINARY);
		// show foreground
		cv::imshow("Extracted Foreground2",foreground);
		cv::imshow("Original frame",frame);
		frame = cvQueryFrame( capture );
		// press key to stop
		if (cv::waitKey(10)>=0)
		stop= true;
	}
	cvDestroyAllWindows();
	cvReleaseCapture(&capture);
	imwrite("bg.jpg", background);
}
