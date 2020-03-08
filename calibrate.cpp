#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include "opencv2/core/opengl_interop.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;
using namespace cv;
bool pauseVideo(false);
bool stop(false);
int pcount;
vector<vector<Point3f> > object_points;
vector<vector<Point2f> > image_points;
vector<Point2f> corners;
vector<Point3f> object;
Point current;
bool input=true;
cv::Mat tempMat, tempMat2;
Mat rotationMatrix;
Mat intrinsic = Mat(3, 3, CV_32FC1);
Mat distCoeffs, tvec;
vector<Mat> rvecs;
vector<Mat> tvecs;
double s;
static void onMouse1( int event, int x, int y, int, void* parameter)
{
	if(input) {
		Mat* image = (Mat*) parameter;
		switch( event ){
			case CV_EVENT_LBUTTONDOWN: {
				//printf("I am here");
				printf("%d %d\n",x,y);
				current = Point(x,y);
				corners.push_back(current);
				pcount++;
				//current = Point(x,y);
				//points = &point;
				circle(*image, current, 0, Scalar( 0, 0, 0 ), 4, 8, 0);
				imshow("Original frame",*image);
			break;	}
		}
	}
}

static void onMouse2( int event, int x, int y, int, void* parameter)
{
	Mat* image = (Mat*) parameter;
	switch( event ){
		case CV_EVENT_LBUTTONDOWN: {
			//printf("I am here");
			printf("%d %d\n",x,y);
			current = Point(x,y);
			cv::Mat uvPoint = cv::Mat::ones(3,1,cv::DataType<double>::type); //u,v,1
			uvPoint.at<double>(0,0) = x; //got this point using mouse callback
			uvPoint.at<double>(1,0) = y;
			tempMat = rotationMatrix.inv() * intrinsic.inv() * uvPoint;
			tempMat2 = rotationMatrix.inv() * tvecs[0];
			s = 0 + tempMat2.at<double>(2,0); //0 represents the height Zconst
			s /= tempMat.at<double>(2,0);
			cout << "s : " << s << endl;
			std::cout << "P = " << rotationMatrix.inv() * (s * intrinsic.inv() * uvPoint - tvecs[0]) << std::endl;
		break;	}
	}
}

int key;
int main(int argc, char** argv) {
	Mat frame=imread(argv[1], CV_LOAD_IMAGE_COLOR);
    double error;
	// object.push_back(Point3f(0.0f,0.0f,0.0f));
	// object.push_back(Point3f(0.0f,2.5,0.0f));
	object.push_back(Point3f(0.500,2.500,0.0f));
	object.push_back(Point3f(0.500,0.0f,0.0f));
	object.push_back(Point3f(16.000,0.0f,0.0f));
	object.push_back(Point3f(16.000,2.500,0.0f));
	// object.push_back(Point3f(16.500,2.500,0.0f));
	// object.push_back(Point3f(16.500,0.0f,0.0f));
	object_points.push_back(object);
	//object_points.push_back(object);
	imshow("Original frame",frame);
	setMouseCallback( "Original frame", onMouse1, &frame );
	while(input) {
		key=waitKey(0);
		input=false;	
	}
	image_points.push_back(corners);
	cout << " " << image_points.size() << endl;
	cout << " " << image_points[0].size() << endl;
	error=calibrateCamera(object_points,image_points,frame.size(),intrinsic, distCoeffs, rvecs, tvecs);
	//Mat H = getPerspectiveTransform(corners, object);
	// cout << "Homography = "<< endl << " "  << H << endl << endl;
	// cout << "corners = "<< endl << " "  << corners << endl << endl;
	// cout << "object = "<< endl << " "  << object << endl << endl;
	cout << "intrinsic = "<< endl << " "  << intrinsic << endl << endl;
	cout << "rvecs = "<< endl << " "  << rvecs[0] << endl << endl;
	cout << "tvecs = "<< endl << " "  << tvecs[0] << endl << endl;
	cout << "distortion Coeffs = "<< endl << " "  << distCoeffs << endl << endl;
	printf("%e\n", error);
	Rodrigues(rvecs[0], rotationMatrix);
	cout << "Rot Matrix = "<< endl << " "  << rotationMatrix << endl << endl;
	setMouseCallback( "Original frame", onMouse2, &frame );
	key=waitKey(0);
	FileStorage fs("test.xml", FileStorage::WRITE);
	Mat imageUndistorted;
    fs << "frameCount" << 25;
    fs << "intrinsic" << intrinsic;
    fs << "Rotation" << rvecs[0];
    fs << "translation" << tvecs[0];
    fs << "distCoeffs" << distCoeffs;
    fs.release();
    printf("I am here\n");
    waitKey(0);
	
	cvDestroyAllWindows();
}