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
static void onMouse( int event, int x, int y, int, void* parameter)
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
				current = Point(x,y);
				//points = &point;
				circle(*image, current, 0, Scalar( 0, 0, 0 ), 4, 8, 0);
				imshow("Original frame",*image);
			break;	}
		}
	}
}


int main(int argc, char** argv) {
	Mat frame=imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat intrinsic = Mat(3, 3, CV_32FC1);
    Mat distCoeffs;
    vector<Mat> rvecs;
    vector<Mat> tvecs;
    double error;
	object.push_back(Point3f(0.0f,0.0f,0.0f));
	object.push_back(Point3f(0.0f,2.5,0.0f));
	object.push_back(Point3f(0.5,2.5,0.0f));
	object.push_back(Point3f(0.5,0.0f,0.0f));
	object.push_back(Point3f(16.0,0.0f,0.0f));
	object.push_back(Point3f(16.0,2.5,0.0f));
	object.push_back(Point3f(16.5,2.5,0.0f));
	object.push_back(Point3f(16.5,0.0f,0.0f));
	object_points.push_back(object);
	imshow("Original frame",frame);
	setMouseCallback( "Original frame", onMouse, &frame );
	while(input) {
		waitKey(0);
		input=false;
	}
	cout << " " << image_points.size() << endl;
	image_points.push_back(corners);
	Mat H = findHomography(object_points, image_points, CV_RANSAC);
	cout << "Homography = "<< endl << " "  << H << endl << endl;
	// error=calibrateCamera(object_points,image_points,frame.size(),intrinsic, distCoeffs, rvecs, tvecs);
	// cout << "intrinsic = "<< endl << " "  << intrinsic << endl << endl;
	// cout << "rvecs = "<< endl << " "  << rvecs[0] << endl << endl;
	// cout << "tvecs = "<< endl << " "  << tvecs[0] << endl << endl;
	// cout << "distortion Coeffs = "<< endl << " "  << distCoeffs << endl << endl;
	//printf("%e\n", error);
	cvDestroyAllWindows();
}