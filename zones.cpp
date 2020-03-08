#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include "opencv2/core/opengl_interop.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;
using namespace cv;

bool track=false;
Point current;
Mat duplicate;

const Point** r;
Point* p;
int* PinR;
int input=1;
int pcount=0;					//point count
int rcount=0;					// region count
int i,j,nop;
CvMemStorage* pstorage = cvCreateMemStorage(0);
CvSeq* pseq = cvCreateSeq( CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), pstorage);
CvMemStorage* rstorage = cvCreateMemStorage(0);
CvSeq* rseq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint*), rstorage);
CvMemStorage* nstorage = cvCreateMemStorage(0);
CvSeq* nseq = cvCreateSeq( CV_32SC1, sizeof(CvSeq), sizeof(int), nstorage);
CvPoint **tempp;
CvPoint *t;
static void onMouse( int event, int x, int y, int, void* parameter)
{
	if(input) {
		Mat* image = (Mat*) parameter;
		switch( event ){
			case CV_EVENT_LBUTTONDOWN: {
				//printf("I am here");
				printf("%d %d\n",x,y);
				current = Point(x,y);
				pcount++;
				cvSeqPush( pseq, &current );
				*image = duplicate.clone();
				track=false;
				current = Point(x,y);
				//points = &point;
				circle(*image, current, 0, Scalar( 0, 0, 0 ), 4, 8, 0);
				imshow("Original frame",*image);
			break;	}
		
			case CV_EVENT_LBUTTONUP:
				track=true;
				printf("%d %d\n",x,y);
			break;

			case CV_EVENT_MOUSEMOVE: {
			if(track) {
				duplicate = (*image).clone();
				//printf("%d %d %d %d\n",x,y,current.x, current.y);
				Point temp = Point(x,y);
				line(duplicate, current, temp, Scalar( 0, 0, 0 ), 1, 8, 0);
				imshow("Original frame",duplicate);
			}
			break; }
		}
	}
}

int main(int argc, char** argv)
{
	// Open the video file
	bool check = true;
	int one=1;
	// current video frame
	Mat frame=imread("1.jpg");
	imshow("Original frame",frame);
	duplicate =frame.clone();
	setMouseCallback( "Original frame", onMouse, &frame );
	
	// ---------------------------- Creating Regions to be omitted ---------------------------------///
	int key;
	while(check) {
		
		key=waitKey(0);
		printf("You pressed %d \n", key);
		if(key==1048675) {                              // Small case C - 'c'
			if(pcount>2) {
				rcount++;
				p = new Point[pseq->total];
				cvCvtSeqToArray(pseq, p, CV_WHOLE_SEQ);
				track=false;
				duplicate =frame.clone();
				line(duplicate, *(CV_GET_SEQ_ELEM(cv::Point, pseq, 0)), current, Scalar( 0, 0, 0 ), 1, 8, 0);
				imshow("Original frame",duplicate);
				cvSeqPush( nseq, &pcount );
				cvSeqPush( rseq, &p );
				p=NULL;
				pcount=0;
				cvClearSeq(pseq);
			}
		}
		else {
				cout << "You pressed Enter" << endl;
				check=false;
				input=0;
			}
	}
	////// ------------------------------------- END --------------------------------------------/////
	r=(const Point**) new Point*[rcount];
	PinR=new int[rcount];
	cvCvtSeqToArray(rseq, r, CV_WHOLE_SEQ);
	cvCvtSeqToArray(nseq, PinR, CV_WHOLE_SEQ);
	FileStorage fs("zones.xml", FileStorage::WRITE);
	fs << "redZone" << "{";
    if(pcount>0 || rcount>0) {
    	printf("%d\n", rcount );
    	printf("%d\n", PinR[0] );
    	fs << "NofR" << rcount;
    	string pname, rname, number;
    	stringstream strs;
		//fillPoly(mask, r, PinR, rcount, Scalar( 0, 0, 0 ), 8, 0);
		for (int i = 1; i <= rcount; i++)
		{
			strs << i;
			number = strs.str();
			rname="region"+number;
			fs << rname << "{";
			strs.str( std::string() );
			strs.clear();
			fs << "NofP" << PinR[i-1];
			for (int j = 1; j <= PinR[i-1]; j++)
			{
				strs << j;
				number = strs.str();
				pname="point"+number;
				fs << pname << "{";
				fs << "X" << r[i-1][j-1].x;
				fs << "Y" << r[i-1][j-1].y;
				fs << "}";
				strs.str( std::string() );
				strs.clear();
			}
			fs << "}";
		}
		//fillPoly(mask, r, PinR, rcount, Scalar(0), 8, 0);
	}
	fs << "}";
	fs.release();
	////-------------------------------- Displaying the extracted Foreground ----------------------------------------------//////////
	
	cvDestroyAllWindows();
	cvReleaseMemStorage( &pstorage );
	cvReleaseMemStorage( &rstorage );
	cvReleaseMemStorage( &nstorage );
	///////// ----------------------------------------------------END-----------------------------------------//////////////////////
}
