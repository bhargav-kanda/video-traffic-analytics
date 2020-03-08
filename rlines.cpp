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
Point** dir;
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
CvMemStorage* pointstorage = cvCreateMemStorage(0);
CvSeq* pointseq = cvCreateSeq( CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), pointstorage);
CvMemStorage* dirstorage = cvCreateMemStorage(0);
CvSeq* dirseq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint*), dirstorage);
CvPoint **tempp;
CvPoint *t;
void drawArrow(Mat image, Point p, Point q, Scalar color, int arrowMagnitude = 9, int thickness=1, int line_type=8, int shift=0) 
{
    //Draw the principle line
    line(image, p, q, color, thickness, line_type, shift);
    const double PI = 3.141592653;
    //compute the angle alpha
    double angle = atan2((double)p.y-q.y, (double)p.x-q.x);
    //compute the coordinates of the first segment
    p.x = (int) ( q.x +  arrowMagnitude * cos(angle + PI/4));
    p.y = (int) ( q.y +  arrowMagnitude * sin(angle + PI/4));
    //Draw the first segment
    line(image, p, q, color, thickness, line_type, shift);
    //compute the coordinates of the second segment
    p.x = (int) ( q.x +  arrowMagnitude * cos(angle-PI/4));
    p.y = (int) ( q.y +  arrowMagnitude * sin(angle-PI/4));
    //Draw the second segment
    line(image, p, q, color, thickness, line_type, shift);
}  
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
				cvSeqPush( pointseq, &current );
				*image = duplicate.clone();
				track=false;
				//current = Point(x,y);
				//points = &point;
				circle(*image, current, 0, Scalar( 0, 0, 0 ), 4, 8, 0);
				imshow("Mask",*image);
			break;	}
		
			case CV_EVENT_LBUTTONUP: {
				if(pcount<2) {
					track=true;
					printf("%d %d\n",x,y);
				} else if(pcount==2) {
					p = new Point[pointseq->total];
					cvCvtSeqToArray(pointseq, p, CV_WHOLE_SEQ);
					//printf("Hiiiiiii\n");
					track=false;
					//drawArrow(duplicate, *(CV_GET_SEQ_ELEM(cv::Point, pointseq, 0)), current, Scalar( 0, 0, 0 ), 1, 8, 0);
					imshow("Mask",duplicate);
					cvSeqPush( dirseq, &p );
					cout << p[1] << endl;
					p=NULL;
					pcount=0;
					cvClearSeq(pointseq);
				}
			break; }

			case CV_EVENT_MOUSEMOVE: {
			if(track) {
				duplicate = (*image).clone();
				//printf("%d %d %d %d\n",x,y,current.x, current.y);
				Point temp = Point(x,y);
				//line(duplicate, current, temp, Scalar( 0, 0, 0 ), 1, 8, 0);
				drawArrow(duplicate, current, temp, CV_RGB(0,0,0), 20);
				imshow("Mask",duplicate);
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
	// ---------------------------- Loading Regions to be omitted ---------------------------------///
	FileStorage fs("rlrzones.xml", FileStorage::READ);
	FileNode n, reg, pt;
	n=fs["redZone"];
	rcount=(int)n["NofR"];
	//cout<< rcount << endl;
	string pname, rname, number;
    stringstream strs;
	for (i = 0; i < rcount; i++)
	{
		strs << i+1;
		number = strs.str();
		rname="region"+number;
		reg=n[rname];
		strs.str( std::string() ); strs.clear();
		pcount=reg["NofP"];
		for (j = 0; j < pcount; j++)
		{
			strs << j+1;
			number = strs.str();
			pname="point"+number;
			pt=reg[pname];
			current.x=pt["X"];
			current.y=pt["Y"];
			strs.str( std::string() ); strs.clear();
			cvSeqPush( pseq, &current );
		}
		p = new Point[pseq->total];
		cvCvtSeqToArray(pseq, p, CV_WHOLE_SEQ);
		cvSeqPush( rseq, &p );
		cvSeqPush( nseq, &pcount );
		p=NULL;
		pcount=0;
		cvClearSeq(pseq);
	}
	fs.release();
	
	////// ------------------------------------- END --------------------------------------------/////
	Mat mask = Mat(frame.rows,frame.cols,CV_8UC1);
	Mat* masks = new Mat[rcount];
	mask=Scalar(0);
	r=(const Point**) new Point*[rcount];
	PinR=new int[rcount];
	cvCvtSeqToArray(rseq, r, CV_WHOLE_SEQ);
	cvCvtSeqToArray(nseq, PinR, CV_WHOLE_SEQ);
    if(pcount>0 || rcount>0) {
		for (int i = 0; i < rcount; ++i)
		{
			masks[i]=Mat(frame.rows,frame.cols,CV_8UC1);
			masks[i]=Scalar(0);
			fillPoly(masks[i], &r[i], &PinR[i], 1, 256, 8, 0);
		}
	}
	add(masks[0], masks[1], mask);
	add(mask, masks[2], mask);
	add(mask, masks[3], mask);
	duplicate=mask.clone();
	imshow("Mask", mask);
	setMouseCallback( "Mask", onMouse, &mask );
	
	// ---------------------------- Creating Regions to be omitted ---------------------------------///
	int key;
	key=waitKey(0);
	printf("You pressed %d \n", key);
	////// ------------------------------------- END --------------------------------------------/////
	dir=(Point**) new Point*[rcount];
	Point** finalDir=(Point**) new Point*[rcount];
	cvCvtSeqToArray(dirseq, dir, CV_WHOLE_SEQ);
	cvCvtSeqToArray(dirseq, finalDir, CV_WHOLE_SEQ);
	cout<< dir[1][1]<< endl;
	for (int i = 0; i < rcount; ++i)
	{
		current=dir[i][0];
		cout<< current << endl;
		for (int j = 0; j < rcount; ++j)
		{
			if(masks[j].at<uchar>(current.y,current.x)!=0) {
				cout<< masks[j].at<uchar>(current.y,current.x) << endl;
				cout<< i << j << endl;
				finalDir[j]=dir[i];
				break;
			}
		}		
	}
	cout<< finalDir[1][1]<< endl;
	
	fs=FileStorage("rlrflowdir.xml", FileStorage::WRITE);
	fs << "redZone" << "{";
    if(pcount>0 || rcount>0) {
    	fs << "NofR" << rcount;
    	string pname, rname, number;
    	stringstream strs;
		for (int i = 1; i <= rcount; i++)
		{
			strs << i;
			number = strs.str();
			rname="region"+number;
			fs << rname << "{";
			strs.str( std::string() );
			strs.clear();
			for (int j = 1; j <= 2; j++)  //// 2 points - start and end
			{
				strs << j;
				number = strs.str();
				pname="point"+number;
				fs << pname << "{";
				fs << "X" << finalDir[i-1][j-1].x;
				fs << "Y" << finalDir[i-1][j-1].y;
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
	cvDestroyAllWindows();
	cvReleaseMemStorage( &pstorage );
	cvReleaseMemStorage( &rstorage );
	cvReleaseMemStorage( &nstorage );
	cvReleaseMemStorage( &pointstorage );
	cvReleaseMemStorage( &dirstorage );
	///////// ----------------------------------------------------END-----------------------------------------//////////////////////
}
