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

int mdistance=25;
int main(int argc, char** argv)
{
	// Open the video file
	int one=1;
	CvCapture* capture = cvCreateFileCapture( argv[1] );
	if (!capture) {
		printf("cannot open file");
		return 0;
	}
	// current video frame
	Mat frame;
	// foreground binary image
	Mat foreground, foreground_shadows;
	Mat background;
	Mat IBackground=imread("bg.jpg");
	// The Mixture of Gaussian object
	BackgroundSubtractorMOG2 mog(1000, mdistance, true);
	mog.set("nmixtures", 3);
	//mog.setBackgroundImage(IBackground);
	// for all frames in video
	frame=IBackground;
	Mat mask = Mat(frame.rows,frame.cols,CV_8UC1);
	imshow("Original frame",frame);
	duplicate =frame.clone();
	createTrackbar("M Distance", "Original frame", &mdistance, 100);
	//change_mdistance(mdistance, 0);
	
	// ---------------------------- Creating Regions to be omitted ---------------------------------///
	FileStorage fs("zones.xml", FileStorage::READ);
	FileNode n, reg, pt;
	n=fs["redZone"];
	rcount=(int)n["NofR"];
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
	
	// ---------------------------- Reading Calibration data ---------------------------------///
	FileStorage fs1("calibration.xml", FileStorage::READ);
	Mat intrinsic, rvecs, tvecs, distCoeffs;
	fs1["intrinsic"] >> intrinsic;
	fs1["Rotation"] >> rvecs;
	fs1["translation"] >> tvecs;
	fs1["distCoeffs"] >> distCoeffs;
	cout << "intrinsic = "<< endl << " "  << intrinsic << endl << endl;
	cout << "rvecs = "<< endl << " "  << rvecs << endl << endl;
	cout << "tvecs = "<< endl << " "  << tvecs << endl << endl;
	cout << "distortion Coeffs = "<< endl << " "  << distCoeffs << endl << endl;
	fs1.release();
	////// ------------------------------------- END --------------------------------------------/////

	int color;
	bool stop(false);
	bool pause(false);
	r=(const Point**) new Point*[rcount];
	PinR=new int[rcount];
	cvCvtSeqToArray(rseq, r, CV_WHOLE_SEQ);
	cvCvtSeqToArray(nseq, PinR, CV_WHOLE_SEQ);
	vector<vector<Point> > contours,contours1,fcontours;
	Mat masked_frame, masked_foreground, foreground_noshadow,CMFrame, background_gray,final_contour_mask(frame.size(), CV_8UC1);
	Mat all_edges,bg_edges,outer_edges,outer_edges1,outer_edges2,outer_edges3,contour_edges,masked_foreground_gray,inner_edges1(frame.size(), CV_8UC1), inner_edges(frame.size(), CV_16SC1); 
	Mat foreground_edges(frame.size(), CV_8UC1);
	Mat contour_mask = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
	Mat noshadow_img_or,noshadow_img_and,hor_img = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
	Mat ver_img = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
	Point start=Point(0,0);
	Point finish=Point(0,0);
	mask=Scalar(128);
	int thresh2_l = 50;
	int thresh2_u = 200;
	int thresh=18;
	int scale=16;
	//imshow("inner edges", inner_edges);
	createTrackbar("outer_edges lower","inner edges", &thresh, 100);
	createTrackbar("outer_edges upper","inner edges", &scale, 40);
    if(pcount>0 || rcount>0) {
		//fillPoly(mask, r, PinR, rcount, Scalar( 0, 0, 0 ), 8, 0);
		//fillPoly(mask, r, PinR, rcount, Scalar(0), 8, 0);
	}
	vector<Vec4i> hierarchy;
	////-------------------------------- Displaying the extracted Foreground ----------------------------------------------//////////
	while (!stop) {
		// read next frame if any
		if (!cvQueryFrame( capture ))
			break;
		if(pause) {
			waitKey(0);
			pause=false;
		}
		// update the background
		// and return the foreground
		frame.copyTo(masked_frame,mask);
		mog.set("varThreshold",mdistance);
		mog(masked_frame,foreground_shadows,0.01);
		mog.getBackgroundImage(background);
		//imshow("Current background",background);
		//imshow("Extracted Foreground with shadows",foreground_shadows);
		threshold(foreground_shadows,foreground,126,255,THRESH_BINARY);				
		threshold(foreground_shadows,foreground_noshadow,127,255,THRESH_BINARY);				// Removing the Shadows
		Canny( foreground, outer_edges1, thresh2_l, thresh2_u, 5 );
		Mat element = getStructuringElement(MORPH_RECT, Size(3,3), Point(2,2));
		//erode(foreground, foreground, element, Point(-1,-1), 1);
		morphologyEx(foreground,foreground,MORPH_CLOSE,getStructuringElement(MORPH_RECT,Size(5,5)));
		masked_frame.copyTo(masked_foreground,foreground);
		cvtColor(masked_foreground, masked_foreground_gray, CV_BGR2GRAY);
		cvtColor(background, background_gray, CV_BGR2GRAY);
		Sobel( masked_foreground_gray, all_edges, -1,1,1,3,scale/10.0 );
		threshold(all_edges,all_edges,thresh,255,3);
		Sobel( background_gray, bg_edges, -1,1,1,3,1.5 );
		threshold(all_edges,all_edges,0,255,THRESH_BINARY);	
		Canny( foreground, outer_edges2, thresh2_l, thresh2_u, 5 );
		Sobel( foreground, outer_edges3, -1,1,1,3 );
		threshold(outer_edges2,outer_edges2,0,255,THRESH_BINARY);	
		addWeighted(outer_edges1,1,outer_edges2,1,0,outer_edges);
		addWeighted(all_edges,1,bg_edges,-1,0,all_edges);
		addWeighted(all_edges,1,outer_edges3,-1,0,inner_edges);
		addWeighted(inner_edges,1,outer_edges,-1,0,inner_edges);
		//imshow("inner edges1",inner_edges);
		//threshold(inner_edges,inner_edges,0,255,THRESH_BINARY);	
		cv::findContours( outer_edges, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
		for (size_t idx = 0; idx < contours.size(); idx++) {
			if(contours[idx].size()>200) {
				//color=idx%contours.size();
				cv::drawContours(masked_frame, contours, idx, Scalar(0,0,255), CV_FILLED);
				cv::drawContours(contour_mask, contours, idx, Scalar(255), CV_FILLED);
			// 	approxPolyDP(Mat(contours[idx]), contours_poly[idx], 3, true);
			// 	boundRect[idx]=boundingRect(Mat(contours_poly[idx]));
			// 	rectangle(masked_frame,boundRect[idx].tl(),boundRect[idx].br(),Scalar(0,0,255),2,8,0);
			}
    	}
    	addWeighted(contour_mask,1,foreground,1,0,contour_mask);
    	Canny(contour_mask,contour_edges,thresh2_l, thresh2_u, 5);
    	findContours( contour_edges, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
    	for (size_t idx = 0; idx < contours1.size(); idx++) {
			if(contours1[idx].size()>200) {
				//color=idx%contours.size();
				cv::drawContours(final_contour_mask, contours1, idx, Scalar(255), -1);
			}
    	}
    	for(i=0;i<hor_img.rows;i++) {
    		for(j=0;j<hor_img.cols;j++) {
    			while(foreground.at<uchar>(i,j)!=0) {
    				if(inner_edges.at<uchar>(i,j)!=0 && start==Point(0,0)) {
    					start=Point(j,i);
    					finish=Point(j,i);
    				}
    				j++;
    				if(inner_edges.at<uchar>(i,j)!=0) {
    					finish=Point(j,i);
    				}
    			}
    			if(start!=Point(0,0) && finish!=Point(0,0)) {
    				line(hor_img, start, finish, Scalar(255), 1);
    				start=Point(0,0);
    				finish=Point(0,0);
    			}
    		}
    	}
    	for(j=0;j<ver_img.cols;j++) {
    		for(i=0;i<ver_img.rows;i++) {
    			while(foreground.at<uchar>(i,j)!=0) {
    				if(inner_edges.at<uchar>(i,j)!=0 && start==Point(0,0)) {
    					start=Point(j,i);
    					finish=Point(j,i);
    				}
    				i++;
    				if(inner_edges.at<uchar>(i,j)!=0) {
    					finish=Point(j,i);
    				}
    			}
    			if(start!=Point(0,0) && finish!=Point(0,0)) {
    				line(ver_img, start, finish, Scalar(255), 1);
    				start=Point(0,0);
    				finish=Point(0,0);
    			}
    		}
    	}
    	bitwise_and(hor_img, ver_img, noshadow_img_and);
    	morphologyEx(noshadow_img_and,noshadow_img_and,MORPH_CLOSE,getStructuringElement(MORPH_RECT,Size(5,5)));
    	addWeighted(noshadow_img_and,1,foreground_noshadow,1,0,foreground_noshadow);
    	Canny(foreground_noshadow, foreground_edges, 50, 150, 5);
    	findContours(foreground_edges, fcontours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
    	std::vector<Rect> boundRect(fcontours.size());
		vector<vector<Point> > contours_poly(fcontours.size());
    	//printf("I am here\n");
		for (size_t idx = 0; idx < fcontours.size(); idx++) {
			if(fcontours[idx].size()>200) {
				//color=idx%contours.size();
				cv::drawContours(frame, fcontours, idx, Scalar(0,0,255), CV_FILLED);
				approxPolyDP(Mat(fcontours[idx]), contours_poly[idx], 3, true);
				boundRect[idx]=boundingRect(Mat(contours_poly[idx]));
				rectangle(frame,boundRect[idx].tl(),boundRect[idx].br(),Scalar(0,0,255),2,8,0);
			}
    	}
		masked_frame.copyTo(CMFrame, contour_mask);
		// imshow("Contours", contour_mask);
		// imshow("Final Contours", final_contour_mask);
 		imshow("all edges", all_edges);
 		imshow("outer edges", outer_edges3);
 		imshow("inner edges", inner_edges);
		imshow("Extracted Foreground",foreground);
		imshow("Original frame",masked_frame);
		//imshow("Masked Foreground", masked_foreground);
		imshow("no shadow and", noshadow_img_and);
		imshow("final foreground", foreground_noshadow);
		imshow("horizontal", hor_img);
		imshow("vertical", ver_img);
		imshow("Foreground edges", frame);
		masked_foreground=Scalar(0);
		CMFrame=Scalar(0);
		contour_mask=Scalar(0);
		final_contour_mask=Scalar(0);
		hor_img=Scalar(0);
		ver_img=Scalar(0);
		frame = cvQueryFrame( capture );
		// press key to stop
		if(waitKey(10)==1048608)					// Space bar
			pause= true;
		else if(waitKey(10)==1048603){				// Escape key (hold the key)
			stop=true;
		}
	}
	cvDestroyAllWindows();
	cvReleaseCapture(&capture);
	cvReleaseMemStorage( &pstorage );
	cvReleaseMemStorage( &rstorage );
	cvReleaseMemStorage( &nstorage );
	imwrite("bg.jpg", background);
	///////// ----------------------------------------------------END-----------------------------------------//////////////////////
}