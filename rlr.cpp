#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include "opencv2/core/opengl_interop.hpp"
#include "cvaux.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;
using namespace cv;

const Point** r;
Point* p, current;
int* PinR;
int input=1;
int pcount=0;					//point count
int rcount=0;					// region count
int i,j,k,q,nop;
CvMemStorage* pstorage = cvCreateMemStorage(0);
CvSeq* pseq = cvCreateSeq( CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), pstorage);
CvMemStorage* rstorage = cvCreateMemStorage(0);
CvSeq* rseq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint*), rstorage);
CvMemStorage* dirstorage = cvCreateMemStorage(0);
CvSeq* dirseq = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint*), dirstorage);
CvMemStorage* nstorage = cvCreateMemStorage(0);
CvSeq* nseq = cvCreateSeq( CV_32SC1, sizeof(CvSeq), sizeof(int), nstorage);
CvPoint **tempp;
Mat uvPoint, tempMat, tempMat2, rotationMatrix, intrinsic, rvecs, tvecs, distCoeffs;
double s;

void realCoord(Point2f current, Point3f* RealCurrent) {
	uvPoint = cv::Mat::ones(3,1,cv::DataType<double>::type); //u,v,1
	uvPoint.at<double>(0,0) = current.x; //got this point using mouse callback
	uvPoint.at<double>(1,0) = current.y;
	tempMat = rotationMatrix.inv() * intrinsic.inv() * uvPoint;
	tempMat2 = rotationMatrix.inv() * tvecs;
	s = 1 + tempMat2.at<double>(2,0); //0 represents the height Zconst
	s /= tempMat.at<double>(2,0);
	uvPoint = rotationMatrix.inv() * (s * intrinsic.inv() * uvPoint - tvecs);
	uvPoint.at<double>(2,0) = 0;
	Point3f Real(uvPoint);
	*RealCurrent=Real;
	//cout<<"Poinsts are"<<Real<<"      "<<RealCurrent<<endl;
}

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
	Mat prevframe, frame;
	prevframe=cvQueryFrame( capture );
	frame=imread("bg.jpg");
	//(prevframe.rows,prevframe.cols,CV_8UC1);
	imshow("Original frame",frame);
	//change_mdistance(mdistance, 0);
	
	// ---------------------------- Creating Regions to be omitted ---------------------------------///
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

	// ---------------------------- Creating Regions to be omitted ---------------------------------///
	fs=FileStorage("rlrflowdir.xml", FileStorage::READ);
	n=fs["redZone"];
	if(rcount==(int)n["NofR"]) {
		//cout<< rcount << endl;
		for (i = 0; i < rcount; i++)
		{
			strs << i+1;
			number = strs.str();
			rname="region"+number;
			reg=n[rname];
			strs.str( std::string() ); strs.clear();
			for (j = 0; j < 2; j++)
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
			cvSeqPush( dirseq, &p );
			p=NULL;
			pcount=0;
			cvClearSeq(pseq);
		}
		fs.release();
	} else {
		printf("The number of regions do not match in the inputs\n");
		return 1;
	}
		
	////// ------------------------------------- END --------------------------------------------/////


	
	// ---------------------------- Reading Calibration data ---------------------------------///
	FileStorage fs1("calibration.xml", FileStorage::READ);
	fs1["intrinsic"] >> intrinsic;
	fs1["Rotation"] >> rvecs;
	fs1["translation"] >> tvecs;
	fs1["distCoeffs"] >> distCoeffs;
	cout << "intrinsic = "<< endl << " "  << intrinsic << endl << endl;
	cout << "rvecs = "<< endl << " "  << rvecs << endl << endl;
	cout << "tvecs = "<< endl << " "  << tvecs << endl << endl;
	cout << "distortion Coeffs = "<< endl << " "  << distCoeffs << endl << endl;
	Rodrigues(rvecs, rotationMatrix);
	fs1.release();
	////// ------------------------------------- END --------------------------------------------/////
	Mat* masks = new Mat[rcount];
	Mat mask(prevframe.rows,prevframe.cols,CV_8UC1),redMask(prevframe.rows,prevframe.cols,CV_8UC1);

	int color;
	bool stop(false);
	bool pause(false);
	r=(const Point**) new Point*[rcount];
	Point** dir=(Point**) new Point*[rcount];
	PinR=new int[rcount];
	cvCvtSeqToArray(rseq, r, CV_WHOLE_SEQ);
	cvCvtSeqToArray(dirseq, dir, CV_WHOLE_SEQ);
	cvCvtSeqToArray(nseq, PinR, CV_WHOLE_SEQ);
	double magnitude, xCompCurrent, yCompCurrent,xComp[rcount], yComp[rcount], TotalPts[rcount], MovingPts[rcount];
	for (int i = 0; i < rcount; ++i)
	{
		xCompCurrent=dir[i][1].x-dir[i][0].x;
		yCompCurrent=dir[i][1].y-dir[i][0].y;
		magnitude=(xCompCurrent*xCompCurrent)+(yCompCurrent*yCompCurrent);
		magnitude=sqrt(magnitude);
		xComp[i]=xCompCurrent/magnitude;
		yComp[i]=yCompCurrent/magnitude;
	}
	cout<< xComp[0] << xComp[1] <<xComp[2] <<xComp[3] << endl;
	cout<< yComp[0] << yComp[1] <<yComp[2] <<yComp[3] << endl;
	Mat masked_frame,masked_frameclone;
	Mat framebgr(frame.size(), CV_8UC1),masked_framebgr(frame.size(), CV_8UC1),prevframebgr(prevframe.size(), CV_8UC1),framebgrclone,framebgrclone2,prevframebgrclone;
	cout<< dir[1][1] << endl;
    if(pcount>0 || rcount>0) {
		//fillPoly(mask, r, PinR, rcount, Scalar( 0, 0, 0 ), 8, 0);
		for (int i = 0; i < rcount; ++i)
		{
			masks[i]=Mat(prevframe.rows,prevframe.cols,CV_8UC1);
			masks[i]=Scalar(0);
			fillPoly(masks[i], &r[i], &PinR[i], 1, Scalar(256,256,256), 8, 0);
		}
	}
	add(masks[0], masks[1], mask);
	add(mask, masks[2], mask);
	add(mask, masks[3], mask);
	int win_size = 7;
	vector<Point2f> points1, points0, points2, start;
	vector<uchar> status;
    vector<float> err;
    TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);
	cvtColor(prevframe,prevframebgr,CV_BGR2GRAY);
	k=0;
	vector<cv::KeyPoint> keypoints;
	vector<int> queryIdxs;
	Point3f Real0,Real1;
	int same=0, step=3, t, zone;
	vector<double> times, distances, startFrame, velocity, dotPro;
	double res;
	int red, prevRed=-1, redFlag=0;

	////////-------------------------------- Displaying the extracted Foreground ----------------------------------------------//////////
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
		frame = cvQueryFrame( capture );
		k++;
		frame.copyTo(masked_frame,mask);
///////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////// DELAY COMPUTATION //////////////////////////////////////////////////////////////////////
		if(k==1) {															/// Initialization
			cvtColor(masked_frame,prevframebgr,CV_BGR2GRAY);
			FAST(prevframebgr,keypoints,20,true);
			KeyPoint::convert(keypoints, points0, queryIdxs);
			cout<<"Points0 size"<<points0.size()<<endl;
			// for( i=j=0; i < points0.size(); i++ )
	  //       {
	  //       	realCoord(points0[i], &Real0);
	  //       	if(Real0.y<70) {
		 //        	points0.erase(points0.begin()+i);
		 //        	i--;
	  //       	}
   //          }
            //start=points0;
            //startFrame.assign(points0.size(), cvGetCaptureProperty(capture, 1));
            velocity.assign(points0.size(), 0);
            dotPro.assign(points0.size(), 0);
		}

		if(k%step==step-1) {
			redMask=Scalar(0);
			cvtColor(frame,framebgr,CV_BGR2GRAY);
			cvtColor(masked_frame,masked_framebgr,CV_BGR2GRAY);
			calcOpticalFlowPyrLK(prevframebgr, framebgr, points0, points2, status, err, Size(win_size, win_size), 3, termcrit, 1, 0.5);
			FAST(masked_framebgr,keypoints,20,true);
			//cout<<"points2 "<<points2.size()<<endl;
			KeyPoint::convert(keypoints, points1, queryIdxs);
			// cout<<"Points1 size"<<points1.size()<<endl;
			// cout<<"Points0 size"<<points0.size()<<endl;
			// cout<<"Points2 size"<<points2.size()<<endl;
			for(i=t=0;i<points2.size();i++) {
				if(status[i]==0 || mask.at<uchar>(points0[i].y,points0[i].x)==0) {
					// startFrame.erase(startFrame.begin()+i);
					// start.erase(start.begin()+i);
					status.erase(status.begin()+i);
					points2.erase(points2.begin()+i);
					points0.erase(points0.begin()+i);
					velocity.erase(velocity.begin()+i);
					dotPro.erase(dotPro.begin()+i);
					i--;
				}
			}
			for (j = 0; j < rcount; ++j)
			{
				MovingPts[j]=0;
				TotalPts[j]=0;
			}
			for( i = 0; i < points2.size(); i++ )
	        {
	        	realCoord(points0[i], &Real0);
	        	realCoord(points2[i], &Real1);
	        	for (j = 0; j < rcount; ++j)
	        	{
	        		if(masks[j].at<uchar>(points0[i].y,points0[i].x)!=0) {
	        			zone=j;
	        			TotalPts[zone]++;
	        			break;
	        		}
	        	}
	        	xCompCurrent=points2[i].x-points0[i].x;
	        	yCompCurrent=points2[i].y-points0[i].y;
	        	magnitude=(xCompCurrent*xCompCurrent)+(yCompCurrent*yCompCurrent);
				magnitude=sqrt(magnitude);
				if (magnitude>2)
				{
					xCompCurrent=xCompCurrent/magnitude;
					yCompCurrent=yCompCurrent/magnitude;
					dotPro[i]=(xCompCurrent*xComp[zone])+(yCompCurrent*yComp[zone]);
				} else
					dotPro[i]=0;
				
	        	res = cv::norm(Real0-Real1);
	        	velocity[i]=res/(step/25.0);
	        }

			//cout<<"Max of steps "<<t<<endl;
			//printf("crossed this\n");
			framebgr.copyTo(framebgrclone);
			framebgr.copyTo(framebgrclone2);
			prevframebgr.copyTo(prevframebgrclone);
			masked_frame.copyTo(masked_frameclone);
	        for( i = 0; i < points2.size(); i++ )
	        {
	        	if(dotPro[i]>0.85 && velocity[i]>1) {
	        		circle(masked_frameclone, points2[i], 0, Scalar(0,0,255), 4, 8, 0);
	        		drawArrow(masked_frameclone, points0[i], points2[i], CV_RGB(0,0,255), 20);
	        		for (j = 0; j < rcount; ++j)
	        		{
	        			if(masks[j].at<uchar>(points0[i].y,points0[i].x)!=0) {
		        			MovingPts[j]++;
		        			//cout<< j<< "  "<< xComp[j]<<"  "<< yComp[j]<<"  "<< points0[i]<< endl;
		        			break;
	        			}
	        		}
	        	}
	        	else {
	        		circle(masked_frameclone, points2[i], 0, Scalar(0,0,0), 4, 8, 0);
	        		
	        	}
            }
            for( i = 0; i < points0.size(); i++ )
	        {
	        	circle(prevframebgrclone, points0[i], 0, Scalar(0,255,0), 4, 8, 0);
            }
            for( i = 0; i < points1.size(); i++ )
	        {
	        	circle(framebgrclone, points1[i], 0, Scalar(0,0,255), 4, 8, 0);
            }

            for (i = 0; i < rcount; ++i)
            {
            	if ((MovingPts[i]/TotalPts[i])>0.15)
            	{
            		if (redFlag==0)
            		{
            		 	redFlag=1;
            		 	red=i;
            		}
            		if (redFlag==1)
            		{
            			if ((MovingPts[i]/TotalPts[i])>(MovingPts[red]/TotalPts[red]))
            			{
            				red=i;
            			}
            		}
            	}
            }

            if (redFlag==0)
            {
            	red=prevRed;
            	redFlag=1;
            }

            if(redFlag!=0 && prevRed!=-1) {
            	masks[red].copyTo(redMask);	
            }
            

            //minMaxLoc(QLengths,0,&QLength,0,0,noArray());
	        //cout<<"Distance between 2 points is" <<res<<endl;
	        //drawKeypoints(frame, keypoints, prevframe, Scalar(255,0,0));
			//cornerSubPix( framebgr, points1, cvSize( win_size, win_size ) , Size( -1, -1 ), termcrit);
			imshow("Red Zone", redMask);
			imshow("Imp points",prevframebgrclone);
			imshow("Imp points1",framebgrclone);
			imshow("Imp points2",masked_frameclone);

			for( i = 0; i < points1.size(); i++ )
	        {
	        	same=0;
	        	//realCoord(points1[i], &Real0);
		        	for(j=0;j < points2.size(); j++ ) {
		        		if(norm(points1[i]-points2[j])<3) {
		        			same=1;
		        			break;
		        		}
		        	}
		        	if(!same) {
		        		points2.push_back(points1[i]);
		        		start.push_back(points1[i]);
		        		startFrame.push_back(cvGetCaptureProperty(capture, 1));
		        		velocity.push_back(0);
		        		dotPro.push_back(0);
		        	}
            }
   //          cout<<"Points2 size"<<points2.size()<<endl;
			// cout<<"Points1 size"<<points1.size()<<endl;
			
			//calcOpticalFlowPyrLK(prevframebgr, framebgr, points0, points1, status, err, Size(win_size, win_size), 3, termcrit, 0);
			
			framebgr.copyTo(prevframebgr);
			//goodFeaturesToTrack( prevframebgr, points0, corner_count, 0.01, 5, mask, 3, 0, 0.04 );	
			swap(points2,points0);
			points2.resize(0);
			velocity.assign(points0.size(), 0);
            dotPro.assign(points0.size(), 0);
            prevRed=red;
            red=-1;
            redFlag=0;
		}
		imshow("Original frame",frame);
		// press key to stop
		if(waitKey(10)==1048608)					// Space bar
			pause= true;
		else if(waitKey(10)==1048603) {				// Escape key (hold the key)
			stop=true;
		}
	}
	cvDestroyAllWindows();
	cvReleaseCapture(&capture);
	cvReleaseMemStorage( &pstorage );
	cvReleaseMemStorage( &rstorage );
	cvReleaseMemStorage( &nstorage );
	///////// ----------------------------------------------------END-----------------------------------------//////////////////////
}