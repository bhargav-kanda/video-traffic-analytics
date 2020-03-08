/* Create blackhat and sobel filter and take intersection.
Then implement the new paper found
Author: Ram
Project: ANPR
*/
#include "cv.h"
#include "highgui.h"
#include "stdio.h"

void extract(IplImage* in, IplImage* out) {
	IplImage* blackhat = cvCreateImage(cvGetSize(in),IPL_DEPTH_8U,3);
	IplImage* grayscale = cvCreateImage(cvGetSize(in),IPL_DEPTH_8U,1);
	IplImage* binary = cvCreateImage(cvGetSize(in),IPL_DEPTH_8U,1);
	IplConvKernel* blackhat_kernel = cvCreateStructuringElementEx( 9, 9, 5, 5, CV_SHAPE_RECT );
	cvMorphologyEx(in, blackhat, NULL, blackhat_kernel, CV_MOP_BLACKHAT , 1);
	cvCopy(blackhat, out);
	cvShowImage("black hat", blackhat); 
	cvShowImage("gray scale", grayscale); 
	cvShowImage("binary", binary); 
	cvShowImage("in", in);
	cvShowImage("out", out);
	cvWaitKey(0);
	cvReleaseImage(&blackhat); cvReleaseImage(&grayscale); cvReleaseImage(&binary); 
}

int main( int argc, char** argv )
{
	IplImage* input = cvLoadImage( argv[1] );
	IplImage* output = cvCreateImage(cvGetSize(input),IPL_DEPTH_8U,3);
	extract(input, output);  																				//implement the extraction function
	cvNamedWindow("input");
	cvNamedWindow("output");
    cvShowImage("input", input);
    cvShowImage("output", output);
    cvWaitKey(0);
    cvReleaseImage(&input);
    cvReleaseImage(&output);
    return 0;
}
