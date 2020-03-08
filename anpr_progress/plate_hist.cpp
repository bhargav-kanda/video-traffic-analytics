#include <cv.h>
#include <highgui.h>

IplImage* DrawHistogram( CvHistogram*,float ,float);

int main(int argc, char** argv){

	IplImage* img = cvLoadImage(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

    int numBins = 256; //number of bins in the dimension
    float range[] = {0, 255}; //value pair for the single dimension
    float *ranges[] = { range }; // ranges array to store the ranges for each dim

    // Create histogram with 1 dimension
    CvHistogram *hist = cvCreateHist(1, &numBins, CV_HIST_ARRAY, ranges, 1);
    // Clear histogram from garbage values
    cvClearHist(hist);

    // Create single channel images to store binary image
    IplImage* thres = cvCreateImage(cvGetSize(img), 8, 1);

    cvThreshold(img,thres,128,255,CV_THRESH_OTSU);

    // Calculate histogram for each color and draw it to the single channel image for the same color
    cvCalcHist(&thres, hist, 0, 0);
    IplImage* imgHistThres = DrawHistogram(hist,1,1);
    cvClearHist(hist);

    cvNamedWindow("Thres", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("ThresHist", CV_WINDOW_AUTOSIZE);

    cvShowImage("Thres", thres);
    cvShowImage("ThresHist", imgHistThres);

    cvWaitKey(0);
    return 0;
}

IplImage* DrawHistogram( CvHistogram *hist, float scaleX=1, float scaleY=1){

    float histMax = 0;
    cvGetMinMaxHistValue(hist, 0, &histMax, 0, 0);

    IplImage* imgHist = cvCreateImage(cvSize(256*scaleX, 64*scaleY), 8 ,1);
    cvZero(imgHist);
    int i;
    for(i=0;i<255;i++)
    {
        float histValue = cvQueryHistValue_1D(hist, i);
        float nextValue = cvQueryHistValue_1D(hist, i+1);

        CvPoint pt1 = cvPoint(i*scaleX, 64*scaleY);
        CvPoint pt2 = cvPoint(i*scaleX+scaleX, 64*scaleY);
        CvPoint pt3 = cvPoint(i*scaleX+scaleX, (64-nextValue*64/histMax)*scaleY);
        CvPoint pt4 = cvPoint(i*scaleX, (64-histValue*64/histMax)*scaleY);

        int numPts = 5;
        CvPoint pts[] = {pt1, pt2, pt3, pt4, pt1};

        cvFillConvexPoly(imgHist, pts, numPts, cvScalar(255,0,0,0),8,0);
    }
    return imgHist;
}
