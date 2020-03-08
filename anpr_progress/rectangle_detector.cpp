// The "Rectangle Detector" program.
// It loads several images sequentially and tries to find rectangles in
// each image

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <math.h>
#include <string.h>

using namespace cv;
using namespace std;

void help()
{
	cout <<
	"\nA program using pyramid scaling, Canny, contours, contour simpification and\n"
	"memory storage (it's got it all folks) to find\n"
	"rectangles in an image\n"
	"Returns sequence of rectangles detected on the image.\n"
	"the sequence is stored in the specified memory storage\n"
	"Call:\n"
	"./<executable_name>\n"
    "Using OpenCV version %s\n" << CV_VERSION << "\n" << endl;
}


int thresh = 50, N = 11;
const char* wndname = "Rectangle Detection Demo";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// returns sequence of rectangles detected on the image.
// the sequence is stored in the specified memory storage
void findRectangles( const Mat& image, vector<vector<Point> >& rectangles )
{
    rectangles.clear();
    
    Mat pyr, timg, gray0(image.size(), CV_8U), gray;
    
    // down-scale and upscale the image to filter out the noise
    //pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
   // pyrUp(pyr, timg, image.size());
    vector<vector<Point> > contours;
    
    // find rectangles in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&image, 1, &gray0, 1, ch, 1);
        
        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch rectangles with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

            vector<Point> approx;
            
            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);
                
                // rectangle contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)) )
                {
                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.3 )
                    	rectangles.push_back(approx);
                }
            }
        }
    }
}


// the function draws all the rectangles in the image
void drawRectangles( Mat& image, const vector<vector<Point> >& rectangles )
{
    for( size_t i = 0; i < rectangles.size(); i++ )
    {
        const Point* p = &rectangles[i][0];
        int n = (int)rectangles[i].size();
        polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, CV_AA);
    }

    imshow(wndname, image);
    imwrite("rect_blobs_pic7.jpg",image);
}


int main(int argc, char** argv)
{
    /*static const char* names[] = { "pic1.jpg", "pic2.png", "pic3.png",
        "pic4.png", "pic5.png", "pic6.png", 0 };*/
    help();
    namedWindow( wndname, CV_WINDOW_NORMAL );
    vector<vector<Point> > rectangles;
    
    /*for( int i = 0; names[i] != 0; i++ )
    {*/
        Mat image = imread(argv[1], 1), blurred;
        //Mat thres(image.cols,image.rows,CV_8UC1);
        //threshold(image,thres,128,255,CV_THRESH_OTSU);

        if( image.empty() )
        {
            cout << "Couldn't load " << argv[1] << endl;
            //continue;
        }
        
        // Sharpening the image
        GaussianBlur(image,blurred,Size(17,17),3);
        addWeighted(image,1.5,blurred,-0.5,0,image);

        findRectangles(image, rectangles);
        drawRectangles(image, rectangles);

        waitKey(0);
        //int c = waitKey();
        //if( (char)c == 27 )
            //break;
    //}

    return 0;
}
