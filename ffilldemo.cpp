#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

static void help()
{
    cout << "\nThis program demonstrated the floodFill() function\n"
            "Call:\n"
            "./ffilldemo [image_name -- Default: fruits.jpg]\n" << endl;

    cout << "Hot keys: \n"
            "\tESC - quit the program\n"
            "\tc - switch color/grayscale mode\n"
            "\tm - switch mask mode\n"
            "\tr - restore the original image\n"
            "\ts - use null-range floodfill\n"
            "\tf - use gradient floodfill with fixed(absolute) range\n"
            "\tg - use gradient floodfill with floating(relative) range\n"
            "\t4 - use 4-connectivity mode\n"
            "\t8 - use 8-connectivity mode\n" << endl;
}

Mat image0, image, gray, mask;
int ffillMode = 1;
int loDiff = 20, upDiff = 20;
int connectivity = 4;
int isColor = true;
bool useMask = false;
int newMaskVal = 255;

static void onMouse( int event, int x, int y, int, void* )
{
    switch( event ){
    case CV_EVENT_MOUSEMOVE: 
        if( drawing_box ){
            theBox.width = x-theBox.x;
            theBox.height = y-theBox.y;
        }
        break;

    case CV_EVENT_LBUTTONDOWN:
        drawing_box = true;
        theBox = cvRect( x, y, 0, 0 );
        break;

    case CV_EVENT_LBUTTONUP:
        drawing_box = false;
        if( theBox.width < 0 ){
            theBox.x += theBox.width;
            theBox.width *= -1;
        }
        if( theBox.height < 0 ){
            theBox.y += theBox.height;
            theBox.height *= -1;
        }
        draw_box( image, theBox );
        //cv::rectangle(parameter, theBox, CV_RGB(255,0,0));
        break;
	}
}


int main( int argc, char** argv )
{
    char* filename = argc >= 2 ? argv[1] : (char*)"fruits.jpg";
    image0 = imread(filename, 1);

    if( image0.empty() )
    {
        cout << "Image empty. Usage: ffilldemo <image_name>\n";
        return 0;
    }
    help();
    image0.copyTo(image);
    cvtColor(image0, gray, CV_BGR2GRAY);
    mask.create(image0.rows+2, image0.cols+2, CV_8UC1);

    namedWindow( "image", 0 );
    createTrackbar( "lo_diff", "image", &loDiff, 255, 0 );
    createTrackbar( "up_diff", "image", &upDiff, 255, 0 );

    setMouseCallback( "image", onMouse, 0 );

    for(;;)
    {
        imshow("image", isColor ? image : gray);

        int c = waitKey(0);
        if( (c & 255) == 27 )
        {
            cout << "Exiting ...\n";
            break;
        }
        switch( (char)c )
        {
        case 'c':
            if( isColor )
            {
                cout << "Grayscale mode is set\n";
                cvtColor(image0, gray, CV_BGR2GRAY);
                mask = Scalar::all(0);
                isColor = false;
            }
            else
            {
                cout << "Color mode is set\n";
                image0.copyTo(image);
                mask = Scalar::all(0);
                isColor = true;
            }
            break;
        case 'm':
            if( useMask )
            {
                destroyWindow( "mask" );
                useMask = false;
            }
            else
            {
                namedWindow( "mask", 0 );
                mask = Scalar::all(0);
                imshow("mask", mask);
                useMask = true;
            }
            break;
        case 'r':
            cout << "Original image is restored\n";
            image0.copyTo(image);
            cvtColor(image, gray, CV_BGR2GRAY);
            mask = Scalar::all(0);
            break;
        case 's':
            cout << "Simple floodfill mode is set\n";
            ffillMode = 0;
            break;
        case 'f':
            cout << "Fixed Range floodfill mode is set\n";
            ffillMode = 1;
            break;
        case 'g':
            cout << "Gradient (floating range) floodfill mode is set\n";
            ffillMode = 2;
            break;
        case '4':
            cout << "4-connectivity mode is set\n";
            connectivity = 4;
            break;
        case '8':
            cout << "8-connectivity mode is set\n";
            connectivity = 8;
            break;
        }
    }

    return 0;
}
