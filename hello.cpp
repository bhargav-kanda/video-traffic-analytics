#include <highgui.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char** argv) {
     /* Create a window */
     cvNamedWindow("Example2", CV_WINDOW_AUTOSIZE);
     /* capture frame from video file */
     cv::VideoCapture capture(argv[1]);
	if(!capture.isOpened())

{

//Just to check if the video gets loaded or not

printf("Video Can't be loaded"); 
return 0;

}
     /* Create IplImage to point to each frame */
     cv::Mat frame;
     /* Loop until frame ended or ESC is pressed */
     while(1)
     {
        /* grab frame image, and retrieve */
        /* exit loop if fram is null / movie end */
        if(!capture.read(frame)) break;
        /* display frame into window */
        cv::imshow("Extracted Foreground",frame);
        /* if ESC is pressed then exit loop */
        char c = cvWaitKey(33);
        if(c==27) break;
     }

     /* destroy pointer to video */
     /* delete window */
     cvDestroyWindow("Example2");

     return EXIT_SUCCESS;
}
