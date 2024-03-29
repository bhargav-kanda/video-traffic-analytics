// *****************************************************************************
// OpenCV : Contract Limited Adaptive Histogram Equaliztion (CLAHE) Interface

// Author : Toby Breckon, toby.breckon@cranfield.ac.uk

// Copyright (c) 2009 School of Engineering, Cranfield University
// License : LGPL - http://www.gnu.org/licenses/lgpl.html

#include <cv.h>       // open cv general include file

// *****************************************************************************

// CLAHE input/output range flag definitions

#define CV_CLAHE_RANGE_FULL 0
#define CV_CLAHE_RANGE_INPUT 1

// *****************************************************************************

// cvAdaptEqualize(src, dst, xdivs, ydivs, bins)
//
// src - pointer to source image (must be single channel 8-bit)
// dst - pointer to destination image (must be single channel 8-bit)
// xdivs - number of cell divisions to use in vertical (x) direction (MIN=2 MAX = 16)
// ydivs - number of cell divisions to use in vertical (y) direction (MIN=2 MAX = 16)
// bins - number of histogram bins to use per division
// range - either of CV_CLAHE_RANGE_INPUT or CV_CLAHE_RANGE_FULL to limit the output
//         pixel range to the min/max range of the input image or the full range of the
//         pixel depth (8-bit in this case)

void cvAdaptEqualize(IplImage *src, IplImage *dst, 
			unsigned int xdivs, unsigned int ydivs, unsigned int bins, int range);

// cvCLAdaptEqualize(src, dst, xdivs, ydivs, bins, limit)
//
// src - pointer to source image (must be single channel 8-bit)
// dst - pointer to destination image (must be single channel 8-bit)
// xdivs - number of cell divisions to use in vertical (x) direction (MIN=2 MAX = 16)
// ydivs - number of cell divisions to use in vertical (y) direction (MIN=2 MAX = 16)
// bins - number of histogram bins to use per division
// limit - contrast limit for localised changes in contrast
// range - either of CV_CLAHE_RANGE_INPUT or CV_CLAHE_RANGE_FULL to limit the output
//         pixel range to the min/max range of the input image or the full range of the
//         pixel depth (8-bit in this case)


void cvCLAdaptEqualize(IplImage *src, IplImage *dst, 
			unsigned int xdivs, unsigned int ydivs, unsigned int bins, float limit,
			int range);
	
// *****************************************************************************
	
/*
  redefine : CV_ERROR_LOCAL macro unconditionally raises error with passed code and message.
  After raising error, control will be transferred to the exit label.
*/

#define CV_ERROR_LOCAL( Code, Msg )                                       \
{                                                                   \
     cvError( (Code), "CLAHE code", Msg, __FILE__, __LINE__ );       \
     exit(1);                                                          \
}

			
// *****************************************************************************
