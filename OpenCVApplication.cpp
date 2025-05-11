// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h" // doar pt windows
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <algorithm>
#include <random>
#include <queue>
#include <stack>
#include <fstream>

wchar_t* projectPath;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

// Function to perform color segmentation in HSV space
Mat colorSegmentation(const Mat& input, int hueMin, int hueMax,
    int satMin, int satMax, int valMin, int valMax) {
    Mat hsvImg, mask;

    // Convert from BGR to HSV color space
    cvtColor(input, hsvImg, COLOR_BGR2HSV);

    // Create the threshold for color segmentation
    Scalar lowerBound(hueMin, satMin, valMin);
    Scalar upperBound(hueMax, satMax, valMax);

    // Create binary mask
    inRange(hsvImg, lowerBound, upperBound, mask);

    // Apply some morphological operations to remove noise
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);

    return mask;
}

// Function to segment multiple colors and combine masks
void detectTrafficSignsByColor(const Mat& input) {
    // Create windows for visualization
    namedWindow("Original", WINDOW_AUTOSIZE);
    namedWindow("Red Mask", WINDOW_AUTOSIZE);
    namedWindow("Blue Mask", WINDOW_AUTOSIZE);
    namedWindow("Yellow Mask", WINDOW_AUTOSIZE);

    // Display original image
    imshow("Original", input);

    // Red color segmentation (note: red in HSV wraps around 0/180)
    Mat redMask1 = colorSegmentation(input, 0, 10, 100, 255, 50, 255);   // Lower red range
    Mat redMask2 = colorSegmentation(input, 160, 180, 100, 255, 50, 255); // Upper red range
    Mat redMask = redMask1 | redMask2;  // Combine both red ranges
    imshow("Red Mask", redMask);

    // Blue color segmentation
    Mat blueMask = colorSegmentation(input, 90, 130, 50, 255, 50, 255);
    imshow("Blue Mask", blueMask);

    // Yellow color segmentation
    Mat yellowMask = colorSegmentation(input, 20, 40, 100, 255, 100, 255);
    imshow("Yellow Mask", yellowMask);

    // Apply masks to original image to visualize segmented regions
    Mat redResult, blueResult, yellowResult;
    bitwise_and(input, input, redResult, redMask);
    bitwise_and(input, input, blueResult, blueMask);
    bitwise_and(input, input, yellowResult, yellowMask);

    imshow("Red Signs", redResult);
    imshow("Blue Signs", blueResult);
    imshow("Yellow Signs", yellowResult);

    waitKey();
}

// Modified test function to process images
void testTrafficSignDetection()
{
    char fname[MAX_PATH];
    while (openFileDlg(fname))
    {
        Mat src = imread(fname);
        if (src.empty()) {
            printf("Could not open or find the image\n");
            continue;
        }

        // Process the image
        detectTrafficSignsByColor(src);
    }
}

int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	projectPath = _wgetcwd(0, 0);

	//testOpenImage();
    testTrafficSignDetection();
	return 0;
}