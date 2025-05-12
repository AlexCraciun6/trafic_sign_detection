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

using namespace std;

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

// Function to perform edge detection using Canny
Mat cannyEdgeDetection(const Mat& input, double threshold1, double threshold2, int apertureSize = 3) {
    Mat grayImage, edges;

    // Convert to grayscale if image is colored
    if (input.channels() > 1) {
        cvtColor(input, grayImage, COLOR_BGR2GRAY);
    }
    else {
        grayImage = input.clone();
    }

    // Apply Gaussian blur to reduce noise
    GaussianBlur(grayImage, grayImage, Size(5, 5), 0);

    // Apply Canny edge detector
    Canny(grayImage, edges, threshold1, threshold2, apertureSize);

    return edges;
}

// Function to perform edge detection using Sobel
Mat sobelEdgeDetection(const Mat& input, int dx, int dy, int ksize = 3) {
    Mat grayImage, gradX, gradY, absGradX, absGradY, edges;

    // Convert to grayscale if image is colored
    if (input.channels() > 1) {
        cvtColor(input, grayImage, COLOR_BGR2GRAY);
    }
    else {
        grayImage = input.clone();
    }

    // Apply Gaussian blur to reduce noise
    GaussianBlur(grayImage, grayImage, Size(5, 5), 0);

    // Compute gradients in x and y direction
    Sobel(grayImage, gradX, CV_16S, dx, 0, ksize);
    Sobel(grayImage, gradY, CV_16S, 0, dy, ksize);

    // Convert to absolute values
    convertScaleAbs(gradX, absGradX);
    convertScaleAbs(gradY, absGradY);

    // Combine gradients
    addWeighted(absGradX, 0.5, absGradY, 0.5, 0, edges);

    return edges;
}

// Function to detect contours from edge images or binary masks
void detectAndDrawContours(const Mat& original, const Mat& binaryImage, const String& windowName) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    // Find contours in the binary image
    findContours(binaryImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Create a colored image for displaying contours
    Mat contourImage = original.clone();

    // Filter contours based on area to remove noise
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area < 100) continue; // Skip small contours

        // Draw contour with a random color
        Scalar color = Scalar(0, 255, 0); // Green contours
        drawContours(contourImage, contours, (int)i, color, 2, LINE_8, hierarchy);

        // Get bounding rectangle for the contour
        Rect boundRect = boundingRect(contours[i]);
        rectangle(contourImage, boundRect, Scalar(255, 0, 0), 2); // Blue rectangle
    }

    // Display result
    imshow(windowName, contourImage);
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


// Improved function to identify traffic sign shapes from contours
void detectShapes(const Mat& input, const Mat& colorMask, const String& windowName, const String& colorName) {
    // Find contours in the color mask
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(colorMask.clone(), contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Create image for drawing results
    Mat shapesImage = input.clone();

    // Process each contour
    for (size_t i = 0; i < contours.size(); i++) {
        // Filter small contours
        double area = contourArea(contours[i]);
        if (area < 100) continue;

        // Get the minimum enclosing circle
        Point2f center;
        float radius;
        minEnclosingCircle(contours[i], center, radius);

        // Calculate circularity - how close to a perfect circle the shape is
        double perimeter = arcLength(contours[i], true);
        double circularity = 4 * CV_PI * area / (perimeter * perimeter);

        // Approximate contour with polygon
        vector<Point> approx;
        double epsilon = 0.03 * perimeter;
        approxPolyDP(contours[i], approx, epsilon, true);

        // Get the number of vertices in the polygon
        int vertices = (int)approx.size();

        // Get bounding rect for text placement
        Rect boundRect = boundingRect(approx);
        Point textPoint(boundRect.x, boundRect.y - 10);

        // Get some shape properties for better classification
        double width = boundRect.width;
        double height = boundRect.height;
        double aspectRatio = width / height;

        // Classify shape based on number of vertices and other properties
        String shapeName;
        Scalar color;
        String signType = "Unknown";

        if (circularity > 0.8 && vertices >= 8) {
            shapeName = "Circle";
            color = Scalar(255, 0, 255); // Magenta

            // Determine sign type based on color
            if (colorName == "Red") {
                signType = "Prohibition";
            }
            else if (colorName == "Blue") {
                signType = "Information";
            }
            else if (colorName == "Yellow") {
                signType = "Warning";
            }
        }
        else if (vertices == 3) {
            shapeName = "Triangle";
            color = Scalar(0, 255, 0); // Green
            signType = "Warning";
        }
        else if (vertices == 4) {
            if (aspectRatio >= 0.9 && aspectRatio <= 1.1) {
                shapeName = "Square";
            }
            else {
                shapeName = "Rectangle";
            }
            color = Scalar(0, 0, 255); // Red
            signType = "Information";
        }
        else if (vertices == 8) {
            shapeName = "Octagon";
            color = Scalar(255, 0, 0); // Blue
            signType = "Stop";
        }
        else {
            shapeName = "Other";
            color = Scalar(255, 255, 0); // Yellow
        }

        // Draw contour
        drawContours(shapesImage, contours, (int)i, color, 2);

        // Draw shape name
        putText(shapesImage, shapeName, Point(boundRect.x, boundRect.y - 20),
            FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

        // Draw sign type
        putText(shapesImage, signType, Point(boundRect.x, boundRect.y - 5),
            FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

        // Draw bounding box for clarity
        rectangle(shapesImage, boundRect, Scalar(0, 255, 255), 2);

        // Draw minimum enclosing circle for non-circular objects to visualize fit
        if (shapeName != "Circle" && shapeName != "Octagon") {
            circle(shapesImage, center, (int)radius, Scalar(128, 128, 128), 1);
        }
    }

    imshow(windowName, shapesImage);
}


// Enhanced function to segment and detect traffic signs
void detectTrafficSignsByColorAndShape(const Mat& input) {
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

    // Apply edge detection to the original image
    Mat cannyEdges = cannyEdgeDetection(input, 50, 150);
    imshow("Canny Edges", cannyEdges);

    // Apply shape detection on each color mask, passing the color name
    detectShapes(input, redMask, "Red Shapes", "Red");
    detectShapes(input, blueMask, "Blue Shapes", "Blue");
    detectShapes(input, yellowMask, "Yellow Shapes", "Yellow");

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
        detectTrafficSignsByColorAndShape(src);
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