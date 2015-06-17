/*
 Copyright 2010 Computer Vision Lab,
 Ecole Polytechnique Federale de Lausanne (EPFL), Switzerland.
 All rights reserved.

 Authors: Eray Molla, Michael Calonder, and Vincent Lepetit

 This file is part of the BRIEF_demo software.

 BRIEF_demo is  free software; you can redistribute  it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free  Software Foundation; either  version 2 of the  License, or
 (at your option) any later version.

 BRIEF_demo is  distributed in the hope  that it will  be useful, but
 WITHOUT  ANY   WARRANTY;  without  even  the   implied  warranty  of
 MERCHANTABILITY  or FITNESS FOR  A PARTICULAR  PURPOSE. See  the GNU
 General Public License for more details.

 You should  have received a copy  of the GNU  General Public License
 along  with   BRIEF_demo;  if  not,  write  to   the  Free  Software
 Foundation,  Inc.,  51  Franklin  Street, Fifth  Floor,  Boston,  MA
 02110-1301, USA
 */

#include "BRIEF.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <bitset>
#include <fstream>

using namespace std;
using namespace cv;

int matchDescriptors(CvMat& match1, CvMat& match2,
		const vector<bitset<CVLAB::DESC_LEN> > descs1,
		const vector<bitset<CVLAB::DESC_LEN> > descs2,
		const vector<cv::KeyPoint>& kpts1, const vector<cv::KeyPoint>& kpts2);

void showMatches(const int matchCount);

void putImagesSideBySide(IplImage* result, const IplImage* img1,
		const IplImage* img2);

// Frame width and height of the capture
static const int FRAME_WIDTH = 640;
static const int FRAME_HEIGHT = 480;

IplImage* sideBySideImage;

// Maximum number of keypoint matches allowed by the program
static const int MAXIMUM_NUMBER_OF_MATCHES = 5000;

// Minimum  scale ratio of  the program.  It indicates  that templates
// having scales  [0.5, 1]  of the original  template will be  used to
// generate new templates. Scales are determined in a logarithm base.
static const float SMALLEST_SCALE_CHANGE = 0.5;

// Number of different scales used to generate the templates.
static const int NUMBER_OF_SCALE_STEPS = 3;

// Number  of   different  rotation   angles  used  to   generate  the
// templates. 18 indicates that  [0 20 ... 340] degree-rotated samples
// will be stored in the 2-D array for each scale.
static const int NUMBER_OF_ROTATION_STEPS = 18;

double match1Data[2 * MAXIMUM_NUMBER_OF_MATCHES];
double match2Data[2 * MAXIMUM_NUMBER_OF_MATCHES];

// Generates new templates with different scales and orientations and stores their keypoints and
// BRIEF descriptors.

ofstream myfile;

int main(int argc, char **argv) {

	vector<KeyPoint> kpt1, kpt2;
	Mat desc1, desc2;
	Mat img1 = imread("LENNA2.jpg", -1);
	Mat img2 = imread("LENNA.jpg", -1);

	if (img1.empty() || img2.empty()) {
		cout << "Could not read one of the two images !" << endl;
		return EXIT_FAILURE;
	}

	Mat img11(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);
	Mat img22(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);

	resize(img1, img11, img11.size(), 0, 0, CV_INTER_LINEAR);
	resize(img2, img22, img22.size(), 0, 0, CV_INTER_LINEAR);

	OrbFeatureDetector detector;
	detector.detect(img11, kpt1);
	detector.detect(img22, kpt2);

	myfile.open("keypoints.txt");
	for (int i = 0; i < kpt1.size(); i++)
		myfile << kpt1[i].pt << endl;
	myfile.close();

	drawKeypoints(img11, kpt1, img11, Scalar(255, 0, 0));
	drawKeypoints(img22, kpt2, img22, Scalar(255, 0, 0));
	/*namedWindow("finalWithOrb");
	 imshow("finalWithOrb", img1);
	 imshow("finalWithOrb", img2);
	 waitKey(0);*/

	detector.compute(img11, kpt1, desc1);
	detector.compute(img22, kpt2, desc2);
	cout << desc1 << endl;
	// 32 x 429
	//cout << "Size : " << desc1.size() << endl;
	//cout << "Size : " << desc2.size() << endl;

	CVLAB::BRIEF brief;
	//bitset<256> descriptor;

	IplImage* image1;
	IplImage* image2;

	/* Convert to one channel image */
	cvtColor(img11, img11, CV_BGR2GRAY, 1);
	cvtColor(img22, img22, CV_BGR2GRAY, 1);

	//cout << " columns : " << img1.cols << endl;
	//cout << " rows : " << img1.rows << endl;

	image1 = cvCreateImage(cvSize(img11.cols, img11.rows), 8, 1);
	image2 = cvCreateImage(cvSize(img22.cols, img22.rows), 8, 1);

	IplImage ipltemp1 = img11;
	IplImage ipltemp2 = img22;
	cvCopy(&ipltemp1, image1);
	cvCopy(&ipltemp2, image2);

	IplImage *img1resized = cvCreateImage(cvSize(FRAME_WIDTH, FRAME_HEIGHT),
	IPL_DEPTH_8U, 1);
	IplImage *img2resized = cvCreateImage(cvSize(FRAME_WIDTH, FRAME_HEIGHT),
	IPL_DEPTH_8U, 1);
	cvResize(image1, img1resized, CV_INTER_LINEAR);
	cvResize(image2, img2resized, CV_INTER_LINEAR);

	IplImage *img1resizedSmooth = cvCreateImage(cvSize(FRAME_WIDTH, FRAME_HEIGHT),
	IPL_DEPTH_8U, 1);
	cvSmooth(img2resized, img1resizedSmooth, CV_GAUSSIAN, 3, 0, 1.0, 0.0);
	cvShowImage("resizedSmooth", img1resizedSmooth);
	//cvShowImage("resized2", img2resized);

	vector<bitset<256> > descriptors;
	vector<bitset<256> > descriptors2;
	brief.getBriefDescriptors(descriptors, kpt1, img1resized);
	brief.getBriefDescriptors(descriptors2, kpt2, img2resized);

	//cout << descriptors << endl;
	cout << "Keypoints Size : " << kpt1.size() << endl;
	cout << "Descripteur Size : " << descriptors.size() << endl;

	BFMatcher matcher;
	vector<DMatch> matches;
	matcher.match(desc1, desc2, matches);
	Mat output;
	drawMatches(img11, kpt1, img22, kpt2, matches, output);
	imshow("ouptutORB", output);
	waitKey(0);
	/* Convert vector to Mat for the matching */
	//Mat descriptor11(descriptors, false);
	//Mat descriptor22(descriptors2, false);
	//cout << "OK" << endl;
	Mat imgFinal;
	CvMat match1;
	CvMat match2;

	const int nbMatches = matchDescriptors(match1, match2, descriptors,
			descriptors2, kpt1, kpt2);

	cout << "NbMatches : " << nbMatches << endl;
	//cout << match1.rows << endl;
	//putImagesSideBySide("Hellor", 2, image1, image2);
	sideBySideImage = cvCreateImage(cvSize(FRAME_WIDTH * 2, FRAME_HEIGHT),
	IPL_DEPTH_8U, 1);
	putImagesSideBySide(sideBySideImage, img1resized, img2resized);

	showMatches(nbMatches);

	waitKey(0);

	return EXIT_SUCCESS;
}

int matchDescriptors(CvMat& match1, CvMat& match2,
		const vector<bitset<CVLAB::DESC_LEN> > descs1,
		const vector<bitset<CVLAB::DESC_LEN> > descs2,
		const vector<cv::KeyPoint>& kpts1, const vector<cv::KeyPoint>& kpts2) {
	static const int MAX_MATCH_DISTANCE = 50;
	int numberOfMatches = 0;
	int bestMatchInd2 = 0;

	cout << "Taille Desc1 : " << descs1.size() << endl;
	cout << "Taille Desc2 : " << descs2.size() << endl;

	for (unsigned int i = 0;
			i < descs1.size() && numberOfMatches < MAXIMUM_NUMBER_OF_MATCHES;
			i++) {
		int minDist = CVLAB::DESC_LEN;
		for (unsigned int j = 0; j < descs2.size(); j++) {

			/*cout << "desc1 : " << descs1[i] << endl;
			 cout << "desc2 : " << descs2[j] << endl;*/
			const int dist = CVLAB::HAMMING_DISTANCE(descs1[i], descs2[j]);
			//cout << "distance : " << dist << endl;
			//cout << "dist : " << dist << endl;
			//cout << "descs1 : " << descs1[i] << " descs2 : " << descs2[j] << endl;

			if (dist < minDist) {
				minDist = dist;
				bestMatchInd2 = j;
			}
		}
		if (minDist > MAX_MATCH_DISTANCE) {
			//continue;
		}
		const int xInd = 2 * numberOfMatches;
		const int yInd = xInd + 1;
		match1Data[xInd] = kpts1[i].pt.x;
		match1Data[yInd] = kpts1[i].pt.y;

		/*cout << "Matching en X : " << match1Data[xInd] << endl;
		 cout << "Matching en Y : " << match1Data[yInd] << endl;*/

		match2Data[xInd] = kpts2[bestMatchInd2].pt.x;
		match2Data[yInd] = kpts2[bestMatchInd2].pt.y;

		cout << "match1DataX : " << match1Data[xInd] << endl;
		cout << "match1DataY : " << match1Data[yInd] << endl;
		cout << "match2DataX : " << match2Data[xInd] << endl;
		cout << "match2DataY : " << match2Data[yInd] << endl;

		numberOfMatches++;

	}

	if (numberOfMatches > 0) {
		cvInitMatHeader(&match1, numberOfMatches, 2, CV_64FC1, match1Data);
		cvInitMatHeader(&match2, numberOfMatches, 2, CV_64FC1, match2Data);

	}

	return numberOfMatches;
}

// Puts img1 and img2 side by side and stores into result
void putImagesSideBySide(IplImage* result, const IplImage* img1,
		const IplImage* img2) {
	// widthStep of the resulting image
	const int bigWS = result->widthStep;
	cout << "BigWS : " << bigWS << endl;
	// half of the widthStep of the resulting image
	const int bigHalfWS = result->widthStep >> 1;
	// widthStep of the image which will be put in the left
	const int lWS = img1->widthStep;
	// widthStep of the image which will be put in the right
	const int rWS = img2->widthStep;

	// pointer to the beginning of the left image
	char *p_big = result->imageData;
	// pointer to the beginning of the right image
	char *p_bigMiddle = result->imageData + bigHalfWS;
	// pointer to the image data which will be put in the left
	const char *p_l = img1->imageData;
	// pointer to the image data which will be put in the right
	const char *p_r = img2->imageData;

	cout << "imgWidth : " << img1->width << endl;

	for (int i = 0; i < FRAME_HEIGHT;
			++i, p_big += bigWS, p_bigMiddle += bigWS) {
		// copy a row of the left image till the half of the resulting image
		memcpy(p_big, p_l + i * lWS, lWS);
		// copy a row of the right image from the half of the resulting image to the end of it
		memcpy(p_bigMiddle, p_r + i * rWS, rWS);
	}
}

void showMatches(const int matchCount) {
	const int iterationEnd = 2 * matchCount;

	IplImage* img = 0;
	//myfile.open("matches.txt");
	//for (int i = 0; i < 2 * matchCount; i++)

	myfile.open("matchesTotal.txt");
	//for (int j = 0; j < 2 * matchCount; j++)
	//myfile << match2Data[j] << endl;
	for (int xCoor = 0, yCoor = 1; xCoor < iterationEnd; xCoor += 2, yCoor += 2) {
		myfile << " PointData1.X : " << match1Data[xCoor] << " PointData1.Y : "
				<< match1Data[yCoor] << endl << " PointData2.X : "
				<< match2Data[xCoor] << " PointData2.Y : " << match2Data[yCoor]
				<< endl;
		// Draw a line between matching keypoints
		cvLine(sideBySideImage, cvPoint(match1Data[xCoor], match1Data[yCoor]),
				cvPoint(match2Data[xCoor] + FRAME_WIDTH, match2Data[yCoor]),
				cvScalar(0, 255, 0), 1);
	}
	myfile.close();
	cvShowImage("test", sideBySideImage);
	cvWaitKey(0);
}
