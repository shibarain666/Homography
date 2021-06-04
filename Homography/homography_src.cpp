/**************************************************
File: homography_src.cpp

Author: Jerry Cheng

Date: 2021/06/03

***************************************************/

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

Mat find_H_matrix(vector<Point2f> src, vector<Point2f> tgt) {

	double Point_matrix[8][8];
	for (int i = 0; i < 4; i++) {
		double row1[8] = { src[i].x, src[i].y, 1,        0,        0, 0, -tgt[i].x * src[i].x, -tgt[i].x * src[i].y };
		double row2[8] = { 0       ,        0, 0, src[i].x, src[i].y, 1, -tgt[i].y * src[i].x, -tgt[i].y * src[i].y };

		/* save 4 point data to 8*8 Matrix */
		memcpy(Point_matrix[2 * i], row1, sizeof(row1));
		memcpy(Point_matrix[2 * i + 1], row2, sizeof(row2));
	}

	/* 8*8 Matrix */
	Mat Point_data(8, 8, CV_64FC1, Point_matrix); 
	Mat Point_target = (Mat_<double>(8, 1) << tgt[0].x, tgt[0].y, tgt[1].x, tgt[1].y, tgt[2].x, tgt[2].y, tgt[3].x, tgt[3].y);
	Mat h8 = Point_data.inv() * Point_target;

	double H8[9]; /* 3*3 H Matrix */
	for (int i = 0; i < 8; i++) {
		H8[i] = h8.at<double>(i, 0);
	}
	H8[8] = 1.0;
	Mat H(3, 3, CV_64FC1, H8);
	return H.clone();
}

Mat do_transform(Mat src, Mat H) {

	Mat tgt(768, 1024, CV_8UC3, Scalar(0, 0, 0));
	Mat X, Xresult;

	/* Target-to-Source */
	for (int i = 0; i < tgt.rows; i++) {
		for (int j = 0; j < tgt.cols; j++) {
			Xresult = (Mat_<double>(3, 1) << j, i, 1.0);
			X = H.inv()*Xresult;
			int x = cvRound(X.at<double>(0, 0) / X.at<double>(2, 0));    /* normalized */
			int y = cvRound(X.at<double>(1, 0) / X.at<double>(2, 0));

			if (x < 0 || y < 0) {
				continue;
			}
			if (x > src.cols - 1 || y > src.rows - 1) {
				continue;
			}

			tgt.at<Vec3b>(i, j) = src.at<Vec3b>(y, x);
		}
	}

	return tgt.clone();
}

int main(int argc, char *argv[]) {

	Mat img_ori = imread("origin_pic.JPG");
	Mat img_gt = imread("GroundTruth.JPG");
	Mat img_homo(img_gt.rows, img_gt.cols, CV_8UC3, Scalar(0, 0, 0));

	/* Correspondence Points */
	vector<Point2f> img_ori_points = { Point2f(559, 529), Point2f(2041, 349), Point2f(573, 1733), Point2f(2053, 1887) }; /* 4 sets of points on source image */
	vector<Point2f> img_homo_points = { Point2f(0, 0), Point2f(1024, 0), Point2f(0, 768), Point2f(1024, 768) };          /* 4 sets of points on target image */

	/* OpenCV's API, for comparison use */
	Mat H_opencv = findHomography(img_ori_points, img_homo_points);

	double t = (double)getTickCount();

	/* my self-implemented function for finding H */
	Mat H = find_H_matrix(img_ori_points, img_homo_points);
	img_homo = do_transform(img_ori, H);

	t = (double)getTickCount() - t;
	cout << "time:" << t/(getTickFrequency()) << endl;

	cout << "Self-implemented H matrix:" << endl;
	cout << H << endl;
	cout << "OpenCV H matrix:" << endl;
	cout << H_opencv << endl;

	imwrite("Homography.JPG", img_homo);
	imshow("After homography", img_homo);

	Mat img_diff = img_gt - img_homo;

	imwrite("Diff.BMP", img_diff);
	imshow("Diff", img_diff);

	waitKey(0);
	return 0;
}