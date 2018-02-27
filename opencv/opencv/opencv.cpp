// opencv.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <iostream>  
#include <windows.h>  
 
using namespace std;  
using namespace cv; 

//四边形四角点逆时针排序
void RankPoint(vector<Point2f> &corners, vector<int> &vecIndex)
{
	CvMat* p=cvCreateMat(3,4,CV_32FC1);//图像上点的矩阵,竖着依次为u,v,1
    cvmSet(p,0,0,corners[0].x); cvmSet(p,0,1,corners[1].x); cvmSet(p,0,2,corners[2].x); cvmSet(p,0,3,corners[3].x);
    cvmSet(p,1,0,corners[0].y); cvmSet(p,1,1,corners[1].y); cvmSet(p,1,2,corners[2].y); cvmSet(p,1,3,corners[3].y);
    cvmSet(p,2,0,1);  cvmSet(p,2,1,1);  cvmSet(p,2,2,1);  cvmSet(p,2,3,1);

    CvMat* p_dst=cvCreateMat(3,4,CV_32FC1);//图像上点的矩阵,竖着依次为u,v,1
    double a[2][4];
    a[0][0]=cvmGet(p,0,0); a[0][1]=cvmGet(p,0,1); a[0][2]=cvmGet(p,0,2); a[0][3]=cvmGet(p,0,3);
    a[1][0]=cvmGet(p,1,0); a[1][1]=cvmGet(p,1,1); a[1][2]=cvmGet(p,1,2); a[1][3]=cvmGet(p,1,3);
    //cout<<a[0][2]<<endl;
    double b;
    for(int j=3;j>1;j--)
    {
        for(int i=0;i<j;i++)
        {
            if(a[0][i]>=a[0][i+1])
            {
                b=a[0][i];
                a[0][i]=a[0][i+1];
                a[0][i+1]=b;

                b=a[1][i];
                a[1][i]=a[1][i+1];
                a[1][i+1]=b;
            }
        }
    }

    for(int i=0;i<4;i=i+2)
    {
        if(a[1][i]>=a[1][i+1])
        {
            b=a[0][i];
            a[0][i]=a[0][i+1];
            a[0][i+1]=b;

            b=a[1][i];
            a[1][i]=a[1][i+1];
            a[1][i+1]=b;
        }
    }
    cvmSet(p_dst,0,0,a[0][1]); cvmSet(p_dst,0,1,a[0][3]); cvmSet(p_dst,0,2,a[0][2]); cvmSet(p_dst,0,3,a[0][0]);
    cvmSet(p_dst,1,0,a[1][1]); cvmSet(p_dst,1,1,a[1][3]); cvmSet(p_dst,1,2,a[1][2]); cvmSet(p_dst,1,3,a[1][0]);
    cvmSet(p_dst,2,0,1);       cvmSet(p_dst,2,1,1);       cvmSet(p_dst,2,2,1);       cvmSet(p_dst,2,3,1);

	vecIndex.clear();
	for(int k=0;k<4;k++)
	{
		for(int m=0;m<4;m++)
		{
			if(cvmGet(p, 0,k) == cvmGet(p_dst, 0,m) && cvmGet(p, 1,k) == cvmGet(p_dst, 1,m))
				vecIndex.push_back(m);
		}
	}
	/*------释放内存------*/
    cvReleaseMat(&p_dst);
	cvReleaseMat(&p);
    /*------------*/
}

//透视变换
Mat PerspectiveTrans(Mat src, Point2f* scrPoints, Point2f* dstPoints, Size dstSize)
{
	Mat dst;
	Mat Trans = getPerspectiveTransform(scrPoints, dstPoints);
	warpPerspective(src, dst, Trans, dstSize, CV_INTER_CUBIC);
	return dst;
}

//截取灰度图面积最大轮廓矩面积
void FindContourMaxArea(Mat &grayImg, Mat &dst)
{
	vector<vector<Point> > contours;   //轮廓数组 
	vector<Point2d>  centers;    //轮廓质心坐标   
	vector<vector<Point> >::iterator itr;  //轮廓迭代器  
	vector<Point2d>::iterator  itrc;    //质心坐标迭代器  
	vector<vector<Point> > con;    //当前轮廓  
	double area;  
	double minarea = 100;  
	double maxarea = 0;  
	Moments mom; // 轮廓矩  
	Mat edge; 

	edge = grayImg.clone();
	//blur(edge, edge, Size(3,3));   //模糊去噪  
	threshold(edge,edge,130,255,THRESH_BINARY_INV);   //二值化处理  

	/*寻找轮廓*/  
	findContours( edge, contours,  
		CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );  
	itr = contours.begin();     //使用迭代器去除噪声轮廓  
	while(itr!=contours.end())  
	{  
		area = contourArea(*itr);  
		if(area<minarea)  
		{  
			itr = contours.erase(itr);  //itr一旦erase，需要重新赋值  
		}  
		else  
		{  
			itr++;  
		}  
		if (area>maxarea)  
		{  
			maxarea = area;  
		}  
	}  
	dst = Mat::zeros(grayImg.rows,grayImg.cols,CV_8UC1);  

	/*绘制连通区域轮廓，计算质心坐标*/  
	Point2d center;  
	itr = contours.begin();  
	while(itr!=contours.end())  
	{  
		area = contourArea(*itr);  
		con.push_back(*itr);  
		if(area==maxarea)  
			drawContours(dst,con,-1,Scalar(255),10);  //最大面积白色绘制  
		else  
			//drawContours(dst,con,-1,Scalar(255,0,0),2);   //其它面积蓝色绘制  
			con.pop_back();  

		//计算质心  
		mom = moments(*itr);  
		center.x = (int)(mom.m10/mom.m00);  
		center.y = (int)(mom.m01/mom.m00);  
		centers.push_back(center);  

		itr++;  
	}  	
}

//截取灰度图面积最大轮廓长度
void FindContourMaxLength(Mat &grayImg, Mat &dst)
{
	vector<vector<Point> > contours;   //轮廓数组 
	vector<Point2d>  centers;    //轮廓质心坐标   
	vector<vector<Point> >::iterator itr;  //轮廓迭代器  
	vector<Point2d>::iterator  itrc;    //质心坐标迭代器  
	vector<vector<Point> > con;    //当前轮廓  
	double area;  
	double minarea = 100;  
	double maxarea = 0;  
	Moments mom; // 轮廓矩  
	Mat edge; 

	edge = grayImg.clone();
	//blur(edge, edge, Size(3,3));   //模糊去噪  
	threshold(edge,edge,160,255,THRESH_BINARY_INV);   //二值化处理  

	/*寻找轮廓*/  
	findContours( edge, contours,  
		CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );  
	itr = contours.begin();     //使用迭代器去除噪声轮廓  
	while(itr!=contours.end())  
	{  
		area = arcLength(*itr, true); 
		if(area<minarea)  
		{  
			itr = contours.erase(itr);  //itr一旦erase，需要重新赋值  
		}  
		else  
		{  
			itr++;  
		}  
		if (area>maxarea)  
		{  
			maxarea = area;  
		}  
	}  
	dst = Mat::zeros(grayImg.rows,grayImg.cols,CV_8UC1);  

	/*绘制连通区域轮廓，计算质心坐标*/  
	Point2d center;  
	itr = contours.begin();  
	while(itr!=contours.end())  
	{  
		area = arcLength(*itr, true);  
		con.push_back(*itr);  
		if(area==maxarea)  
			drawContours(dst,con,-1,Scalar(255),10);  //最大面积白色绘制  
		else  
			//drawContours(dst,con,-1,Scalar(255,0,0),2);   //其它面积蓝色绘制  
			con.pop_back();  

		//计算质心  
		mom = moments(*itr);  
		center.x = (int)(mom.m10/mom.m00);  
		center.y = (int)(mom.m01/mom.m00);  
		centers.push_back(center);  

		itr++;  
	}  	
}

//彩图亚像素检测ROI角点
void ROIcornerSubPix(Mat src, Mat roi, int maxCorners, vector<Point2f> &corners)
{
	//int maxCorners = 4;    //角点个数的最大值   
	/// Shi-Tomasi的参数设置  
	double qualityLevel = 0.01;  
	double minDistance = 200;//相邻点的最小距离  
	int blockSize = 5;  
	bool useHarrisDetector = false;   //不使用Harris检测算法  
	double k = 0.04;  
	Mat srcGray;
	corners.clear();
	cvtColor(src, srcGray, COLOR_BGR2GRAY);

	/// 应用Shi-Tomasi角点检测算法   
	goodFeaturesToTrack( srcGray,   
		corners,  
		maxCorners,  
		qualityLevel,  
		minDistance,  
		roi,   //未选择感兴趣区域   
		blockSize,  
		useHarrisDetector,  
		k );  
	/// 角点位置精准化参数  
	Size winSize = Size( 5, 5 );  
	Size zeroZone = Size( -1, -1 );  
	TermCriteria criteria = TermCriteria(   
		CV_TERMCRIT_EPS + CV_TERMCRIT_ITER,   
		40, //maxCount=40  
		0.04 );  //epsilon=0.001  
	/// 计算精准化后的角点位置  
	cornerSubPix( srcGray, corners, winSize, zeroZone, criteria );  
}

//检测
void CheckLineOfCube(Mat src)
{
	//单通道像素 50W
	int picPix = 0;
	//正方形像素
	int cubeEdges = 500;
	//正方形实际长度mm
	float cubeLength = 173.0;
	//比例
	float ratio = cubeLength/cubeEdges;
	//实际线段长度mm
	float lineLength = 100;
	//是否画线
	bool isDrawLines = true;

	Mat srcStemp;  
	srcStemp = src.clone();
//多通道分离
	vector<Mat> channels;
	Mat imageBlueChannel;
	Mat imageGreenChannel;
	Mat imageRedChannel;
	split(srcStemp, channels);//分离色彩通道
	imageBlueChannel = channels.at(0);
	imageGreenChannel = channels.at(1);
	imageRedChannel = channels.at(2); 
	picPix = imageBlueChannel.rows*imageBlueChannel.cols;

//截取目标轮廓
	Mat dstContour;
	FindContourMaxArea(imageBlueChannel, dstContour);

//亚像素检测四边形角点
	vector<Point2f> corners;
	ROIcornerSubPix(srcStemp, dstContour, 4, corners);
	
	//四边形四角点逆时针排序
	vector<int> vecIndex;
	RankPoint(corners, vecIndex);

	/// 2--显示精准化后的角点  
	int r = 2;  
	for( int i = 0; i < corners.size(); i++ )  
	{  
		// 标示出角点  
		circle( srcStemp, corners[vecIndex[i]], r, Scalar(255,0,255), -1, 8, 0 );   		
	}

	if(isDrawLines)
	{
		line( srcStemp, corners[vecIndex[0]], corners[vecIndex[1]], Scalar( 0, 255, 0), r);
		line( srcStemp, corners[vecIndex[0]], corners[vecIndex[2]], Scalar( 0, 255, 0), r);
		line( srcStemp, corners[vecIndex[2]], corners[vecIndex[3]], Scalar( 0, 255, 0), r);
		line( srcStemp, corners[vecIndex[1]], corners[vecIndex[3]], Scalar( 0, 255, 0), r);
	}

//四边形透视变换
	Size cubeSize(cubeEdges, cubeEdges);
	Point2f AffinePoints0[4] = { corners[vecIndex[0]], corners[vecIndex[1]], corners[vecIndex[2]], corners[vecIndex[3]] };
	Point2f AffinePoints1[4] = { Point2f(0, 0), Point2f(0, cubeSize.height), Point2f(cubeSize.width, 0), Point2f(cubeSize.width, cubeSize.height) };
	Mat dst_perspective = PerspectiveTrans(src, AffinePoints0, AffinePoints1, cubeSize);

	imshow("srcStemp", srcStemp);

//分割出测量的线段
	Mat dstImage,grayImage;
	cvtColor(dst_perspective, dstImage, COLOR_BGR2GRAY);
	FindContourMaxLength(dstImage, dstContour);

//亚像素检测两个线段角点
	ROIcornerSubPix(dst_perspective, dstContour, 2, corners);

	/// 2--显示精准化后的角点  
	for( int i = 0; i < corners.size(); i++ )  
	{  
		// 标示出角点  
		circle( dst_perspective, corners[i], r, Scalar(255,0,255), -1, 8, 0 );   		
	}
	//绘画测量线段
	if(isDrawLines)
		line( dst_perspective, corners[0], corners[1], Scalar( 0, 255, 0), r);

	double distance;    
    distance = powf((corners[0].x - corners[1].x),2) + powf((corners[0].y - corners[1].y),2);    
    distance = sqrtf(distance)*ratio;  
	double actualError = lineLength - distance;
	//actualError = abs(actualError);

	//在图像中显示文本字符串
	putText(dst_perspective,cv::format("actualError(mm): %0.2f", actualError),Point(50,60),FONT_HERSHEY_SIMPLEX,1,Scalar(255,23,0),4,4);
	imshow("Result", dst_perspective); 
}

int _tmain(int argc, _TCHAR* argv[])
{
	Mat src,srcStemp;  
    src = imread("../../pic/line/box_in_scene3.bmp");

	try
	{
		CheckLineOfCube(src);
	}
	catch(...)
	{

	}

	waitKey();  

	return 0;
}



