// opencv.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <iostream>  
#include <windows.h>  
 
using namespace std;  
using namespace cv; 

//�ı����Ľǵ���ʱ������
void RankPoint(vector<Point2f> &corners, vector<int> &vecIndex)
{
	CvMat* p=cvCreateMat(3,4,CV_32FC1);//ͼ���ϵ�ľ���,��������Ϊu,v,1
    cvmSet(p,0,0,corners[0].x); cvmSet(p,0,1,corners[1].x); cvmSet(p,0,2,corners[2].x); cvmSet(p,0,3,corners[3].x);
    cvmSet(p,1,0,corners[0].y); cvmSet(p,1,1,corners[1].y); cvmSet(p,1,2,corners[2].y); cvmSet(p,1,3,corners[3].y);
    cvmSet(p,2,0,1);  cvmSet(p,2,1,1);  cvmSet(p,2,2,1);  cvmSet(p,2,3,1);

    CvMat* p_dst=cvCreateMat(3,4,CV_32FC1);//ͼ���ϵ�ľ���,��������Ϊu,v,1
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
	/*------�ͷ��ڴ�------*/
    cvReleaseMat(&p_dst);
	cvReleaseMat(&p);
    /*------------*/
}

//͸�ӱ任
Mat PerspectiveTrans(Mat src, Point2f* scrPoints, Point2f* dstPoints, Size dstSize)
{
	Mat dst;
	Mat Trans = getPerspectiveTransform(scrPoints, dstPoints);
	warpPerspective(src, dst, Trans, dstSize, CV_INTER_CUBIC);
	return dst;
}

//��ȡ�Ҷ�ͼ���������������
void FindContourMaxArea(Mat &grayImg, Mat &dst)
{
	vector<vector<Point> > contours;   //�������� 
	vector<Point2d>  centers;    //������������   
	vector<vector<Point> >::iterator itr;  //����������  
	vector<Point2d>::iterator  itrc;    //�������������  
	vector<vector<Point> > con;    //��ǰ����  
	double area;  
	double minarea = 100;  
	double maxarea = 0;  
	Moments mom; // ������  
	Mat edge; 

	edge = grayImg.clone();
	//blur(edge, edge, Size(3,3));   //ģ��ȥ��  
	threshold(edge,edge,130,255,THRESH_BINARY_INV);   //��ֵ������  

	/*Ѱ������*/  
	findContours( edge, contours,  
		CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );  
	itr = contours.begin();     //ʹ�õ�����ȥ����������  
	while(itr!=contours.end())  
	{  
		area = contourArea(*itr);  
		if(area<minarea)  
		{  
			itr = contours.erase(itr);  //itrһ��erase����Ҫ���¸�ֵ  
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

	/*������ͨ����������������������*/  
	Point2d center;  
	itr = contours.begin();  
	while(itr!=contours.end())  
	{  
		area = contourArea(*itr);  
		con.push_back(*itr);  
		if(area==maxarea)  
			drawContours(dst,con,-1,Scalar(255),10);  //��������ɫ����  
		else  
			//drawContours(dst,con,-1,Scalar(255,0,0),2);   //���������ɫ����  
			con.pop_back();  

		//��������  
		mom = moments(*itr);  
		center.x = (int)(mom.m10/mom.m00);  
		center.y = (int)(mom.m01/mom.m00);  
		centers.push_back(center);  

		itr++;  
	}  	
}

//��ȡ�Ҷ�ͼ��������������
void FindContourMaxLength(Mat &grayImg, Mat &dst)
{
	vector<vector<Point> > contours;   //�������� 
	vector<Point2d>  centers;    //������������   
	vector<vector<Point> >::iterator itr;  //����������  
	vector<Point2d>::iterator  itrc;    //�������������  
	vector<vector<Point> > con;    //��ǰ����  
	double area;  
	double minarea = 100;  
	double maxarea = 0;  
	Moments mom; // ������  
	Mat edge; 

	edge = grayImg.clone();
	//blur(edge, edge, Size(3,3));   //ģ��ȥ��  
	threshold(edge,edge,160,255,THRESH_BINARY_INV);   //��ֵ������  

	/*Ѱ������*/  
	findContours( edge, contours,  
		CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );  
	itr = contours.begin();     //ʹ�õ�����ȥ����������  
	while(itr!=contours.end())  
	{  
		area = arcLength(*itr, true); 
		if(area<minarea)  
		{  
			itr = contours.erase(itr);  //itrһ��erase����Ҫ���¸�ֵ  
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

	/*������ͨ����������������������*/  
	Point2d center;  
	itr = contours.begin();  
	while(itr!=contours.end())  
	{  
		area = arcLength(*itr, true);  
		con.push_back(*itr);  
		if(area==maxarea)  
			drawContours(dst,con,-1,Scalar(255),10);  //��������ɫ����  
		else  
			//drawContours(dst,con,-1,Scalar(255,0,0),2);   //���������ɫ����  
			con.pop_back();  

		//��������  
		mom = moments(*itr);  
		center.x = (int)(mom.m10/mom.m00);  
		center.y = (int)(mom.m01/mom.m00);  
		centers.push_back(center);  

		itr++;  
	}  	
}

//��ͼ�����ؼ��ROI�ǵ�
void ROIcornerSubPix(Mat src, Mat roi, int maxCorners, vector<Point2f> &corners)
{
	//int maxCorners = 4;    //�ǵ���������ֵ   
	/// Shi-Tomasi�Ĳ�������  
	double qualityLevel = 0.01;  
	double minDistance = 200;//���ڵ����С����  
	int blockSize = 5;  
	bool useHarrisDetector = false;   //��ʹ��Harris����㷨  
	double k = 0.04;  
	Mat srcGray;
	corners.clear();
	cvtColor(src, srcGray, COLOR_BGR2GRAY);

	/// Ӧ��Shi-Tomasi�ǵ����㷨   
	goodFeaturesToTrack( srcGray,   
		corners,  
		maxCorners,  
		qualityLevel,  
		minDistance,  
		roi,   //δѡ�����Ȥ����   
		blockSize,  
		useHarrisDetector,  
		k );  
	/// �ǵ�λ�þ�׼������  
	Size winSize = Size( 5, 5 );  
	Size zeroZone = Size( -1, -1 );  
	TermCriteria criteria = TermCriteria(   
		CV_TERMCRIT_EPS + CV_TERMCRIT_ITER,   
		40, //maxCount=40  
		0.04 );  //epsilon=0.001  
	/// ���㾫׼����Ľǵ�λ��  
	cornerSubPix( srcGray, corners, winSize, zeroZone, criteria );  
}

//���
void CheckLineOfCube(Mat src)
{
	//��ͨ������ 50W
	int picPix = 0;
	//����������
	int cubeEdges = 500;
	//������ʵ�ʳ���mm
	float cubeLength = 173.0;
	//����
	float ratio = cubeLength/cubeEdges;
	//ʵ���߶γ���mm
	float lineLength = 100;
	//�Ƿ���
	bool isDrawLines = true;

	Mat srcStemp;  
	srcStemp = src.clone();
//��ͨ������
	vector<Mat> channels;
	Mat imageBlueChannel;
	Mat imageGreenChannel;
	Mat imageRedChannel;
	split(srcStemp, channels);//����ɫ��ͨ��
	imageBlueChannel = channels.at(0);
	imageGreenChannel = channels.at(1);
	imageRedChannel = channels.at(2); 
	picPix = imageBlueChannel.rows*imageBlueChannel.cols;

//��ȡĿ������
	Mat dstContour;
	FindContourMaxArea(imageBlueChannel, dstContour);

//�����ؼ���ı��νǵ�
	vector<Point2f> corners;
	ROIcornerSubPix(srcStemp, dstContour, 4, corners);
	
	//�ı����Ľǵ���ʱ������
	vector<int> vecIndex;
	RankPoint(corners, vecIndex);

	/// 2--��ʾ��׼����Ľǵ�  
	int r = 2;  
	for( int i = 0; i < corners.size(); i++ )  
	{  
		// ��ʾ���ǵ�  
		circle( srcStemp, corners[vecIndex[i]], r, Scalar(255,0,255), -1, 8, 0 );   		
	}

	if(isDrawLines)
	{
		line( srcStemp, corners[vecIndex[0]], corners[vecIndex[1]], Scalar( 0, 255, 0), r);
		line( srcStemp, corners[vecIndex[0]], corners[vecIndex[2]], Scalar( 0, 255, 0), r);
		line( srcStemp, corners[vecIndex[2]], corners[vecIndex[3]], Scalar( 0, 255, 0), r);
		line( srcStemp, corners[vecIndex[1]], corners[vecIndex[3]], Scalar( 0, 255, 0), r);
	}

//�ı���͸�ӱ任
	Size cubeSize(cubeEdges, cubeEdges);
	Point2f AffinePoints0[4] = { corners[vecIndex[0]], corners[vecIndex[1]], corners[vecIndex[2]], corners[vecIndex[3]] };
	Point2f AffinePoints1[4] = { Point2f(0, 0), Point2f(0, cubeSize.height), Point2f(cubeSize.width, 0), Point2f(cubeSize.width, cubeSize.height) };
	Mat dst_perspective = PerspectiveTrans(src, AffinePoints0, AffinePoints1, cubeSize);

	imshow("srcStemp", srcStemp);

//�ָ���������߶�
	Mat dstImage,grayImage;
	cvtColor(dst_perspective, dstImage, COLOR_BGR2GRAY);
	FindContourMaxLength(dstImage, dstContour);

//�����ؼ�������߶νǵ�
	ROIcornerSubPix(dst_perspective, dstContour, 2, corners);

	/// 2--��ʾ��׼����Ľǵ�  
	for( int i = 0; i < corners.size(); i++ )  
	{  
		// ��ʾ���ǵ�  
		circle( dst_perspective, corners[i], r, Scalar(255,0,255), -1, 8, 0 );   		
	}
	//�滭�����߶�
	if(isDrawLines)
		line( dst_perspective, corners[0], corners[1], Scalar( 0, 255, 0), r);

	double distance;    
    distance = powf((corners[0].x - corners[1].x),2) + powf((corners[0].y - corners[1].y),2);    
    distance = sqrtf(distance)*ratio;  
	double actualError = lineLength - distance;
	//actualError = abs(actualError);

	//��ͼ������ʾ�ı��ַ���
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



