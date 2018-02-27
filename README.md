CameraMeasureLine
===

采用普通相机测量直线
---

* 效果图
   *    
   ![](https://github.com/HQlin/CameraMeasureLine/blob/master/pic/halcon效果图.png "halcon效果图")
   *    
   ![](https://github.com/HQlin/CameraMeasureLine/blob/master/pic/opencv效果图.png "opencv效果图")
* 实现思路
   * 1、截取原图的重要边缘
   * 2、亚像素顺(逆)时针获取四边形的角点
   * 3、对测量目标原图进行透视变换
   * 4、截取原图变换后需要测量直线
   * 5、对直线进行比率换算与实际计算误差
* 实现平台
   * halcon11
   * opencv2.4.9 + vs2010
* opencv环境配置
   * 安装路径：D:\Program Files\opencv
   * 包含目录：D:\Program Files\opencv\build\include
   * 库目录：D:\Program Files\opencv\build\x86\vc10\lib
   * 附加依赖项：
	opencv_ml249d.lib
	opencv_calib3d249d.lib
	opencv_contrib249d.lib
	opencv_core249d.lib
	opencv_features2d249d.lib
	opencv_flann249d.lib
	opencv_gpu249d.lib
	opencv_highgui249d.lib
	opencv_imgproc249d.lib
	opencv_legacy249d.lib
	opencv_objdetect249d.lib
	opencv_ts249d.lib
	opencv_video249d.lib
	opencv_nonfree249d.lib
	opencv_ocl249d.lib
	opencv_photo249d.lib
	opencv_stitching249d.lib
	opencv_superres249d.lib
	opencv_videostab249d.lib
* 改进空间
	* 进行相机标定，矫正畸变提高精度
	* 增强预检鲁棒性，提高对测量目标识别的判断
