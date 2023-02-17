#ifndef __CRNN__
#define __CRNN__
#include <net.h>
#include <opencv2/opencv.hpp>
#include <iostream>

const std::vector<std::string> plate_string_={"#","京","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","皖", \
"闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁","新","学","警","港","澳","挂","使","领","民","航","危", \
"0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z","险","品"};

static char *color_[]={"黑色","蓝色","绿色","白色","黄色"}; 

class CRNN 
{
   public:
   CRNN();
   int load(const char *param,const char *bin);
   int plate_rec_color(cv::Mat & bgr,int target_w ,int target_h,std::string & plate_result,std::string & plate_color);
   ncnn::Net crnn;
};

#endif