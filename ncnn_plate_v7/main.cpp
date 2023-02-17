#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolov7face.h"
#include "crnn.h"
#include <opencv2/freetype.hpp>
#define TARGET_SIZE 640

void drawBboxes(cv::Mat &img ,std::vector<Object> &bboxes,std::string &text,int i)
{
    std::string ttf_pathname = "/mnt/Gu/xiaolei/cplusplus/trt_project/trt_plate/font/NotoSansCJK-Regular.otf";
    int top = bboxes[i].rect.y;
    int left = bboxes[i].rect.x;
    int baseLine1;
    cv::Size labelSize1 = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine1);
    top = cv::max(top, labelSize1.height);
    //画框
    cv::rectangle(img, cv::Point(left, top - round(1.6*labelSize1.height)), cv::Point(left + round(1.2*labelSize1.width), top + baseLine1), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::Ptr<cv::freetype::FreeType2> ft2 = cv::freetype::createFreeType2();

    ft2->loadFontData(ttf_pathname,0);
    //标签
    ft2->putText(img, text, cv::Point(left, top), 21, cv::Scalar(255, 0, 0), -1,4,true);

}

float getNorm2(float x,float y)
{
    return sqrt(x*x+y*y);
}

cv::Mat get_split_merge(cv::Mat &img)   //双层车牌 分割 拼接
{
    cv::Rect  upper_rect_area = cv::Rect(0,0,img.cols,int(5.0/12*img.rows));
    cv::Rect  lower_rect_area = cv::Rect(0,int(1.0/3*img.rows),img.cols,img.rows-int(1.0/3*img.rows));
    cv::Mat img_upper = img(upper_rect_area);
    cv::Mat img_lower =img(lower_rect_area);
    cv::resize(img_upper,img_upper,img_lower.size());
    cv::Mat out(img_lower.rows,img_lower.cols+img_upper.cols, CV_8UC3, cv::Scalar(114, 114, 114));
    img_upper.copyTo(out(cv::Rect(0,0,img_upper.cols,img_upper.rows)));
    img_lower.copyTo(out(cv::Rect(img_upper.cols,0,img_lower.cols,img_lower.rows)));

    return out;
}

cv::Mat getTransForm(cv::Mat &src_img, cv::Point2f  order_rect[4]) //透视变换
{cv::Point2f w1=order_rect[0]-order_rect[1];
            cv::Point2f w2=order_rect[2]-order_rect[3];
            auto width1 = getNorm2(w1.x,w1.y);
            auto width2 = getNorm2(w2.x,w2.y);
            auto maxWidth = std::max(width1,width2);

            cv::Point2f h1=order_rect[0]-order_rect[3];
            cv::Point2f h2=order_rect[1]-order_rect[2];
            auto height1 = getNorm2(h1.x,h1.y);
            auto height2 = getNorm2(h2.x,h2.y);
            auto maxHeight = std::max(height1,height2);
            //  透视变换
            std::vector<cv::Point2f> pts_ori(4);
            std::vector<cv::Point2f> pts_std(4);

            pts_ori[0]=order_rect[0];
            pts_ori[1]=order_rect[1];
            pts_ori[2]=order_rect[2];
            pts_ori[3]=order_rect[3];

            pts_std[0]=cv::Point2f(0,0);
            pts_std[1]=cv::Point2f(maxWidth,0);
            pts_std[2]=cv::Point2f(maxWidth,maxHeight);
            pts_std[3]=cv::Point2f(0,maxHeight);

            cv::Mat M = cv::getPerspectiveTransform(pts_ori,pts_std);
            cv:: Mat dstimg;
            cv::warpPerspective(src_img,dstimg,M,cv::Size(maxWidth,maxHeight));
            return dstimg;
}

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		std::cout << "Usage:" << argv[0] << " <test image path>\n";
		return -1;
	}

	int num_threads = 1;
	YoloFace yolov7face;
	const char * param_path_det = "../model_detect/yolov7-lite-t-sim-opt-fp16.param";
	const char * bin_path_det = "../model_detect/yolov7-lite-t-sim-opt-fp16.bin";
	int ret = yolov7face.load(param_path_det,bin_path_det, TARGET_SIZE);
    
	const char * param_path_rec = "../model_rec/best_plate_color.param";
	const char * bin_path_rec = "../model_rec/best_plate_color.bin";

	CRNN my_crnn;
    my_crnn.load(param_path_rec,bin_path_rec);
    // cv::Mat img = cv::imread("/mnt/Gu/xiaolei/cplusplus/NCNN/ncnn_plate_v7/imgs/tmpD60D.png");
    int target_h = 48;
    int target_w = 168;
   
	

	cv::Mat image = cv::imread(argv[1]);
	if(image.empty())
	{
		std::cout << "input image is empty!\n";
		return -1;
	}

	std::vector<Object> results;
	auto start = cv::getTickCount();
	yolov7face.detect(image, results,TARGET_SIZE);
	
	cv::Point2f order_rect[4];
	for(int i = 0; i < results.size(); i ++)
	{
		 std::string plate_no;
    	 std::string plate_color;
		float x1 = results[i].rect.x;
		float y1 = results[i].rect.y;
		float w1 = results[i].rect.width;
		float h1 = results[i].rect.height;
		cv::Rect2f box(x1, y1, w1, h1);
	
		for(int j = 0; j < NUM_KEY_POINTS; j ++)
		{
			cv::Point2f pt;
			pt.x = results[i].pts[j].x;
			pt.y = results[i].pts[j].y;

			if(results[i].pts[j].score > 0.5)
			{
				cv::circle(image, pt, 1, cv::Scalar(0,255,0), 2, 8, 0);
				 order_rect[j]=pt;
			}
		}
		 cv::Mat roiImg = getTransForm(image,order_rect);
		 if (results[i].label==1)
			roiImg = get_split_merge(roiImg);
		 cv::resize(roiImg,roiImg,cv::Size(target_w,target_h));
		 my_crnn.plate_rec_color(roiImg,target_w,target_h,plate_no,plate_color);

		  std::cout<<plate_no<<" "<<plate_color<<std::endl;	  
		  cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2, 8, 0);

		   std::string label1=plate_no+std::string(" ")+plate_color;
            drawBboxes(image ,results,label1,i);
	}

	auto end = cv::getTickCount();
	auto time_gap = (end-start)/cv::getTickFrequency()*1000;
	
	std::cout << "detected " << results.size() << " flates "<<"use time :"<<time_gap<<" ms\n";

	cv::imwrite("result.jpg", image);

	

	return 0;
}
