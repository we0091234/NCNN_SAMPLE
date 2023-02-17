#include "crnn.h"


CRNN::CRNN()
{
    
}
int CRNN::load(const char * param,const char *bin)
{
    crnn.load_param(param);
    crnn.load_model(bin);
    return 0;
}


int CRNN::plate_rec_color(cv::Mat &img,int target_w,int target_h,std::string & plate_result,std::string & plate_color)
{
     int img_w= img.cols;
    int img_h =img.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img_w, img_h);
    const float mean_vals[3] = {149.94f, 149.94f, 149.94f};
    const float std_vals[3]={0.020319f, 0.020319f, 0.020319f};
    in.substract_mean_normalize(mean_vals, std_vals);
    ncnn::Mat out1;
    ncnn::Mat out2;
    ncnn::Extractor ex = crnn.create_extractor();
    ex.set_light_mode(true);
    ex.input("images", in);
    ex.extract("output_1", out1);      //字符识别分支
    ex.extract("output_2", out2);      //颜色分支


    float tmp = out2[0];

    int index = 0;
    for (int i = 1; i<out2.w; i++)
    {
        if(out2[i]>tmp)
        {
            index = i;
            tmp = out2[i];
        }

    }
    int out1_w = out1.w;
    int out1_h = out1.h;
    std::vector<int> final_code;

    int pre_code =0;

    for(int i = 0;i<out1_h;i++)
    {
        float *feat_ptr=out1.row(i);   //定位到第几行
        int label = 0;
        float max_= feat_ptr[0];
        for(int j = 1; j<out1_w; j++)
        {
            if(feat_ptr[j]>max_)
            {
                label = j;
                max_=feat_ptr[j];
            }
           
        }
         if (label!=0&&label!=pre_code)
            {
                final_code.push_back(label);
            }
            pre_code = label;
    }
    for(int i = 0; i<final_code.size(); i++)
    {
        // std::cout<<plate_string_[final_code[i]]<<"";
        plate_result+=plate_string_[final_code[i]];
    }

    plate_color = color_[index];
}

