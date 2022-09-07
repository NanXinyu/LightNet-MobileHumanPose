//
// Created by nanxinyu on 2022/8/26.
//

#ifndef SINGLEPOSE_MOBILEHUMANPOSE_H
#define SINGLEPOSE_MOBILEHUMANPOSE_H

#include <opencv2/core/core.hpp>
#include <net.h>

struct KeyPoint
{
    cv::Point2f p;
    float prob;
};

class MHPNet
{
public:
    MHPNet();

    int load(const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    int load(AAssetManager* mgr, const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    int detect(const cv::Mat& rgb);

    int draw(cv::Mat& rgb);
    //void detect_pose(cv::Mat &rgb, std::vector<KeyPoint>& keypoints);


private:

    void detect_pose(const cv::Mat& bgr, std::vector<KeyPoint>& keypoints);
    ncnn::Net mhpnet;
    //void detect_pose(cv::Mat &rgb, std::vector<KeyPoint>& keypoints);

    //float* x = new float[192*256*3];

    //std::vector<int> img_size={192,256};

    int target_size;
    float mean_vals[3];
    float norm_vals[3];

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};
#endif //SINGLEPOSE_MOBILEHUMANPOSE_H

