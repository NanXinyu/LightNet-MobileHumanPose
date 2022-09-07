//
// Created by nanxinyu on 2022/8/26.
//

#include "MobileHumanPose.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"
// pixel_mean = (0.485, 0.456, 0.406)
// pixel_std = (0.229, 0.224, 0.225)
const float mean[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
const float norm[3] = {1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f};
MHPNet::MHPNet()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);

}

int MHPNet::load(const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    mhpnet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    mhpnet.opt = ncnn::Option();

#if NCNN_VULKAN
    mhpnet.opt.use_vulkan_compute = use_gpu;
#endif

    mhpnet.opt.num_threads = ncnn::get_big_cpu_count();
    mhpnet.opt.blob_allocator = &blob_pool_allocator;
    mhpnet.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    mhpnet.load_param(parampath);
    mhpnet.load_model(modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int MHPNet::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    mhpnet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    mhpnet.opt = ncnn::Option();

#if NCNN_VULKAN
    mhpnet.opt.use_vulkan_compute = use_gpu;
#endif

    mhpnet.opt.num_threads = ncnn::get_big_cpu_count();
    mhpnet.opt.blob_allocator = &blob_pool_allocator;
    mhpnet.opt.workspace_allocator = &workspace_pool_allocator;
    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    mhpnet.load_param(mgr,parampath);
    mhpnet.load_model(mgr,modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int MHPNet::detect(const cv::Mat& rgb)
{
    //TODO:add person detection
    return 0;
}

void MHPNet::detect_pose(const cv::Mat& bgr, std::vector<KeyPoint>& keypoints)
{
    int w = bgr.cols;
    int h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_RGB, w, h, 192, 256);
    in.substract_mean_normalize(mean,norm);

    ncnn::Extractor ex = mhpnet.create_extractor();

    ex.input("input",in);

    ncnn::Mat out;
    ex.extract("output",out);

    keypoints.clear();
    for(int p = 0; p < out.c; p++)
    {
        const ncnn::Mat m = out.channel(p);

        float max_prob = 0.f;
        int max_x = 0;
        int max_y = 0;
        for(int y = 0; y < out.h; y++)
        {
            const float* ptr = m.row(y);
            for(int x = 0; x < out.w; x++)
            {
                float prob = ptr[x];
                if(prob > max_prob)
                {
                    max_prob = prob;
                    max_x = x;
                    max_y = y;
                }
            }
        }
        KeyPoint keyPoint;
        keyPoint.p = cv::Point2f(max_x * w / (float)out.w, max_y * h / (float)out.h);
        keyPoint.prob = max_prob;

        keypoints.push_back(keyPoint);
    }


}

int MHPNet::draw(cv::Mat& rgb)
{
    std::vector<KeyPoint> points;
    detect_pose(rgb,points);
    cv::Mat image = rgb.clone();

    // skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
    static const int joint_pairs[16][2] = {{0,7},{7,8},{8,9},{9,10},{8,11},{11,12},{12,13},{8,14},{14,15},{15,16},{0,1},{1,2},{2,3},{0,4},{4,5},{5,6}};
            //{0,1},{0,2},{1,3},{2,4},{0,5},{0,6},{5,6},{5,7},{7,9},{6,8},{8,10},{11,12},{5,11},{11,13},{13,15},{6,12},{12,14},{14,16} };

    int color_index[][3] = { {255, 0, 0},
                             {0, 0, 255},
                             {255, 0, 0},
                             {0, 0, 255},
                             {255, 0, 0},
                             {0, 0, 255},
                             {0, 255, 0},
                             {255, 0, 0},
                             {255, 0, 0},
                             {0, 0, 255},
                             {0, 0, 255},
                             {0, 255, 0},
                             {255, 0, 0},
                             {255, 0, 0},
                             {255, 0, 0},
                             {0, 0, 255},
                             {0, 0, 255},
                             {0, 0, 255}, };


    for (int i = 0; i < 16; i++)
    {
        const KeyPoint& p1 = points[joint_pairs[i][0]];
        const KeyPoint& p2 = points[joint_pairs[i][1]];

        if(p1.prob < 0.2f || p2.prob <0.2f)
            continue;

        cv::line(image,p1.p,p2.p,cv::Scalar(color_index[i][0], color_index[i][1], color_index[i][2]),2);

    }
    for (int i = 0; i < points.size(); i++)
    {
        const KeyPoint& keypoint = points[i];

        if (keypoint.prob < 0.2f)
            continue;

        cv::circle(image, keypoint.p, 3, cv::Scalar(100, 255, 150), -1);
    }
    return 0;
}