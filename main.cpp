#include "spherical_dibr.hpp"
#include "debug_print.h"

#include <iostream>
#include <sstream>
#include <vector>
#include "INIReader.h"

using namespace std;
using namespace cv;

#define RAD(x) M_PI*(x)/180.0
#define DEGREE(x) 180.0*(x)/M_PI

vector<string> string_parse(string str, string tok)
{
    vector<string> token;
    size_t pos = 0;
    while ((pos = str.find(tok)) != std::string::npos) 
    {
        token.push_back(str.substr(0, pos));
        str.erase(0, pos + tok.length());
    }
    token.push_back(str);

    return token;
}

Vec3d string_to_vec(string str)
{
    Vec3d vec;
    vector<string> vec_str = string_parse(str, " ");
    vec[0] = stod(vec_str[0]);
    vec[1] = stod(vec_str[1]);
    vec[2] = stod(vec_str[2]);

    return vec;
}

int main()
{
    INIReader reader("camera_info.ini");

    if (reader.ParseError() < 0)
    {
        std::cout << "Can't load 'camera_info.ini'\n";
        return 1;
    }

    int num_of_camera = reader.GetInteger("camera", "number", -1);
    std::cout << "Config loaded from 'camera_info.ini'\n"
              << "number of camera : " <<  reader.GetInteger("camera", "number", -1) << "\n"
              << "rotation vector\n" << reader.Get("camera", "rotation", "UNKNOWN") << "\n"
              << "translation vector\n" << reader.Get("camera", "translation", "UNKNOWN") << "\n"
              << "image name\n" << reader.Get("camera", "imagename", "UNKNOWN") << "\n"
              << "depth name\n" << reader.Get("camera", "depthname", "UNKNOWN") << "\n"
              << "virtual view rotation\n" << reader.Get("virtualview", "rotation", "UNKNOWN") << "\n"
              << "virtual view translation\n" << reader.Get("virtualview", "translation", "UNKNOWN") << "\n";

    int cam_num = reader.GetInteger("camera", "number", -1);
    double depth_min = reader.GetReal("camera", "depthmin", -1);
    double depth_max = reader.GetReal("camera", "depthmax", -1);
    vector<string> cam_rot_str = string_parse(reader.Get("camera", "rotation", "UNKNOWN"), ",");
    vector<string> cam_tran_str = string_parse(reader.Get("camera", "translation", "UNKNOWN"), ",");
    vector<string> cam_name = string_parse(reader.Get("camera", "imagename", "UNKNOWN"), ",");
    vector<string> depth_name = string_parse(reader.Get("camera", "depthname", "UNKNOWN"), ",");
    string vt_rot_str = reader.Get("virtualview", "rotation", "UNKNOWN");
    string vt_tran_str = reader.Get("virtualview", "translation", "UNKNOWN");
    int filter_option = reader.GetInteger("option", "filteroption", -1);
    int render_option = reader.GetInteger("option", "renderoption", -1);

    vector<Vec3d> cam_rot, cam_tran;
    for(int i = 0; i < cam_num; i++)
    {
        cam_rot.push_back(string_to_vec(cam_rot_str[i]));
        cam_tran.push_back(string_to_vec(cam_tran_str[i]));
    }
    Vec3d vt_rot, vt_tran;
    vt_rot = string_to_vec(vt_rot_str);
    vt_tran = string_to_vec(vt_tran_str);

    spherical_dibr sp_dibr;
    vector<Mat> im(cam_num);
    vector<Mat> depth(cam_num);
    vector<Mat> depth_double(cam_num);
    vector<Mat> rot_mat(cam_num);
    vector<Mat> rot_mat_inv(cam_num);
    for(int i = 0; i < cam_num; i++)
    {
        im[i] = imread(cam_name[i], IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
        depth[i] = imread(depth_name[i], IMREAD_ANYDEPTH);
        rot_mat[i] = sp_dibr.eular2rot(Vec3f(RAD(cam_rot[i][0]), RAD(cam_rot[i][1]), RAD(cam_rot[i][2])));
        rot_mat_inv[i] = rot_mat[i].t();

        double min_pixel = 0, max_pixel = 65535;
        double min_dist = depth_min, max_dist = depth_max;
        depth_double[i] = sp_dibr.map_distance(depth[i], min_pixel, max_pixel, min_dist, max_dist);
    }
    Mat vt_rot_mat = sp_dibr.eular2rot(Vec3f(RAD(vt_rot[0]), RAD(vt_rot[1]), RAD(vt_rot[2])));
    
    vector<Mat> img_forward(cam_num);
    vector<Mat> depth_forward(cam_num);
    vector<Mat> depth_cube(cam_num);
    vector<Mat> depth_cube_filter(cam_num);
    vector<Mat> depth_map_result(cam_num);
    vector<Mat> img_result(cam_num);
    vector<double> cam_dist(cam_num);

    for(int i = 0; i < cam_num; i++)
    {
        START_TIME(render_one_image);
        spherical_dibr spd;
        cout << "image " << i << " now do rendering" << endl;

        // Calculate R|t to render virtual view point
        Mat r = vt_rot_mat*rot_mat_inv[i];
        Vec3d t_tmp = vt_tran-cam_tran[i];
        double* rot_mat_data = (double*)rot_mat[i].data;
        Vec3d t;
        t[0] = rot_mat_data[0]*t_tmp[0] + rot_mat_data[1]*t_tmp[1] + rot_mat_data[2]*t_tmp[2];
        t[1] = rot_mat_data[3]*t_tmp[0] + rot_mat_data[4]*t_tmp[1] + rot_mat_data[5]*t_tmp[2];
        t[2] = rot_mat_data[6]*t_tmp[0] + rot_mat_data[7]*t_tmp[1] + rot_mat_data[8]*t_tmp[2];

        // Render virtual view point
        spd.render(im[i], depth_double[i], depth_min, depth_max, r, t, render_option, filter_option);
        STOP_TIME(render_one_image);

        // Put result of each rendering results to vector buffer
        if(render_option == spd.FORWARD_INVERSE)
        {
            img_forward[i] = spd.im_out_forward;
            depth_forward[i] = spd.depth_out_forward;
        }
        depth_cube[i] = spd.depth_cube;
        if(filter_option == spd.FILTER_MEDIAN)
        {
            depth_cube_filter[i] = spd.depth_cube_median;
            depth_map_result[i] = spd.depth_out_median;
            img_result[i] = spd.im_out_inverse_median;
        }
        else if(filter_option == spd.FILTER_CLOSING)
        {
            depth_cube_filter[i] = spd.depth_cube_closing;
            depth_map_result[i] = spd.depth_out_closing;
            img_result[i] = spd.im_out_inverse_closing;
        }
        cam_dist[i] = sqrt(t[0]*t[0] + t[1]*t[1] + t[2]*t[2]);
    }

    int width = im[0].cols;
    int height = im[0].rows;
    Mat blended_img(height, width, CV_16UC3);
    vector<Vec3w*> im_data(cam_num);
    vector<double*> depth_data(cam_num);
    Vec3w* blended_data = (Vec3w*)blended_img.data;
    for(int i = 0; i < cam_num; i++)
    {
        im_data[i] = (Vec3w*)img_result[i].data;
        depth_data[i] = (double*)depth_map_result[i].data;
    }

    START_TIME(Blend_image);
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            Vec3d pixel_val = 0;
            double dist_sum = 0;
            double threshold = 0.01; // consider below than threshold are occluded area
            int valid_count = 0;
            for(int c = 0; c < cam_num; c++)
            {
                if(depth_data[c][i*width + j] > threshold)
                {
                    valid_count++;
                    dist_sum += 1/cam_dist[c];
                }
            }
            for(int c = 0; c < cam_num; c++)
            {
                if(depth_data[c][i*width + j] > threshold)
                {
                    if(valid_count > 1)
                    {
                        pixel_val[0] += (1/cam_dist[c]/dist_sum)*im_data[c][i*width + j][0];
                        pixel_val[1] += (1/cam_dist[c]/dist_sum)*im_data[c][i*width + j][1];
                        pixel_val[2] += (1/cam_dist[c]/dist_sum)*im_data[c][i*width + j][2];
                    }
                    else if(valid_count == 1)
                    {
                        pixel_val[0] += im_data[c][i*width + j][0];
                        pixel_val[1] += im_data[c][i*width + j][1];
                        pixel_val[2] += im_data[c][i*width + j][2];
                    }
                }
            }
            blended_data[i*width + j][0] = pixel_val[0];
            blended_data[i*width + j][1] = pixel_val[1];
            blended_data[i*width + j][2] = pixel_val[2];
        }
    }
    STOP_TIME(Blend_image);

    // Save images
    cout << "Save images" << endl;
    vector<int> param;
    param.push_back(IMWRITE_PNG_COMPRESSION);
    param.push_back(0);
    for(int i = 0; i < cam_num; i++)
    {
        if(render_option == sp_dibr.FORWARD_INVERSE)
        {
            string forward_image_name = "test_result";
            forward_image_name = forward_image_name + to_string(i);
            forward_image_name = forward_image_name + "_forward.png";
            cv::imwrite(forward_image_name, img_forward[i], param);

            double min_pixel = 0, max_pixel = 65535;
            double min_dist = depth_min, max_dist = depth_max;
            string depth_forward_name = "test_depth";
            depth_forward_name = depth_forward_name + to_string(i);
            depth_forward_name = depth_forward_name + "_forward.png";
            cv::imwrite(depth_forward_name, sp_dibr.remap_distance(depth_forward[i], min_dist, max_dist, min_pixel, max_pixel), param);
        }

        string image_name = "test_result";
        image_name = image_name + to_string(i);
        image_name = image_name + "_inverse.png";
        cv::imwrite(image_name, img_result[i], param);

        double min_pixel = 0, max_pixel = 65535;
        double min_dist = depth_min, max_dist = depth_max;
        string depth_closing_name = "test_depth";
        depth_closing_name = depth_closing_name + to_string(i);
        depth_closing_name = depth_closing_name + ".png";
        cv::imwrite(depth_closing_name, sp_dibr.remap_distance(depth_map_result[i], min_dist, max_dist, min_pixel, max_pixel), param);

        string test_depth_cube_name = "test_depth_cube";
        test_depth_cube_name = test_depth_cube_name + to_string(i);
        test_depth_cube_name = test_depth_cube_name + "_forward.png";
        cv::imwrite(test_depth_cube_name, sp_dibr.remap_distance(depth_cube[i], min_dist, max_dist, min_pixel, max_pixel), param);

        string test_depth_cube_filter_name = "test_depth_cube";
        test_depth_cube_filter_name = test_depth_cube_filter_name + to_string(i);
        test_depth_cube_filter_name = test_depth_cube_filter_name + "_filter.png";
        cv::imwrite(test_depth_cube_filter_name, sp_dibr.remap_distance(depth_cube_filter[i], min_dist, max_dist, min_pixel, max_pixel), param);
    }
    string blended_name = "blend.png";
    cv::imwrite(blended_name, blended_img, param);

    return 0;
}
