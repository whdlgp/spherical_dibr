#include "spherical_dibr.hpp"
/*
int main(int argc, char** argv)
{
    spherical_dibr sp_dibr;
    sp_dibr.test(argc, argv);
    return 0;
}*/

// Example that shows simple usage of the INIReader class

#include <iostream>
#include <sstream>
#include <vector>
#include "INIReader.h"

using namespace std;
using namespace cv;

#define RAD(x) M_PI*(x)/180.0
#define DEGREE(x) 180.0*(x)/M_PI

// from: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
Vec3f rot2euler(Mat &R)
{
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
    bool singular = sy < 1e-6; // If
    float x, y, z;

    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }

    return Vec3f(x, y, z);
}

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
    std::cout << "Config loaded from 'test.ini'\n"
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
        rot_mat_inv[i] = rot_mat[i].inv();

        double min_pixel = 0, max_pixel = 65535;
        double min_dist = depth_min, max_dist = depth_max;
        depth_double[i] = sp_dibr.map_distance(depth[i], min_pixel, max_pixel, min_dist, max_dist);
    }
    Mat vt_rot_mat = sp_dibr.eular2rot(Vec3f(RAD(vt_rot[0]), RAD(vt_rot[1]), RAD(vt_rot[2])));
    
    for(int i = 0; i < cam_num; i++)
    {
        spherical_dibr spd;
        cout << "image " << i << " now do rendering" << endl;
        Mat r = rot_mat_inv[i]*vt_rot_mat;
        Vec3d r_vec = rot2euler(r);
        cout << DEGREE(r_vec[0]) << ',' << DEGREE(r_vec[1]) << ',' << DEGREE(r_vec[2]) << endl;
        Vec3d t = vt_tran-cam_tran[i];
        spd.render(im[i], depth_double[i], r, t);
        string name = "test_result";
        name = name + to_string(i);
        name = name + ".png";
        imwrite(name, spd.im_out_inverse_closing);
    }

    return 0;
}
