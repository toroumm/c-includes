#ifndef LBP_H
#define LBP_H

#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>

using namespace cv;
using namespace std;

class LBP
{
public:
    LBP();

    vector<int> histogram(vector<int> img,int max, int _bins);
    vector<int> histogram(Mat img,int max, int _bins);

    Mat convert_vector_2D_to_Mat(vector<vector<float> >vec);
    vector<vector<float> >convert_mat_to_vector_2D(Mat mat);


    Mat SLBP(Mat img); ///S from simple
    Mat ELBP( Mat src, int radius, int neighbors);
    vector<int> RLBP(Mat src, int radius, int neighbors); //R from rotation invariant
    vector<int> URLBP(Mat src, int radius, int neighbors);
    int max_bit_wise(vector<int>vet,int neighbors);
    int uniform_rlbp(vector<int>vet);


};

#endif // LBP_H
