#include "lbp.h"
#include <iostream>
#include <algorithm>

LBP::LBP(){}

//*****************************************************************************************
vector<vector<float> > LBP::convert_mat_to_vector_2D(Mat mat){

    vector<vector<float> >vec;
    vec.resize(mat.rows);

    for(int i = 0; i < mat.rows; i++)
        for(int j = 0; j < mat.cols; j++)
            vec[i].push_back(mat.at<uchar>(i,j));

    return vec;
}
//*****************************************************************************************
Mat LBP::convert_vector_2D_to_Mat(vector<vector<float> > vec){

    Mat mat(vec.size(),vec[0].size(),CV_8UC1);

    for(int i = 0; i < mat.rows; i++)
        for(int j = 0; j < mat.cols; j++)
            mat.at<uchar>(i,j) = vec[i][j];
    return mat;
}


//*****************************************************************************************
int LBP::uniform_rlbp(vector<int> vet){
    int value = 0;
    for(int i =1; i <(int)vet.size();i++){
        value+=abs(vet[i]-vet[i-1]);
    }
    return value;
}
//*****************************************************************************************
int LBP::max_bit_wise(vector<int>vet,int neighbors){

    int max = 0;
    for(int i = 0; i < (int)vet.size(); i++){
        int x= 0;
        for(int j = 0; j < (int)vet.size(); j++){

            x+=vet[j]<<(j+i)%neighbors;
        }
        if(x > max)
            max = x;
    }
    return max;
}

//*****************************************************************************************
Mat LBP::ELBP( Mat src, int radius, int neighbors) {
    neighbors = max(min(neighbors,31),1); // set bounds...

    // Note: alternatively you can switch to the new OpenCV Mat_
    // type system to define an unsigned int matrix... I am probably
    // mistaken here, but I didn't see an unsigned int representation
    // in OpenCV's classic typesystem...
    Mat dst = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_8UC1);



    for(int n=0; n<neighbors; n++) {
        // sample points
        float x = static_cast<float>(radius) * cos(2.0*M_PI*n/static_cast<float>(neighbors));
        float y = static_cast<float>(radius) * -sin(2.0*M_PI*n/static_cast<float>(neighbors));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        vector<int>aux;
        for(int i=radius; i < src.rows-radius;i++) {
            for(int j=radius;j < src.cols-radius;j++) {
                float t = w1*src.at<uchar>(i+fy,j+fx) + w2*src.at<uchar>(i+fy,j+cx) + w3*src.at<uchar>(i+cy,j+fx) + w4*src.at<uchar>(i+cy,j+cx);
                // we are dealing with floating point precision, so add some little tolerance
                dst.at<unsigned char>(i-radius,j-radius) += (((t > src.at<uchar>(i,j))) && (abs(t-src.at<uchar>(i,j)) > std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
    return dst;
}
//*****************************************************************************************
vector<int> LBP::URLBP(Mat src, int radius, int neighbors){

    vector<int>dst;
    for(int i=radius; i < src.rows-radius;i++) {
        for(int j=radius;j < src.cols-radius;j++) {

            vector<int>aux;
            for(int n=0; n<neighbors; n++) {
                // sample points
                float x = static_cast<float>(radius) * cos(2.0*M_PI*n/static_cast<float>(neighbors));
                float y = static_cast<float>(radius) * -sin(2.0*M_PI*n/static_cast<float>(neighbors));
                // relative indices
                int fx = static_cast<int>(floor(x));
                int fy = static_cast<int>(floor(y));
                int cx = static_cast<int>(ceil(x));
                int cy = static_cast<int>(ceil(y));
                // fractional part
                float ty = y - fy;
                float tx = x - fx;
                // set interpolation weights
                float w1 = (1 - tx) * (1 - ty);
                float w2 =      tx  * (1 - ty);
                float w3 = (1 - tx) *      ty;
                float w4 =      tx  *      ty;

                float t = w1*src.at<uchar>(i+fy,j+fx) + w2*src.at<uchar>(i+fy,j+cx) + w3*src.at<uchar>(i+cy,j+fx) + w4*src.at<uchar>(i+cy,j+cx);
                // we are dealing with floating point precision, so add some little tolerance
                aux.push_back(((t > src.at<uchar>(i,j))) && (abs(t-src.at<uchar>(i,j)) > std::numeric_limits<float>::epsilon()));
            }
            // dst.push_back(max_bit_wise(aux,neighbors));
            if(uniform_rlbp(aux)<=2)
                dst.push_back(max_bit_wise(aux,neighbors));
        }
    }
    return dst;
}

//*****************************************************************************************

vector<int> LBP::RLBP(Mat src, int radius, int neighbors){

    //  vector<int>asd(src.cols-2*radius); //src.rows-2*radius, src.cols-2*radius
    //  vector<vector<int> >dst(src.rows-2*radius,asd);
    vector<int>dst;
    for(int i=radius; i < src.rows-radius;i++) {
        for(int j=radius;j < src.cols-radius;j++) {

            vector<int>aux;
            for(int n=0; n<neighbors; n++) {
                // sample points
                float x = static_cast<float>(radius) * cos(2.0*M_PI*n/static_cast<float>(neighbors));
                float y = static_cast<float>(radius) * -sin(2.0*M_PI*n/static_cast<float>(neighbors));
                // relative indices
                int fx = static_cast<int>(floor(x));
                int fy = static_cast<int>(floor(y));
                int cx = static_cast<int>(ceil(x));
                int cy = static_cast<int>(ceil(y));
                // fractional part
                float ty = y - fy;
                float tx = x - fx;
                // set interpolation weights
                float w1 = (1 - tx) * (1 - ty);
                float w2 =      tx  * (1 - ty);
                float w3 = (1 - tx) *      ty;
                float w4 =      tx  *      ty;

                float t = w1*src.at<uchar>(i+fy,j+fx) + w2*src.at<uchar>(i+fy,j+cx) + w3*src.at<uchar>(i+cy,j+fx) + w4*src.at<uchar>(i+cy,j+cx);
                // we are dealing with floating point precision, so add some little tolerance
                aux.push_back(((t > src.at<uchar>(i,j))) && (abs(t-src.at<uchar>(i,j)) > std::numeric_limits<float>::epsilon()));

            }
            dst.push_back(max_bit_wise(aux,neighbors));
        }
    }
    return dst;
}

//*****************************************************************************************

Mat LBP::SLBP(Mat img){
    Mat dst = Mat::zeros(img.rows-2, img.cols-2, CV_8UC1);
    for(int i=1;i<img.rows-1;i++) {
        for(int j=1;j<img.cols-1;j++) {
            uchar center = img.at<uchar>(i,j);
            unsigned char code = 0;
            code |= ((img.at<uchar>(i-1,j-1)) > center) << 7;
            code |= ((img.at<uchar>(i-1,j)) > center) << 6;
            code |= ((img.at<uchar>(i-1,j+1)) > center) << 5;
            code |= ((img.at<uchar>(i,j+1)) > center) << 4;
            code |= ((img.at<uchar>(i+1,j+1)) > center) << 3;
            code |= ((img.at<uchar>(i+1,j)) > center) << 2;
            code |= ((img.at<uchar>(i+1,j-1)) > center) << 1;
            code |= ((img.at<uchar>(i,j-1)) > center) << 0;
            dst.at<uchar>(i-1,j-1) = code;
        }
    }
    return dst;
}

//*****************************************************************************************


vector<int> LBP::histogram(Mat img,int max, int _bins){

    vector<int>bins(_bins), hist_max(max);

    for(int i = 0; i < img.rows; i++)
        for(int j = 0; j < img.cols; j++)
            hist_max[int(img.at<uchar>(i,j))] +=1;
    int step = max/_bins;

    if((max % _bins) >0)
        step+=1;

    for(int i = 0; i < (int)bins.size();i++){
        for(int j = 0; j < step; j++){

            if(i*step +j >= max)
                break;

            bins[i] += hist_max[i*step +j];
        }
    }
    return bins;
}

//*****************************************************************************************

vector<int> LBP::histogram(vector<int> img,int max, int _bins){

    vector<int>bins(_bins), hist_max(max);

    for(int i = 0; i < (int)img.size(); i++)
        hist_max[int(img[i])] +=1;
    int step = max/_bins;

    if(max % _bins >0)
        step+=1;

    for(int i = 0; i < (int)bins.size();i++){
        for(int j = 0; j < step; j++){

            if(i*step +j >= max)
                break;

            bins[i] += hist_max[i*step +j];
        }
    }
    return bins;
}
//*****************************************************************************************



//*****************************************************************************************
