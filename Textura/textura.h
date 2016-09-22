#ifndef Textura_H
#define Textura_H

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/mat.hpp"
#include <vector>

using namespace std;

class Textura
{

public:

    Textura();
    void TexCOM(cv::Mat vtr, int width, int height, vector<vector<float> > &atr);
    void TexNGTDM(cv::Mat vtr, int width, int height,vector<vector<float> >&atr);
    void TexTU(cv::Mat vtr, int width, int height, vector<vector<float> > &atr);



private:


     void TexWindow(cv::Mat vtr, vector<vector<int> > &pixel, int x, int y);
     void TexZeraMat(float **matn, float **matv);
     void TexDefineVector(vector<vector<float> > &vtr, int cols, int rows, int width, int height, int attributes);
     void TexClearVector(vector<vector<float> >&vtr);
     void TexFree(vector<vector<float> >&vtr);
     void TexFree(vector<vector<int> >&vtr);
     void TexFree(vector<vector<double> >&vtr);
     void TexFree(vector<float>&vtr);

     //COM
     void TexExtractCOM(float **matn, float **matv, vector<vector<int> > pixel, int width, int height);
     void TexMedidasCOM(float **matn, vector<float>&vet);

    //NGTDM
     void TexExtractNGTDM(float mat[][3], float vet[], int width);
     void TexSiNGTDM(vector<vector<int> >pixel,float mat[][3],int distancia);

     //TU
     void TexExtractTU(vector<vector<int> >pixel, float **mat);
     void TexFrequencyTU(float **mat, float vet[6561][8], float total[], int width, int height);
     void TexBwsTU(float vet[6561][8], float ASD[], float total[]);
     void TexGsTU(float vet[6561][8], float ASD[], float total[]);
     void TexDdTU(float vet[6561][8], float ASD[], float total[]);

};

#endif // Textura_H
