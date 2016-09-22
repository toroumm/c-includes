#include "textura.h"
#include <iostream>
#include <cstdlib>
#include <math.h>
#include <cstdio>
#include <numeric>

#include <algorithm>
#include <math.h>

#define D 256
#define COM 6
#define NGTDM 5
#define TU 10

Textura::Textura(){

}
//********************************************************************************************

void Textura::TexClearVector(vector<vector<float> > &vtr){

    for(int i = 0; i < (int)vtr.size(); i++)
        if(accumulate(vtr[i].begin(),vtr[i].end(),0) <= 0){
            vtr[i].erase(vtr[i].begin(),vtr[i].end());

        }

    for(int i = 0; i < (int)vtr.size(); i++){

        if(vtr[i].empty()){
            vtr.erase(vtr.begin()+i);
            i-=1;
        }
    }
}
//********************************************************************************************
void Textura::TexDefineVector(vector<vector<float> > &vtr,int cols, int rows, int width, int height, int attributes){

    vector<float>aa(attributes);

    float resto1 =  (float)cols/(float)width, resto2 = (float)rows/(float)height,asd, fract1;

    fract1 = modf(resto1*resto2,&asd);

    vtr.resize((resto1)*(resto2)+(fract1*resto1)+(fract1*resto2));

    vector<vector<float> >atr(vtr.size(),aa);

    vtr = atr;
}
//**********************************************************************************************
void Textura::TexWindow(cv::Mat vtr,vector<vector<int> >&pixel, int x, int y){

    for(int i =0 ; i < (int)pixel.size(); i++){
        for(int j = 0 ; j < (int)pixel[i].size(); j++){

            pixel[i][j] = 0;

            if((i+x) >= vtr.rows-1 || (j+y) >= vtr.cols-1)
                break;

            pixel[i][j] = vtr.at<uchar>(i+x,j+y);

        }
    }
}
//**********************************************************************************************
void Textura::TexZeraMat(float **matn,float **matv){

    for(int i = 0; i < 4; i++){
        for(int j = 0; j <(D*D); j++){

            matn[i][j] = 0.0;
            matv[i][j] = 0.0;
        }
    }
}
//********************************************************************************************************************
void Textura::TexCOM(cv::Mat vtr,int width,int height,vector<vector<float> >&atr){

    vector<int>aa(width);
    vector<vector<int> >pixel(height,aa);
    vector<float>vet(COM);
    //    TexDefineVector(atr,vtr.cols,vtr.rows,width,height,COM);

    atr.resize(vtr.cols*vtr.rows);

    float **matv = new float*[4];
    float **matn = new float*[4];

    for(int i = 0; i < 4; i++){

        matv[i] = new float[D*D];
        matn[i] = new float[D*D];
    }


    //float **matv; //Dimensões : 1° -  0 ° / 2° - 45° / 3° -  90° / 4° -  135°
    //float **matn; //A mesma matriz acima so que com valores normalizados

    int t= 0;
    for(int x = 1; x < vtr.rows; x+=height ){ //Montagem da Matrix de Co-ocorrencia nos 4 Angulos

        for(int y = 1; y < vtr.cols ; y+=width){

            TexWindow(vtr,pixel,x,y);

            TexExtractCOM(matn,matv,pixel,width, height);

            TexMedidasCOM(matn,vet);

            TexZeraMat(matn,matv);

            if((int)atr.size()<= t)
                atr.resize(atr.size()+1);

            atr[t].resize(COM);

            for(int k = 0; k < COM ; k++){

                atr[t][k] = vet[k];

                if((atr[t][k] > 100000) || (atr[t][k] < 0.000001)){atr[t][k] = 0;}
            }

            t++;
        }
    }
    for(int i = 0; i < 4; i++){
        delete []matn[i];
        delete []matv[i];
    }
    delete []matn;
    delete []matv;

    TexClearVector(atr);
    TexFree(pixel);

}
//*************************************************************************************************************
void Textura::TexExtractCOM(float **matn,float **matv,vector<vector<int> >pixel , int width, int height){

    float T[4] = {0.0,0.0,0.0,0.0}; // Total para normalização da matv

    for(int y = 1; y < height-1; y++ ){ //Montagem da Matrix de Co-ocorrencia nos 4 Angulos

        for(int x = 1; x < width-1 ; x++){


            int valor = pixel[x][y]; //Valor do Pixel

            //QMessageBox::information(this,"ok",QString::number(pixel[x*width + y]))
            int p0 = pixel[x+1][y]; //Pixel a 0° direita
            int p4 = pixel[x-1][y]; //Pixel a 0° esquerda

            int p1 = pixel[x+1][y-1];//Pixel a 45° direita
            int p5 = pixel[x-1][y+1];//Pixel a 45° esquerda

            int p2 = pixel[x][y-1]; //Pixel a 90° direita
            int p6 = pixel[x][y+1];//Pixel a 90° esquerda

            int p3 = pixel[x-1][y-1];//Pixel a 135° direita
            int p7 = pixel[x+1][y+1];//Pixel a 135° esquerda

            //Contador aqui é montado a matrix de acordo com os pontos de referência
            matv[0][p0*D + valor] = matv[0][p0*D + valor]+1; matv[1][p1*D + valor] = matv[1][p1*D + valor]+1;
            matv[2][p2*D + valor] = matv[2][p2*D + valor]+1; matv[3][p3*D + valor] = matv[3][p3*D + valor]+1;
            matv[0][p4*D + valor] = matv[0][p4*D + valor]+1; matv[1][p5*D + valor] = matv[1][p5*D + valor]+1;
            matv[2][p6*D + valor] = matv[2][p6*D + valor]+1; matv[3][p7*D + valor] = matv[3][p7*D + valor]+1;

            T[1] = T[1]+1;
        }
    }T[1] = T[1]*2;

    for(int y = 0; y < 256; y++ ){ //Normalização da matrix matv, assim se controi a matrix matn

        for(int x = 0; x < 256 ; x++){

            matn[0][y*256 +x] = matv[0][y*256 +x]/T[1];    matn[1][y*256 +x] = matv[1][y*256 +x]/T[1];
            matn[2][y*256 +x] = matv[2][y*256 +x]/T[1];    matn[3][y*256 +x] = matv[3][y*256 +x]/T[1];
        }
    }
}
//***********************************************************************************************************************
void Textura::TexMedidasCOM(float **matn, vector<float> &vet){


    // Matrix Med = Medidas -> Linhas são os Angulos e as Colunas são as Medidas abaixo

    /* Esta função vai retornar as medias de :

        SMA ou Energia ->Uniformidade da Textura (SMA)
        Entropia ->Expressa a desordem contida na Textura (EN)
        Contraste -> Nivel de Espalhamento dos Tons na Imagem (CON)
        Heterogeneidade -> Variancia apresenta valores baixos se a imagem é mais heterogenia (HET)
        Correlação-> Mede a dependencia Linear entre os tons de cinza (COR)
        Homogeneidade -> Apresenta Correlação inversa com o Contraste (HOM)   */

    for(int y = 0; y < COM; y++)vet[y] = 0;

    float mi[4] =  {0.0, 0.0, 0.0, 0.0} , mj [4] = {0.0, 0.0, 0.0, 0.0};

    float varx[4] =  {0.0, 0.0, 0.0, 0.0} , vary [4] = {0.0, 0.0, 0.0, 0.0};

    for(int y = 0; y < 256; y++ ){

        for(int x = 0; x < 256 ; x++){

            //Media em X para calcular Heterogeneidade e Correlação
            mi[0] = mi[0] + (x*matn[0][y*D +x]);  mi[1] = mi[1] + (x*matn[1][y*D +x]);
            mi[2] = mi[2] + (x*matn[2][y*D +x]);  mi[3] = mi[3] + (x*matn[3][y*D +x]);

            //Media em Y para calcular Heterogeneidade e Correlação
            mj[0] = mj[0] + (y*matn[0][y*D +x]);  mj[1] = mj[1] + (y*matn[1][y*D +x]);
            mj[2] = mj[2] + (y*matn[2][y*D +x]);  mj[3] = mj[3] + (y*matn[3][y*D +x]);
        }
    }

    for(int y = 0; y < D; y++ ){

        for(int x = 0; x < D ; x++){

            //Variância para valcular Hetegeneidade de Correlção
            varx[0] = varx[0] + (pow(x-mi[0],2) * matn[0][y*D +x]);    varx[1] = varx[1] + (pow(x-mi[1],2) * matn[1][y*D +x]);
            varx[2] = varx[2] + (pow(x-mi[2],2) * matn[2][y*D +x]);    varx[3] = varx[3] + (pow(x-mi[3],2) * matn[3][y*D +x]);

            vary[0] = vary[0] + (pow(y-mj[0],2) * matn[0][y*D +x]);    vary[1] = vary[1] + (pow(y-mj[1],2) * matn[1][y*D +x]);
            vary[2] = vary[2] + (pow(y-mj[2],2) * matn[2][y*D +x]);    vary[3] = vary[3] + (pow(y-mj[3],2) * matn[3][y*D +x]);
        }
    }

    for(int y = 0; y < D; y++ ){

        for(int x = 0; x < D ; x++){

            //SMA
            vet[0] = vet[0] + ((pow(matn[0][y*D +x],2) + pow(matn[1][y*D +x],2)+ pow(matn[2][y*D +x],2) + pow(matn[3][y*D +x],2))/4);

            //Contraste
            vet[1] = vet[1] + (((pow((float)x-y,2)* (matn[0][y*D +x])) + (pow((float)x-y,2)* matn[1][y*D +x]) + (pow((float)x-y,2)*  matn[2][y*D +x])+ (pow((float)x-y,2)*  matn[3][y*D +x]))/4);

            //Homogeneidade
            vet[2] = vet[2] + ((((1/1+pow((float)x-y,2)) * matn[0][y*D +x]) + ((1/1+pow((float)x-y,2)) * matn[1][y*D +x]) + ((1/1+pow((float)x-y,2)) * matn[2][y*D +x]) + ((1/1+pow((float)x-y,2)) * matn[3][y*D +x]))/4);

            //Entropia
            if(matn[0][y*D +x] && matn[1][y*D +x] && matn[2][y*D +x] && matn[3][y*D +x]  > 0.000000001)
                vet[3] = vet[3] + (-1*((matn[0][y*D +x] * log10(matn[0][y*D +x])) + (matn[1][y*D +x] * log10(matn[1][y*D +x])) + (matn[2][y*D +x] * log10(matn[2][y*D +x])) + (matn[3][y*D +x] * log10(matn[3][y*D +x]))))/4;

            //Correlação
            if(varx[0] && varx[1] && varx[2] && varx[3] != 0)
                vet[4] = vet[4] + (((((x-mi[0]) * (y-mj[0])) / (sqrt(varx[0])* sqrt(vary[0]))) * matn[0][y*D +x]) +
                        ((((x-mi[1])  * (y-mj[1])) / (sqrt(varx[1])* sqrt(vary[1]))) * matn[1][y*D +x]) +
                        ((((x-mi[2])  * (y-mj[2])) / (sqrt(varx[2])* sqrt(vary[2]))) * matn[2][y*D +x]) +
                        ((((x-mi[3])  * (y-mj[3])) / (sqrt(varx[3])* sqrt(vary[3]))) * matn[3][y*D +x]))/4;
            else vet[4] = 0;
        }
    }
    //Heterogeneidade

    if(varx[0] && varx[1] && varx[2] && varx[3] != 0)
        vet[5] = (sqrt(pow(varx[0],2) + pow(vary[0],2)) + sqrt(pow(varx[1],2) + pow(vary[1],2)) +
                sqrt(pow(varx[2],2) + pow(vary[2],2)) + sqrt(pow(varx[3],2) + pow(vary[3],2)))/4;

    else vet[5] = 0;

}
//********************************************************************************************************************
void Textura::TexNGTDM(cv::Mat vtr, int width, int height,vector<vector<float> >&atr){

    vector<int>aa(width);
    vector<vector<int> >pixel(height,aa);

    //  TexDefineVector(atr,vtr.cols,vtr.rows,width,height,NGTDM);

    atr.resize(vtr.cols*vtr.rows);

    float mat[256][3];
    float vet[NGTDM];

    int k = 0;

    for(int x  = 1; x < vtr.rows; x+=height){

        for(int y = 1; y < vtr.cols; y+=width ){

            TexWindow(vtr,pixel,x,y);

            TexSiNGTDM(pixel,mat,1);

            TexExtractNGTDM(mat,vet,width);

            if((int)atr.size()<=k)
                atr.resize(atr.size()+1);

            atr[k].resize(NGTDM);

            for(int i = 0; i < NGTDM ; i++){

                 atr[k][i] = vet[i];

                 if((atr[k][i] > 100000) || (atr[k][i] < 0.000001)){atr[k][i] = 0;}

            }k++;
        }
    }

    // TexFree(aa);
    TexFree(pixel);
    TexClearVector(atr);

}
//********************************************************************************************************************
void Textura::TexSiNGTDM(vector<vector<int> >pixel,float mat[][3],int distancia){

    for(int x= 0; x < 256;x++){
        mat[x][0] = 0; mat[x][1] = 0; mat[x][2] = 0;
    }
    for(int x = distancia; x < (int)pixel.size(); x++ ){

        for(int y  = distancia; y < (int)pixel[x].size() ; y++){

            int valor = pixel[x][y];

            float media = 0.0;

            for(int i  = x-distancia; i <= x+distancia && i < (int)pixel.size(); i++)
                for(int j = y-distancia; j<= y+distancia && j < (int)pixel[i].size(); j++)
                    if(i != j)media += pixel[i][j];

            media /= (float)(pow(2*distancia+1,2)-1);

            mat[valor][0] +=  (float)fabs((valor - media));

        }
    }
}
//********************************************************************************************************************
void Textura::TexExtractNGTDM(float mat[][3], float vet[],int width){

    float seg = 0,  ng = 0;

    for(int i = 0;i<5;i++)vet[i]= 0;

    for(int x= 0; x < 256;x++){

        if(mat[x][0]!=0){

            mat[x][1] = mat[x][0]/(pow((float)width,2));  // P(i)

            if(mat[x][1]<1){

                mat[x][2] = mat[x][1]* mat[x][0]; // P(i) * S(i)

                vet[0] +=  mat[x][2]; //Calculando Aspereza

                if(mat[x][0]!= 0)ng = ng + 1;  // Segundo termo dos Demais Parametros
                seg += mat[x][0];        // Segundo termo dos Demais Parametros
            }
        }
    }

    for(int x = 0; x <256 ; x++ ){ // Somatório de do 1° termo  das demais funções

        for(int y  = 0; y < 256 ; y++){

            if(mat[x][1]<1 && y != x){

                if(mat[x][1]!= 0 && mat[y][1]!=0 ){

                    float z = x-y; // Calculando Contraste

                    vet[1] += (mat[x][1]*mat[y][1]*pow(z,2)); // Calculando Contraste
                    vet[2] += (x*mat[x][1] - y*mat[y][1]); // Calculando Fineza

                    vet[3] += ((fabs(x-y) / (pow((float)width,2)*(mat[x][1]+mat[y][1]))) *
                            ((mat[x][1] * mat[x][0]) + (mat[y][1] * mat[y][0])));//  Complexidade Final

                    vet[4] += ((mat[x][1] + mat[y][1]) * pow(z,2));// Calculando Força
                }
            }
        }
    }

    vet[0] = 1 / (1+vet[0]); if(vet[0]==1)vet[0] = 0; //Resultado Aspereza

    vet[1] = (vet[1]) * (1.0/(ng*(ng-1))); // Calculando Contraste
    vet[1] = (vet[1]) * (seg/pow((float)width,2));// Contraste Final

    vet[2] = (float)seg / (vet[2]+1);  // Fineza Final

    vet[4] = vet[4] / (1+seg);// Força Final
}
//********************************************************************************************************************
void Textura::TexTU(cv::Mat vtr,int width, int height, vector<vector<float> >&atr){

    vector<int>aa(width);
    vector<vector<int> >pixel(height,aa);

    int k = 0;

    float total[8], ASD[TU], vet[6561][8];

    float** mat = (float**) malloc(8*sizeof(float*));

    atr.resize(vtr.cols*vtr.rows);
    for(int i = 0; i < 8; i++)mat[i] = (float*)calloc((width*height),sizeof(float));

    for(int x = 1; x < vtr.rows; x+=height){ //Contruindo o Espectro de Unidade de Textura

        for(int y = 1; y < vtr.cols; y+=width){

            TexWindow(vtr,pixel,x,y);
            TexExtractTU(pixel,mat);
            TexFrequencyTU(mat,vet,total,width,height);
            TexBwsTU(vet,ASD,total);
            TexGsTU (vet,ASD,total);
            TexDdTU(vet,ASD,total);

            if((int)atr.size() <= k)
                atr.resize(atr.size()+1);

            atr[k].resize(TU);

            for(int i = 0; i < TU ; i++){

                if(isnan(ASD[i]))ASD[i]=0;

                atr[k][i] = ASD[i];

                if((atr[k][i] > 100000) || (atr[k][i] < 0.000001)){atr[k][i] = 0;}
            }k++;
        }
    }
    TexClearVector(atr);

    TexFree(pixel);

    for(int i = 0; i < 8; i ++)
        free(mat[i]);

    free(mat);
}
//********************************************************************************************************************
void Textura::TexExtractTU(vector<vector<int> >pixel,float **mat){ //Antigo builder TU

    float cal[8];
    for(int x =1 ; x < (int)pixel.size()-1; x++){

        for(int y = 1 ; y < (int)pixel[x].size()-1; y++){

            float valor = pixel[x][y];

            cal[0] = pixel[(x-1)][(y-1)]; cal[1] = pixel[x-1][y];
            cal[2] = pixel[(x-1)][(y+1)]; cal[3] = pixel[x][y+1];
            cal[4] = pixel[(x+1)][(y+1)]; cal[5] = pixel[(x+1)][y];
            cal[6] = pixel[(x+1)][(y-1)]; cal[7] = pixel[x][(y-1)];

            if(cal[0] < valor)cal[0] = 0;  if(cal[0] == valor)cal[0] = 1;  if(cal[0] > valor)cal[0] = 2;
            if(cal[1] < valor)cal[1] = 0;  if(cal[1] == valor)cal[1] = 1;  if(cal[1] > valor)cal[1] = 2;
            if(cal[2] < valor)cal[2] = 0;  if(cal[2] == valor)cal[2] = 1;  if(cal[2] > valor)cal[2] = 2;
            if(cal[3] < valor)cal[3] = 0;  if(cal[3] == valor)cal[3] = 1;  if(cal[3] > valor)cal[3] = 2;
            if(cal[4] < valor)cal[4] = 0;  if(cal[4] == valor)cal[4] = 1;  if(cal[4] > valor)cal[4] = 2;
            if(cal[5] < valor)cal[5] = 0;  if(cal[5] == valor)cal[5] = 1;  if(cal[5] > valor)cal[5] = 2;
            if(cal[6] < valor)cal[6] = 0;  if(cal[6] == valor)cal[6] = 1;  if(cal[6] > valor)cal[6] = 2;
            if(cal[7] < valor)cal[7] = 0;  if(cal[7] == valor)cal[7] = 1;  if(cal[7] > valor)cal[7] = 2;

            float ntu = 3.0;
            mat[0][y * pixel.size() + x] = cal[0]*pow(ntu,0) + cal[1]*pow(ntu,1) + cal[2]*pow(ntu,2) + cal[3]*pow(ntu,3) +  cal[4]*pow(ntu,4) +  cal[5]*pow(ntu,5) +  cal[6]*pow(ntu,6)+  cal[7]*pow(ntu,7);
            mat[1][y * pixel.size() + x] = cal[0]*pow(ntu,7) + cal[1]*pow(ntu,0) + cal[2]*pow(ntu,1) + cal[3]*pow(ntu,2) +  cal[4]*pow(ntu,3) +  cal[5]*pow(ntu,4) +  cal[6]*pow(ntu,5)+  cal[7]*pow(ntu,6);
            mat[2][y * pixel.size() + x] = cal[0]*pow(ntu,6) + cal[1]*pow(ntu,7) + cal[2]*pow(ntu,0) + cal[3]*pow(ntu,1) +  cal[4]*pow(ntu,2) +  cal[5]*pow(ntu,3) +  cal[6]*pow(ntu,4)+  cal[7]*pow(ntu,5);
            mat[3][y * pixel.size() + x] = cal[0]*pow(ntu,5) + cal[1]*pow(ntu,6) + cal[2]*pow(ntu,7) + cal[3]*pow(ntu,0) +  cal[4]*pow(ntu,1) +  cal[5]*pow(ntu,2) +  cal[6]*pow(ntu,3)+  cal[7]*pow(ntu,4);
            mat[4][y * pixel.size() + x] = cal[0]*pow(ntu,4) + cal[1]*pow(ntu,5) + cal[2]*pow(ntu,6) + cal[3]*pow(ntu,7) +  cal[4]*pow(ntu,0) +  cal[5]*pow(ntu,1) +  cal[6]*pow(ntu,2)+  cal[7]*pow(ntu,3);
            mat[5][y * pixel.size() + x] = cal[0]*pow(ntu,3) + cal[1]*pow(ntu,4) + cal[2]*pow(ntu,5) + cal[3]*pow(ntu,6) +  cal[4]*pow(ntu,7) +  cal[5]*pow(ntu,0) +  cal[6]*pow(ntu,1)+  cal[7]*pow(ntu,2);
            mat[6][y * pixel.size() + x] = cal[0]*pow(ntu,2) + cal[1]*pow(ntu,3) + cal[2]*pow(ntu,4) + cal[3]*pow(ntu,5) +  cal[4]*pow(ntu,6) +  cal[5]*pow(ntu,7) +  cal[6]*pow(ntu,0)+  cal[7]*pow(ntu,1);
            mat[7][y * pixel.size() + x] = cal[0]*pow(ntu,1) + cal[1]*pow(ntu,2) + cal[2]*pow(ntu,3) + cal[3]*pow(ntu,4) +  cal[4]*pow(ntu,5) +  cal[5]*pow(ntu,6) +  cal[6]*pow(ntu,7)+  cal[7]*pow(ntu,0);
        }
    }
}
//********************************************************************************************************************
void Textura::TexFrequencyTU(float **mat, float vet[][8], float total[], int width, int height){

    int l = 0;

    for(int i = 0; i < 6561; i++)
        for(int j = 0; j < 8; j++)
            vet[i][j] = 0;


    for(int y = 0; y < height; y++){

        for(int x = 0; x < width; x++){

            for(int k = 0; k < 8 ; k++){

                l =(int)mat[k][y* width +x];

                if(isnan(l)==false)
                    vet[l][k] +=1;
            }
        }
    }
    for(int y = 0; y < 8; y++ ){ // Numero total de frequencias

        for(int x = 0; x < 6561; x++){

            if(isnan(vet[x][y]))vet[x][y] = 0;

            total[y]  = total[y] + vet[x][y];
        }
    }
}
//********************************************************************************************************************
void Textura::TexBwsTU(float vet[][8], float ASD[], float total[]){

    float BWS[8] = {0,0,0,0,0,0,0,0};
    for(int j = 0; j<3279; j++){ //somatório BWS

        int i = 3280 + j;
        for(int x = 0; x < 8;x++){

            if(isnan(vet[j][x]))vet[j][x] = 0;
            if(isnan(vet[i][x]))vet[i][x] = 0;

            BWS[x] = BWS[x] + fabs(vet[j][x]- vet[i][x]);
        }
    }

    for(int j = 0; j < 8; j++){

        if(isnan(total[j]))total[j] = 1;
        BWS[j] = (1-(BWS[j] / total[j]))*100;
    }
    for(int k = 0; k<8; k++)
        ASD[k] = BWS[k];
}
//***********************************************************************************************************************************************************
void Textura::TexGsTU(float vet[][8], float ASD[], float total[]){

    float G[4] = {0.0, 0.0, 0.0, 0.0}, GS = 0;

    for(int j = 0; j < 4; j++){ //somatório GS

        for(int i = 0; i < 6561; i++){

            if(isnan(vet[i][j]))vet[i][j] = 0;
            if(isnan(vet[i][j+4]))vet[i][j+4] = 0;

            G[j] = G[j] + fabs(vet[i][j] - vet[i][j+4]);
        }
        if(isnan(total[j]))total[j] =1;

        G[j] =  G[j] / (2*total[j]);  GS = GS + G[j];
    }
    GS = (1 - (0.25 * GS))*100;
    ASD[8] = GS;
}
//***********************************************************************************************************************************************************
void Textura::TexDdTU(float vet[][8], float ASD[], float total[])
{
    float g[4] = {0.0, 0.0, 0.0, 0.0}, gs = 0;

    for(int j = 0; j < 7; j++){ //somatório DD

        for(int i = 0; i < 6561; i++){

            if(isnan(vet[i][j]))vet[i][j] = 0;
            if(isnan(vet[i][j+4]))vet[i][j+1] = 0;

            g[j] += fabs(vet[i][j] - vet[i][j+1]);
        }
        if(isnan(total[j]))total[j] = 0;

        g[j] /= (2*total[j]);

        gs +=  g[j];

        cout<<""; //SE RETIRAR VAI DAR PROBLEMA INESPLICAVEL ATE AGORA
    }
    gs *= ((1-(1/6))*gs)*100;
    ASD[9] = gs;
}
//********************************************************************************************************************
void Textura::TexFree(vector<vector<float> > &vtr){

    vtr.erase(vtr.begin(),vtr.end());
    vtr.clear();
    vtr.shrink_to_fit();
}
//*************************************************************************************************************
void Textura::TexFree(vector<vector<int> > &vtr){

    vtr.erase(vtr.begin(),vtr.end());
    vtr.clear();
    vtr.shrink_to_fit();
}
//*************************************************************************************************************
void Textura::TexFree(vector<vector<double> > &vtr){

    vtr.erase(vtr.begin(),vtr.end());
    vtr.clear();
    vtr.shrink_to_fit();
}
//*************************************************************************************************************
void Textura::TexFree(vector<float> &vtr){

    vtr.erase(vtr.begin(),vtr.end());
    vtr.clear();
    vtr.shrink_to_fit();
}
//*************************************************************************************************************
