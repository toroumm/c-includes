#include "utilitarios.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <dirent.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//template <typename T>


Utilitarios::Utilitarios(){ }

void Utilitarios::concatenate_vector(vector<vector<float> > &to, vector<vector<float> > from){

    for(int i = 0; i< (int)from.size();i++)
        to.push_back(from[i]);

}

//************************************************************************************


void Utilitarios::list_directory(vector<string> &list_dir, string path){

    DIR *pDIR;
    struct dirent *entry;


    if( pDIR=opendir(path.c_str()) ){
        while(entry = readdir(pDIR)){


            if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 ){

                list_dir.push_back(entry->d_name);
            }
        }
        closedir(pDIR);
    }
}

//************************************************************************************

void Utilitarios::saveTXTFile(vector<vector<float> > samples, string path, bool save_to_c){

    ofstream file;
    file.open(path);
    if(!file.is_open()){
        cout<<"Erro ao abrir arquivo: "<<path+".txt"<<endl;
        return;
    }
    if(save_to_c ==1)
        file<<samples.size()<<" "<<samples[0].size()<<endl;

    for(int i = 0; i < (int)samples.size(); i++){
        for(int j =0; j < (int)samples[i].size();j++){

            file<<samples[i][j]<<" ";
        }file<<endl;

    }
    file.close();
}
//************************************************************************************

void Utilitarios::saveTXTFile(vector<vector<string> > samples, string path){
    
    ofstream file;
    file.open(path+".txt");
    if(!file.is_open()){
        cout<<"Erro ao abrir arquivo: "<<path+".txt" <<endl;
        return;
    }
    
    file<<samples.size()<<" "<<samples[0].size()<<endl;
    
    for(int i = 0; i < (int)samples.size(); i++){
        for(int j =0; j < (int)samples[i].size();j++){
            
            file<<samples[i][j]<<" ";
        }file<<endl;
        
    }
    file.close();
}

//************************************************************************************

void Utilitarios::saveTXTFile(vector<vector<int> >samples,string path){

    ofstream file;
    file.open(path+".txt");
    if(!file.is_open()){
        cout<<"Erro ao abrir arquivo: "<<path+".txt"<<endl;
        return;
    }

    file<<samples.size()<<" "<<samples[0].size()<<endl;

    for(int i = 0; i < (int)samples.size(); i++){
        for(int j =0; j < (int)samples[i].size();j++){

            file<<samples[i][j]<<" ";
        }file<<endl;

    }
    file.close();
}

//************************************************************************************

void Utilitarios::saveTXTFile(vector<int >&samples,string path){
    
    ofstream file;
    file.open(path);
    if(!file.is_open()){
        cout<<"Erro ao abrir arquivo: "<<path+".txt"<<endl;
        return;
    }
    
    file<<samples.size()<<endl;
    
    for(int i = 0; i < (int)samples.size(); i++)
        file<<samples[i]<<endl;
    
    file.close();
}

//*******************************************************************************************
void Utilitarios::loadTXTFile(vector<float>&samples , string path){

    ifstream file(path);

    samples.clear();

    if(!file.is_open()){
        cout<<"Erro ao abrir arquivo: "<<path<<endl;
        return;
    }
    string linha;

    for(int i = 0; getline(file,linha); ++i){

        istringstream iss(linha);

        copy(istream_iterator<float>(iss), istream_iterator<float>(), back_inserter(samples));

    }
}

//*******************************************************************************************

void Utilitarios::loadTXTFile(vector<vector<float> >&samples,string path){
    
    
    /* ifstream file;
    file.open(path);
    if(!file.is_open()){
        cout<<"Erro ao abrir arquivo: "<<path<<endl;
        return;
    }

    int height, width ;
    file>>height; file>>width;

    samples.resize(height);
    for(int i = 0; i < (int)samples.size();i++){
        samples[i].resize(width);
        for(int j = 0; j < (int)samples[i].size(); j++){
            file>>samples[i][j];
            if(isnan(samples[i][j]))
                samples[i][j] = 0.0;
        }

    }
    file.close();*/

 /*  for(int i = 0; i < (int)samples.size(); i++){
        if(samples[i].size()> 0)
            samples[i].clear();
   }
    samples.clear();*/
    ifstream file(path);

    if(!file.is_open()){
        cout<<"Erro ao abrir arquivo: "<<path<<endl;
        return;
    }
    string linha;


    for(int i = 0; getline(file,linha); ++i){

        istringstream iss(linha);
        vector<float>redes;

        copy(istream_iterator<float>(iss), istream_iterator<float>(), back_inserter(redes));

        samples.push_back(redes);

    }
}

//*******************************************************************************************

void Utilitarios::loadTXTFile(vector<vector<int> >&samples,string path){

    ifstream file;
    file.open(path);
    if(!file.is_open()){
        cout<<"Erro ao abrir arquivo: "<<path<<endl;
        return;
    }

    int height, width;
    file>>height; file>>width;

    samples.resize(height);
    for(int i = 0; i < (int)samples.size();i++){
        samples[i].resize(width);
        for(int j = 0; j < (int)samples[i].size(); j++)
            file>>samples[i][j];
    }
    file.close();
}

//*******************************************************************************************

void Utilitarios::loadTXTFile(vector<int >&samples,string path){

    ifstream file;
    file.open(path);
    if(!file.is_open()){
        cout<<"Erro ao abrir arquivo: "<<path<<endl;
        return;
    }

    int height;
    file>>height;

    samples.resize(height);
    for(int i = 0; i < (int)samples.size();i++){
        file>>samples[i];
    }
    file.close();
}

//*******************************************************************************************

void Utilitarios::loadTXTFile(vector<vector<string> >&samples,string path){

    ifstream file;
    file.open(path);
    if(!file.is_open()){
        cout<<"Erro ao abrir arquivo "<<path<<endl;
        return;
    }

    int height, width;
    file>>height; file>>width;


    samples.resize(height);
    for(int i = 0; i < (int)samples.size();i++){
        samples[i].resize(width);
        for(int j = 0; j < (int)samples[i].size(); j++)
            file>>samples[i][j];
    }
    file.close();

}
//*******************************************************************************************
void Utilitarios::extract_output(vector<vector<float> > &samples, vector<vector<float> > &out, int col_output){

    vector<vector<float> >new_sample;

    new_sample.resize(samples.size());
    out.resize(samples.size());
    for(int i  = 0; i < (int)samples.size();i++){

        for(int j = 0; j < (int)samples[i].size();j++){

            if(col_output < 0 && j >= (int)samples[i].size()+col_output){
                out[i].push_back(samples[i][j]);
            }else
                new_sample[i].push_back(samples[i][j]);
        }
    }
    samples = new_sample;
}



void Utilitarios::mergeMatrix(vector<vector<float> >&samples,vector<vector<float> >&out, vector<vector<float> >in,vector<int>setOut){
    
    int jj = samples.size(),ii = samples.size();
    
    samples.resize(samples.size()+in.size());
    out.resize(out.size()+in.size());
    
    for(; ii < (int)samples.size();ii++){
        samples[ii]  = in[ii-jj];
        
        out[ii].resize(setOut.size());
        for(int j = 0; j < (int)setOut.size(); j++)
            out[ii][j]  = setOut[j];
        
    }
}
//*******************************************************************************************

void Utilitarios::mergeVector(vector<vector<float> > &in, vector<vector<float> > two,vector<vector<float> > &out,vector<vector<float> > two_out){

    if(in.size() != out.size()){cout<<"Utilitatios::Merge Vector: Problem size of vector IN and OUT "<<endl; return;}


    if(two.size() != two_out.size()){cout<<"Utilitatios::Merge Vector: Problem size of vector IN and OUT "<<endl; return;}

    if(two.size() <= 0 || two_out.size() <= 0 || in.size() <= 0 || out.size() <=0 ){cout<<"Utilitarios::Merge Vector: Problem empty vector"; return;}

    for(int i = 0; i < (int)two.size();i++){
        in.push_back(two[i]);
        out.push_back(two_out[i]);
    }
}

//*******************************************************************************************
void Utilitarios::saveXMLFile(vector<vector<float> >samples,string path){

    path +=".xml";
    
    cv::Mat mat(samples.size(),samples[0].size(),CV_32FC1);
    
    cv::FileStorage xmlFile(path,cv::FileStorage::WRITE);
    
    for(int x = 0; x < (int)samples.size();x++)
        for(int y = 0; y < (int)samples[x].size(); y++)
            mat.at<float>(x,y) = (float)samples[x][y];

    xmlFile <<"Atributos"<<mat;
    
    xmlFile.release();
}


//*******************************************************************************************
void Utilitarios::loadXMLFile(vector<vector<float> >&samples,string path){



    cv::Mat mat;
    cv::FileStorage xmlFile(path,cv::FileStorage::READ);
    
    xmlFile["Atributos"] >> mat;
    
    std::vector<float> vec;
    cv::FileNode data = xmlFile["Atributos"];
    cv::FileNodeIterator it = data.begin();
    (*it)["data"] >>vec;
    
    samples.clear();
    samples.resize(mat.rows);
    for(int i = 0; i < mat.rows; i++)
        samples[i].resize(mat.cols);
    for(int i = 0; i < mat.rows; i++){

        for(int j = 0; j < mat.cols; j++){
            if( isnan(vec[i*mat.cols+j]) || isinf(vec[i*mat.cols+j]))
                samples[i][j]  = 0.0;
            else
                samples[i][j] = (float)vec[i*mat.cols+j];
        }
    }
    
    xmlFile.release();
}
//*******************************************************************************************
void Utilitarios::appendXMLFile(vector<vector<float> > &samples, string path){


    if(path.empty()){
        cout<<"Path is empty !"<<endl;
        return;
    }

    cv::Mat matf;
    cv::FileStorage xmlFile(path,cv::FileStorage::READ);

    if(!xmlFile.isOpened()){

        cout<<"It isn't possible open XML file"<<endl;
        return;
    }

    xmlFile["Atributos"] >> matf;

    std::vector<float> vec;
    cv::FileNode data = xmlFile["Atributos"];
    cv::FileNodeIterator it = data.begin();
    (*it)["data"] >>vec;

    xmlFile.release();
    cv::Mat real((samples.size()+matf.rows), matf.cols,CV_32FC1);

    for(int x = 0; x < (int)samples.size();x++)
        for(int y = 0; y < (int)samples[x].size(); y++) //tec é o vetor que conter o numeros de atributos que a técnica possui no caso da COM 6
            real.at<float>(x,y) = samples[x][y];

    int k = 0;
    for(int x = (int)samples.size(); x < (int)(samples.size()+matf.rows);x++)
        for(int y = 0; y < matf.cols; y++){
            real.at<float>(x,y) = (float)vec[k];
            k+=1;
        }

    cv::FileStorage xmlFile2(path,cv::FileStorage::WRITE);

    xmlFile2<<"Atributos"<<real;

    xmlFile2.release();

}
//*******************************************************************************************
void Utilitarios::print(vector<vector<float> > samples){

    for(int i = 0; i < (int)samples.size(); i++){
        cout<<i<<" ";
        for(int j =0; j < (int)samples[i].size(); j++){

            cout<<samples[i][j]<<" ";
        }cout<<endl;
    }
}

void Utilitarios::print(vector<vector<int> > samples){

    for(int i = 0; i < (int)samples.size(); i++){
        cout<<i<<" ";
        for(int j =0; j < (int)samples[i].size(); j++){

            cout<<samples[i][j]<<" ";
        }cout<<endl;
    }
}
//*******************************************************************************************
void Utilitarios::print(vector<vector<string> > samples){

    for(int i = 0; i < (int)samples.size(); i++){
        cout<<i<<" ";
        for(int j =0; j < (int)samples[i].size(); j++){

            cout<<samples[i][j]<<" ";
        }cout<<endl;
    }
}
//*******************************************************************************************
void Utilitarios::desnormalize(vector<vector<float> > &samples, int max){

    for(int i = 0; i < (int)samples.size(); i++)
        for(int j =0; j < (int)samples[i].size(); j++)
            samples[i][j] = max * samples[i][j];

}
//*******************************************************************************************
void Utilitarios::normalize(vector<vector<float> >&samples){


    vector<float>max,min;

    for(int j = 0; j < (int)samples[0].size(); j++){

        max.push_back(samples[0][j]);
        min.push_back(samples[0][j]);

        for(int i = 1; i < (int)samples.size(); i++){
            if(max[max.size()-1] < samples[i][j])
                max[max.size()-1] = samples[i][j];

            else{
                if(min[min.size()-1] > samples[i][j])
                    min[min.size()-1] = samples[i][j];
            }
        }
    }


    for(int j = 0; j < (int)min.size(); j++){

        if(min[j] > 0)
            min[j] =0;

        if(max[j] == 0)
            max[j] =1;

        for(int i = 0; i < (int)samples.size(); i++){
            samples[i][j] = (samples[i][j]-min[j])/(max[j]-min[j]);

        }
    }
}


void Utilitarios::normalize(vector<vector<float> >&samples,int max){

    for(int i = 0; i < (int)samples.size(); i++){

        for(int j = 0; j < (int)samples[i].size(); j++)
            samples[i][j] = (samples[i][j])/(max);

    }
}
//*******************************************************************************************
void Utilitarios::shuffle_samples(vector<vector<float> > &samples){
    random_shuffle(samples.begin(),samples.end());
}


void Utilitarios::shuffle_io_samples(vector<vector<float> > &samples, vector<vector<float> > &out){

    if(samples.size() != out.size()){cout<<"Utilitarios::shuffle_io_samples: There's Diferente size in samples and output' "<<endl; return;}

    vector<vector<float> >reserva;
    reserva.resize(samples.size());

    for(int i = 0; i < (int)reserva.size();i++){
        for(int j = 0; j < (int)samples[i].size(); j++)
            reserva[i].push_back(samples[i][j]);

        for(int j = 0; j < (int)out[i].size(); j++)
            reserva[i].push_back(out[i][j]);
    }

    random_shuffle(reserva.begin(),reserva.end());

    for(int i = 0; i < (int)reserva.size();i++){
        for(int j = 0; j < (int)samples[i].size(); j++)
            samples[i][j] = reserva[i][j];

        for(int j = 0; j < (int)out[i].size(); j++)
            out[i][j] = reserva[i][j+samples[i].size()];
    }
}
//*******************************************************************************************
void Utilitarios::on_cross_validation(vector<vector<float> > &samples, vector<vector<float> > &out,vector<vector<float> > &validation,vector<vector<float> > &validation_out, float percent){

    if(percent <= 0.000001)
        return;
    int fold = samples.size() / float(samples.size()*percent);

    get_extratified_cross_validation(fold,samples,out);
    get_extratified_combination(0,samples,out,validation,validation_out);

}
//***************************************************************************
vector<vector<float> > Utilitarios::matrix_confusion(vector<vector<float> > get, vector<vector<float> > desire,  float trunc_get_data){

    vector<vector<float> >pattern;
    //get Classes
    int check;
    for(int i = 1; i < (int)desire.size(); i++){
        check = 0;
        for(int j = 0; j < (int)pattern.size();j++)
            if(desire[i] == pattern[j]){check =1;break;}

        if(check ==0)
            pattern.push_back(desire[i]);
    }

    cout<<"Classes "<<pattern.size()<<endl;

    vector<float>re(pattern.size());
    vector<vector<float> >result(re.size(),re);

    cout<<"Confere Matrix Conf "<<result.size()<<" "<<result[0].size()<<endl;

    //get matrix confusion
    for(int i = 0; i < (int)desire.size(); i++){
        int a,b;

        //Trunc data
        for(int j = 0; j < (int)desire[i].size();j++){

            if(get[i][j] >= trunc_get_data)
                get[i][j] = 1.0;
            else
                get[i][j] = 0.0;

            if(desire[i][j] >= trunc_get_data)
                desire[i][j] = 1.0;
            else
                desire[i][j] = 0.0;
        }

        for(int j = 0; j < (int)pattern.size(); j++){
            if(desire[i] == pattern[j])
                a = j;

            if(get[i] == pattern[j])
                b = j;
        }
        result[a][b] +=1;
    }
    return result;
}
//************************************************************************************
vector<float>Utilitarios::histogram(vector<float> samples, float min, float max, float bin_size){

    vector<float>hist((max-min)/bin_size);

    float upper = bin_size;

    float lower = min;

    for(int i = 0; i < (int)samples.size();i++){

        for(int j = 0; j < (int)hist.size(); j++){

            if(samples[i] <= upper && samples[i] > lower){
                hist[j]++;
                break;
            }
        }
    }
    return hist;
}
//************************************************************************************
float Utilitarios::shannon_entropy(vector<vector<float> > samples, float min, float max, float bin_size){


    float info_content = 0.0;

    for(int i = 0; i < (int)samples[0].size(); i++){

        vector<int>hist(((max-min)/bin_size)+2);
        for(int j = 0; j < (int)samples.size(); j++){

            float upper = bin_size;

            float lower = min;

            for(int k = 0; k < (int)hist.size(); k++){

                if(samples[j][i] <= upper && samples[j][i] > lower){
                    hist[k]++;

                    break;
                }
                upper+=bin_size;

                lower+=bin_size;
            }
        }

        for(int j = 0; j < (int)samples.size(); j++){

            float upper = bin_size;

            float lower = min;

            for(int k = 0; k < (int)hist.size(); k++){

                if(samples[j][i] <= upper && samples[j][i] > lower){

                    float freq = float(float(hist[k]) / samples.size());

                    if(isnormal(freq))
                        info_content += freq * log2(freq);

                    // cout<<hist[k]<<" "<<freq<<" "<<log2(freq)<<"  "<<info_content<<"  "<<freq * log2(freq)<<endl;
                }
                upper+=bin_size;

                lower+=bin_size;
            }
        }
    }

    return info_content*(-1);
}

//************************************************************************************

float Utilitarios::shannon_entropy_m(vector<vector<float> > samples, float min, float max, float bin_size){


    float info_content = 0.0;
    vector<int>hist(((max-min)/bin_size)+2);
    for(int i = 0; i < (int)samples[0].size(); i++){

        for(int j = 0; j < (int)samples.size(); j++){

            float upper = bin_size;

            float lower = min;

            for(int k = 0; k < (int)hist.size(); k++){

                if(samples[j][i] <= upper && samples[j][i] > lower){
                    hist[k]++;

                    break;
                }
                upper+=bin_size;

                lower+=bin_size;
            }
        }
    }

    for(int i = 0; i < (int)samples[0].size(); i++){


        for(int j = 0; j < (int)samples.size(); j++){

            float upper = bin_size;

            float lower = min;

            for(int k = 0; k < (int)hist.size(); k++){

                if(samples[j][i] <= upper && samples[j][i] > lower){

                    float freq = float(float(hist[k]) / (samples.size()*samples[i].size()));

                    if(isnormal(freq))
                        info_content += freq * log2(freq);

                    // cout<<hist[k]<<" "<<freq<<" "<<log2(freq)<<"  "<<info_content<<"  "<<freq * log2(freq)<<endl;
                }
                upper+=bin_size;

                lower+=bin_size;
            }
        }

    }

    return info_content*(-1);
}

//************************************************************************************
int Utilitarios::number_neurons_shannon_entropy(vector<vector<float> > samples,float min, float max, float bin_size){

    return 64*(shannon_entropy(samples,min,max,bin_size) / samples[0].size());
}
//************************************************************************************
bool    Utilitarios::get_split_classes(vector<vector<float> > samples, vector<vector<float> > out){

    int classes = 0;
    vector<string> freq,freqAux;

    classes = get_number_classes(freq,freqAux,out);

    split_class.clear();
    split_classout.clear();
    split_class.resize(classes); int zz = 0;
    split_classout.resize(classes);
    vector<int>nItens(classes);

    for(int i = 0; i < (int)samples.size(); i++){
        if(freq[i] != "-10"){
            for(int j = 0; j < (int)samples.size();j++){
                if(freq[i] == freqAux[j]){
                    nItens[zz]++;

                }
            }
            zz++;
        }
    }
    for(int i = 0; i <(int)split_class.size(); i++){
        split_class[i].resize(nItens[i]);
        split_classout[i].resize(nItens[i]);
    }

    int classe  = 0;  int x = 0;
    for(int i  = 0; i < (int)samples.size(); i++){

        if(freq[i] != "-10"){

            for(int j = 0; j < (int)samples.size(); j++){

                if(freq[i] == freqAux[j]){

                    for(int k = 0; k < (int)samples[j].size(); k++){
                        split_class[classe][x].push_back(samples[j][k]);
                    }

                    for(int k = 0; k < (int)out[j].size(); k++){
                        split_classout[classe][x].push_back(out[j][k]);

                    }x++;
                }
            }x = 0; classe++;
        }
    }
    if(classes ==0 )
        return false;
    return true;
}
//************************************************************************************
int Utilitarios::get_number_classes(vector<string> &freq_one, vector<string> &freq_two, vector<vector<float> > out){

    freq_one.resize(out.size());
    freq_two.resize(out.size());

    int classes = 0;
    for(int  i = 0; i < (int)out.size();i++){
        for(int j = 0; j < (int)out[i].size(); j++){
            stringstream s; s << out[i][j];
            freq_one[i] += s.str();
            freq_two[i]+= s.str();
        }
    }

    for(int i =0; i < (int)out.size(); i++)
        for(int j = 0; j < (int)out.size();j++)
            if(i != j)
                if(freq_one[i] == freq_one[j])
                    freq_one[j] = "-10";

    for(int i =0; i < (int)out.size(); i++)
        if(freq_one[i] != "-10")
            classes++;

    return classes;
}
//************************************************************************************
bool Utilitarios::get_extratified_combination(int index, vector<vector<float> > &samples, vector<vector<float> > &out, vector<vector<float> > &test, vector<vector<float> > &testout){

    samples.clear(); out.clear(); test.clear(); testout.clear();
    test = extratified_data[index];
    testout = extratified_dataout[index];

    for(int i = 0; i < (int)extratified_data.size();i++)
        if(i != index){

            samples.reserve(extratified_data[i].size()+samples.size());
            samples.insert(samples.end(),extratified_data[i].begin(), extratified_data[i].end());
            out.reserve(extratified_dataout[i].size()+out.size());
            out.insert(out.end(),extratified_dataout[i].begin(), extratified_dataout[i].end());
        }
    return true;
}
//************************************************************************************
bool Utilitarios::get_extratified_cross_validation(int kfold, vector<vector<float> > samples, vector<vector<float> > out){

    if(kfold == 0 || samples.size() == 0 || out.size() == 0)
        return false;

    if(samples.size() != out.size()){

        cout<<" Error in file size: samples.size() = "<<samples.size()<< "  != out.size () = "<<out.size()<<endl;
        return false;
   }

    if(get_split_classes(samples,out) ==false){ cout<<"Utilitarios::get_extratified_cross_validation: Data base classes not found "<<endl; exit(0);}


    extratified_data.clear();
    extratified_dataout.clear();

    extratified_data.resize(kfold);
    extratified_dataout.resize(kfold);
    vector<int>ref(split_class.size());

    for(int i = 0; i < kfold; i++){

        int j = 0;
        while(j < (int)split_class.size()){

            for(int k = ref[j]; k < ref[j]+int(split_class[j].size()/kfold); k++){
                extratified_data[i].push_back(split_class[j][k]);
                extratified_dataout[i].push_back(split_classout[j][k]);
            }

            ref[j] += int(split_class[j].size()/kfold);
            j++;
        }
    }return true;
}
//************************************************************************************
