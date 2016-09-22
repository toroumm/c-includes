#ifndef UTILITARIOS_H
#define UTILITARIOS_H

#include <vector>
#include <string>

using namespace std;

class Utilitarios
{
private:


public:


    Utilitarios();


    void list_directory(vector<string>&list_dir,string path);

    void saveTXTFile(vector<vector<string> > samples, string path);
    void saveTXTFile(vector<vector<int> >samples,string path);
    void saveTXTFile(vector<vector<float> > samples, string path, bool save_to_c =true);

    void saveTXTFile(vector<int> &samples, string path);

    void loadTXTFile(vector<vector<string> >&samples,string path);
    void loadTXTFile(vector<vector<float> >&samples,string path);
    void loadTXTFile(vector<int> &samples,string path);
    void loadTXTFile(vector<vector<int> >&samples,string path);
    void loadTXTFile(vector<float>&samples, string path);


    void saveXMLFile(vector<vector<float> >samples,string path);
    void loadXMLFile(vector<vector<float> > &samples, string path);
    void appendXMLFile(vector<vector<float> >&samples,string path);

    void mergeMatrix(vector<vector<float> >&samples,vector<vector<float> >&out, vector<vector<float> >in,vector<int>setOut);
    void mergeVector(vector<vector<float> > &in, vector<vector<float> >two, vector<vector<float> > &out, vector<vector<float> > two_out);
    void concatenate_vector(vector<vector<float> >&to, vector<vector<float> >from);

    void on_cross_validation(vector<vector<float> > &samples, vector<vector<float> > &out,vector<vector<float> > &validation,vector<vector<float> > &validation_out, float percent);

    void normalize(vector<vector<float> >&samples,int max);
    void normalize(vector<vector<float> > &samples);

    void desnormalize(vector<vector<float> >&samples,int max);

    void shuffle_io_samples(vector<vector<float> >&samples,vector<vector<float> >&out);
    void shuffle_samples(vector<vector<float> >&samples);

    void extract_output(vector<vector<float> >&samples,vector<vector<float> >&out, int col_output);


    vector<vector<float> > matrix_confusion(vector<vector<float> >get, vector<vector<float> >desire, float trunc_get_data = 0.5);

    vector<float>histogram(vector<float>samples,float min, float max, float bin_size);
    float shannon_entropy(vector<vector<float> > samples, float min, float max, float bin_size);
    float shannon_entropy_m(vector<vector<float> > samples, float min, float max, float bin_size);


    int number_neurons_shannon_entropy(vector<vector<float> > samples, float min, float max, float bin_size);

    void print(vector<vector<float> > samples);
    void print(vector<vector<int> > samples);
    void print(vector<vector<string> > samples);

    //extratified cross validation
    vector<vector<vector<float> > >extratified_data, extratified_dataout,split_class,split_classout;

    int  get_number_classes(vector<string>&freq_one, vector<string>&freq_two, vector<vector<float> >out );
    bool get_split_classes(vector<vector<float> > samples,vector<vector<float> > out);
    bool get_extratified_combination(int index,vector<vector<float> > &samples,vector<vector<float> > &out, vector<vector<float> > &test,  vector<vector<float> > &testout);
    bool get_extratified_cross_validation(int kfold, vector<vector<float> > samples,vector<vector<float> > out);
};

#endif // UTILITARIOS_H
