#ifndef CNN_H
#define CNN_H

#include<vector>
#include<string>
#include "nets.h"

using namespace std;


//####################################################################################################

class CNN_NEURON{


public:

    CNN_NEURON(int connections);
    ~CNN_NEURON();
    vector<float>weight,input;

    float get_induced_local_field();
    float get_output();
    float get_gradient();
    float get_erro();

    void set_input();

private:
    float vj, output, gradient, erro;

};


//####################################################################################################

class CNN_FRAME{

public:

    CNN_FRAME(int size, int neighbors);
    ~CNN_FRAME();
    CNN_NEURON *neuron;
    vector<float>bias,input, output;

private:
    float vj, output, gradient, erro;
};


//####################################################################################################

class CNN_TUPLE{

public:

    CNN_TUPLE(int frame_size, int n_frames, int c_neighbors, int s_neighbors, int _overlap = 0);
    ~CNN_TUPLE();

    vector<CNN_FRAME>simplex, complex;

    int get_frame_size(int size, int _overlap);

private:

};


//####################################################################################################

class CNN: public Nets{

public:
    CNN(int height, int width, int neighbors = 8, int _overlap = 0 );

    vector<CNN_TUPLE>hidden_layers;
    Layer output_layer; //verificar std::pair

    ~CNN();
private:

    CONFIG_CNN conf;

protected:

};


//####################################################################################################
class CONF_CNN_LAYER{

public:

    //CONF_CNN_LAYER();
    CONF_CNN_LAYER(int _n_frames =10, int _frame_height =15, int _frame_width = 15, int _c_height =5, int _c_width = 5, int _s_height =1, int _s_width=1, int _overlap =1 ,bool _full_connection = false);

    int get_n_frames();
    int get_complex_window_height();
    int get_complex_window_width();
    int get_simplex_window_height();
    int get_simplex_window_width();
    int get_frame_height();
    int get_frame_width();
    int get_frame_size();
    int get_overlap();
    vector<int>get_connections();

    void set_n_frames(int value);
    void set_complex_window_height(int value);
    void set_complex_window_width(int value);
    void set_simplex_window_height(int value);
    void set_simplex_window_width(int value);
    void set_overlap(int value);
    void set_frame_width(int value);
    void set_frame_height(int value);
    void set_frame_size(int value);

    void set_full_connect(bool value =1);
    void set_layer_connection(vector<int>frames);


 private:

    int n_frames, c_height, c_width, s_height, s_width, overlap, frame_height, frame_width, frame_size;
    bool full;
    vector<int>connection_conf;

};



//####################################################################################################
class CONFIG_CNN{

public:

    CONFIG_CNN(int n_layer);

    int get_n_layers();

    vector<CONF_CNN_LAYER>*set_layer(int number);


private:

    vector<CONF_CNN_LAYER>cnn_layer;


};


//####################################################################################################

#endif // CNN_H
