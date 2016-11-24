#include "cnn.h"
#include <algorithm>
#include <random>

//############################################################################
CNN_NEURON::CNN_NEURON(int connections){

    weight.size(connections);

    random_device rd; mt19937 gen(rd());
    normal_distribution<>d(0,0.01);
    for(int i = 0; i <connections; i++){
        weight.push_back(d(gen));
    }
    vj = 0;
    output = 0;
    gradient =0;
    erro = 0;

}

//****************************************************************************
CNN_FRAME::CNN_FRAME(int size, int neighbors){

    random_device rd; mt19937 gen(rd());
    normal_distribution <>d(0,0.001);

    for(int i = 0; i < size; i++){
           bias.push_back(d(gen));
    }
    neuron = new CNN_NEURON(neighbors);
}


//****************************************************************************
CNN_TUPLE::CNN_TUPLE(int frame_size, int n_frames, int c_neighbors, int s_neighbors, int _overlap = 0 ){

    int complex_size = get_frame_size(frame_size,__overlap);
    int simplex_size = get_frame_size(complex_size,_overlap);

    for(int i = 0; i < n_frames; i++){
        complex.push_back(CNN_FRAME(complex_size,c_neighbors));
        simplex.push_back(CNN_FRAME(simplex_size,s_neighbors));
    }
}


int CNN_TUPLE::get_frame_size(int size, int _overlap){

    int half, frac_part;

    frac_part = modf(sqrt(size),&half);

    if(frac_part > 0)
        half++;

    if(remainder((half+_overlap),neighbors) > 0)
        return pow(((half+_overlap)/neighbors)+1,2);
    else
        return  pow(((half+_overlap)/neighbors),2);
}


//****************************************************************************


CNN::CNN(CONFIG_CNN configuration){

    conf = configuration;

    for(int i = 0; i < (int)conf.get_n_layers(); i++){

        CONF_CNN_LAYER *cl = conf.set_layer(i);
        hidden_layers.push_back(CNN_TUPLE(cl->get_frame_size(),cl->get_n_frames(),cl->get_complex_window_height()*cl->get_complex_window_width(),cl->get_simplex_window_height()*cl->get_simplex_window_width(),cl->get_overlap()));
    }

}
//****************************************************************************
/*
CONF_CNN_LAYER::CONF_CNN_LAYER(){

    set_full_connect(1);
    set_n_frames(10);
    set_window_height(5);
    set_window_width(5);

}*/
/**
 * @brief CONF_CNN_LAYER::CONF_CNN_LAYER: This Contrututor make a model of each tuple convolution layer.
 * @param _n_frames : int Number of frames on this layer
 * @param _height : int Height's of convolution window
 * @param _width : int  WIdths's of convolution window
 * @param _overlap : int Overlap on complex window convolution
 * @param _full_connection: bool
 */

CONF_CNN_LAYER::CONF_CNN_LAYER(int _n_frames, int _frame_height, int _frame_width, int _c_height, int _c_width, int _s_height, int _s_width, int _overlap  ,bool _full_connection ){

    set_full_connect(_full_connection);
    set_n_frames(_n_frames);
    set_complex_window_height(_c_height);
    set_complex_window_width(_c_width);
    set_simplex_window_height(_s_height);
    set_simplex_window_width(_s_width);
    set_overlap(_overlap);
    set_frame_height(_frame_height);
    set_frame_width(_frame_width);
    set_frame_size(get_frame_height()*get_frame_width());
}

void CONF_CNN_LAYER::set_full_connect(bool value){full = value;}
void CONF_CNN_LAYER::set_layer_connection(vector<int> frames){set_full_connect(0);connection_conf = frames;}
void CONF_CNN_LAYER::set_n_frames(int value){n_frames =value;}
void CONF_CNN_LAYER::set_overlap(int value){overlap =value;}
void CONF_CNN_LAYER::set_complex_window_height(int value){c_height = value;}
void CONF_CNN_LAYER::set_complex_window_width(int value){c_width =value;}
void CONF_CNN_LAYER::set_simplex_window_height(int value){s_height =value;}
void CONF_CNN_LAYER::set_simplex_window_width(int value){s_width = value;}
void CONF_CNN_LAYER::set_frame_size(int value){frame_size =value;}
void CONF_CNN_LAYER::set_frame_height(int value){frame_height = value;set_frame_size(frame_height*frame_width);}
void CONF_CNN_LAYER::set_frame_width(int value){frame_width =value;set_frame_size(frame_height*frame_width);}

bool CONF_CNN_LAYER::get_n_frames(){return n_frames;}
vector<int> CONF_CNN_LAYER::get_connections(){return connection_conf;}
int CONF_CNN_LAYER::get_overlap(){return overlap;}
int CONF_CNN_LAYER::get_complex_window_height(){return c_height;}
int CONF_CNN_LAYER::get_complex_window_width(){return c_width;}
int CONF_CNN_LAYER::get_simplex_window_height(){return s_height;}
int CONF_CNN_LAYER::get_simplex_window_width(){return s_width;}
int CONF_CNN_LAYER::get_frame_height(){return frame_height;}
int CONF_CNN_LAYER::get_frame_width(){return frame_width;}
int CONF_CNN_LAYER::get_frame_size(){return frame_size;}

/**
 * @brief CONFIG_CNN::CONFIG_CNN This contructor start the setup of each convolution layer of model
 * @param n_layer : Tuple's number of your convolution model
 */

CONFIG_CNN::CONFIG_CNN(int n_layer){

    cnn_layer.resize(n_layer);
}

/**
 * @brief CONFIG_CNN::set_layer : Give access to any tuple to setup
 * @param number : This parameter it's a number of tuple which you need setup
 * @return desire tuple
 */

vector<CONF_CNN_LAYER> *CONFIG_CNN::set_layer(int number){
    if(number>0 && number < cnn_layer.size())
        return &cnn_layer[number];
    else
         return 0;
}

int CONFIG_CNN::get_n_layers(){return cnn_layer.size();}
//****************************************************************************


