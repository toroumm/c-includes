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
CNN_FRAME::CNN_FRAME(int size, int neightbors){

    random_device rd; mt19937 gen(rd());
    normal_distribution <>d(0,0.001);

    for(int i = 0; i < size; i++){
           bias.push_back(d(gen));
    }
    neuron = new CNN_NEURON(neightbors);
}


//****************************************************************************
CNN_TUPLE::CNN_TUPLE(int frame_size,int n_frames, int neighbors, int _overlap ){

    int complex_size = get_frame_size(frame_size,__overlap);
    int simplex_size = get_frame_size(complex_size,_overlap);

    for(int i = 0; i < n_frames; i++){
        complex.push_back(CNN_FRAME(complex_size,neighbors));
        simplex.push_back(CNN_FRAME(simplex_size,neighbors));
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



//****************************************************************************

CNN::CNN(int height, int width, int n_frames, int neighbors = 8, int _overlap = 0){

    int size = height*width;
    for(int i = 0;i < _layers; i++){

        hidden_layers.push_back(CNN_TUPLE(size,n_frames,neighbors, _overlap));
    }

}
//****************************************************************************



