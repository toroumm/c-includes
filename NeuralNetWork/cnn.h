#ifndef CNN_H
#define CNN_H

#include<vector>
#include<string>
#include "nets.h"

using namespace std;

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

class CNN_FRAME{

public:

    CNN_FRAME(int size, int neightbors);
    ~CNN_FRAME();
    CNN_NEURON *neuron;
    vector<float>bias,input, output;

private:
    float vj, output, gradient, erro;
};

class CNN_TUPLE{

public:

    CNN_TUPLE(int frame_size, int n_frames, int neighbors, int _overlap = 0);
    ~CNN_TUPLE();

    vector<CNN_FRAME>simplex, complex;

    int get_frame_size(int size, int _overlap);


private:

};


class CNN: public Nets{

public:
    CNN(int height, int width, int n_frames, int neighbors = 8, int neighbors = 8, int _overlap = 0 );

    vector<CNN_TUPLE>hidden_layers;
    Layer output_layer; //verificar std::pair

    ~CNN();
private:

protected:

};

#endif // CNN_H
