#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "neuron.h"

using namespace std;

//###########################################################################################################
class Layer{

public:
    Layer();
    Layer(int neurons, int connections);
    ~Layer();
    
    enum ACT_FUNC_TYPE{ SIGMOIDAL =1, HYPERBOLIC =0, SOFTPLUS = 2, SOFTMAX =3};


    Neuron get_one_neuron(int index);
    vector<Neuron> get_neurons();

    void set_activation_function(ACT_FUNC_TYPE FUNCTION);
    void set_param_tanh(float a = 0.1, float b = 0.2);
    
    
    float get_tanh_a();
    float get_tanh_b();
    
    void start_layer(int n_neurons, int connections);
        
    
protected:
    float (Layer::*activation_function)(float);
    float (Layer::*derivate_function)(float);
    
    vector<Neuron>neurons;
    
    //Activation Function

    float tangent_hyperbolic(float vj);
    float derivate_tanh_hyperbolic(float vj);
    float sigmoidal(float vj);
    float derivate_sigmoidal(float vj);
    float rectifier(float vj); //ReLu
    float derivate_rectifier(float vj);

private:    

    float tanh_a, tanh_b;
    
};

class Layer_MLP{

public:

    Layer_MLP(int n_Neuron_MLP, int connections);

    vector<Neuron>neurons;
};

//###########################################################################################################


#endif //LAYER_H
