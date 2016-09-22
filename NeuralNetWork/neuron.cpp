#include <math.h>
#include <algorithm>
#include <iostream>
#include "neuron.h"

using namespace std;
//###########################################################################################################

//****************************************************************
Neuron::Neuron(){}

Neuron::Neuron(int connections){
    start_neuron(connections);
}

Neuron::~Neuron(){}

//****************************************************************
/**
 * @brief Neuron::start_neuron
 * @param connections
 *
 * Type = public
 *
 * Description:
 *
 * It's a constructor of the Neuron class, including weight inicizalization using a normal distribution, and bias.
 *
 * Parameters:
 *
 *  Unique: Number of connection for Neuron.
 */


void Neuron::start_neuron(int connections){

    weight.resize(connections);
    weight_batch.resize(connections);
    input.resize(connections);
    delta.resize(connections);
    old_delta.resize(connections);

    random_device rd; mt19937 gen(rd());
    normal_distribution <>d(0,001);

    for(int i = 0; i < (int)weight.size(); i++)
        weight[i] = d(gen);

    bias = fabs(d(gen));

    bias_backup = 0;
    bias_batch =0;

    if(isnan(bias)){
        cout<<bias<<"  "<<endl; exit(0);
    }
}

//****************************************************************


bool Neuron::get_dropout_result(){return dropout_result;}


/**
 * @brief Neuron::get_bias
 * @return
 *
 * Description: Return bias value
 */
float Neuron::get_bias(){return bias;}

/**
 * @brief Neuron::get_error
 * @return
 *
 * Description: Return the error value calculate by gradient.
 */

float Neuron::get_error(){return erro;}

/**
 * @brief Neuron::get_gradient
 * @return
 *
 * Description: Return the gradient calculate in the backpropagation fase.
 */

float Neuron::get_gradient(){return gradient;}

/**
 * @brief Neuron::get_output
 * @return
 *
 * Description: Return the output got in the foward fase
 */

float Neuron::get_output(){return output;}

/**
 * @brief Neuron::get_vj
 * @return
 *
 * Description: Return the local field got in the foward fase.
 */

float Neuron::get_vj(){return vj;}

/**
 * @brief Neuron::get_weights
 * @return
 *
 * Description: Return a std:vector<float> with all neurons connections (weights).
 */

vector<float>Neuron::get_weights(){return weight;}

/**
 * @brief Neuron::get_weight
 * @param index
 * @return
 *
 * Description: Return the Neuron object with all parameters.
 *
 */

float Neuron::get_weight(int index){return weight[index];}

//****************************************************************

/**
 * @brief Neuron::set_bias
 * @param value
 *
 * Description : Modifies the bias parameter.
 */

void Neuron::set_bias(float value){bias =value;}

/**
 * @brief Neuron::set_error
 * @param value
 *
 * Description: Modifies the error value
 */

void Neuron::set_error(float value){erro = value;}

/**
 * @brief Neuron::set_gradient
 * @param value
 *
 * Description: Modifies the gradient value.
 */

void Neuron::set_gradient(float value){gradient = value;}

/**
 * @brief Neuron::set_output
 * @param value
 *
 * Description: Modifies the output value.
 */

void Neuron::set_output(float value){output = value;}

/**
 * @brief Neuron::set_vj
 * @param value
 *
 * Description: Modifies the local field value.
 *
 *
 */


void Neuron::set_vj(float value){vj = value;}

/**
 * @brief Neuron::set_weight
 * @param index
 * @param value
 *
 *Description: Modifies the weight (connections) connected with the Neuron (index).
 *
 */

void Neuron::set_weight(int index,float value){weight[index] = value;}


void Neuron::set_dropout_result(bool value){dropout_result = value;}

//###########################################################################################################

//***************************************************************************
Neuron_RBM::Neuron_RBM(int connections){

    start_neuron(connections);

    batch.resize(connections);
    momentum.resize(connections);
    bias_momentum = 0;
    batch_bias = 0;

    random_device rd; mt19937 gen(rd());
    normal_distribution <>d(0,1);

    if(connections > 0)
        bias =  fabs(d(gen)*0.01); //float(((rand()%200 + 1)-100)/1000.0);
    else
        bias = 0;
}

Neuron_RBM::Neuron_RBM(){}
//***************************************************************************


//###########################################################################################################

/*
 * Class Instance for load a MLP saved before
 *
 */

Neuron_MLP::Neuron_MLP(){}

/**
 * @brief Neuron_MLP::Neuron_MLP
 * @param connections
 *
 * SuperClass: Neuron
 *
 * Type = Public
 *
 * Description:
 *
 * Neurons's construtor for a Multlayer Perceptron, the @param connections means the number of "weights" the neurons have.
 *
 *
 */


Neuron_MLP::Neuron_MLP(int connections){

    start_neuron(connections);
}

//###########################################################################################################

