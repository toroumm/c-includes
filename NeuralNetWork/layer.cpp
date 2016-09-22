
#include<vector>
#include "layer.h"
#include "neuron.h"
#include <iostream>

using namespace std;

/**
 * @brief Layer::Layer
 *
 * Type =  public
 *
 * Description: Layer constructor without parameters.
 */
Layer::Layer(){ }


/**
 * @brief Layer::Layer
 * @param neurons
 * @param connections
 *
 * Type = public
 *
 * Description: Constructor of the layers, parameters:
 *  First = Number of neurons in that layer.
 *  Second = Number of connections of each neuron.
 *
 */

Layer::Layer(int neurons, int connections){
    
    start_layer(neurons,connections);
}

//*****************************************************************************************
/**
 * @brief Layer::start_layer
 * @param n_neurons
 * @param connections
 *
 * Type = protected
 *
 * Description: Add to layer the objects of the Neuron class with your connections.
 *
 *
 */

void Layer::start_layer(int n_neurons, int connections){
       
    if(n_neurons ==0){
        cout<<"Do not exist neurons in this layer, ending algorithm "<<endl;
        exit(0);
    }
    for(int i = 0; i < n_neurons; i++)
        neurons.push_back(Neuron(connections));
}


//##############################################################################################################

/**
* @brief Layer_MLP::Layer_MLP
* @param n_neurons
* @param connections
*
* SuperClass: None
*
* Type = Public
*
* Description:
*
* Layer's construtor for Multilayer Perceptron, the parameters are:
* First -> Number of neurons in the layer (n_neurons).
* Second -> Numner of connections (weights) of each neurons in the layer.
*
*/

Layer_MLP::Layer_MLP(int n_neurons, int connections){

    if(n_neurons ==0){
        cout<<"Do not exist neurons in this layer, ending algorithm "<<endl;
        exit(0);
    }
    for(int i = 0; i < n_neurons; i++)
        neurons.push_back(Neuron(connections));
}
//##############################################################################################################

