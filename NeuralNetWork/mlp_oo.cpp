#include "mlp_oo.h"
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <random>
#include <iostream>
#include <fstream>
#include <limits>
#include "../Utilitarios/utilitarios.h"
/*
 * Class Instance for load a MLP saved before
 *
 */

/*Neuron_MLP::Neuron_MLP(){}

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


/*Neuron_MLP::Neuron_MLP(int connections){

    start_neuron(connections);
}*/
//**********************************************************************************************************

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

/*Layer_MLP::Layer_MLP(int n_neurons, int connections){

    if(n_neurons ==0){
        cout<<"Do not exist neurons in this layer, ending algorithm "<<endl;
        exit(0);
    }
    for(int i = 0; i < n_neurons; i++)
        neurons.push_back(Neuron(connections));
}*/
//**********************************************************************************************************
/**
 * @brief MLP::MLP
 * @param input
 * @param output
 * @param hidden
 *
 * SuperClass: Nets
 *
 * Type = Public
 *
 * Description:
 *
 * Multilayer Percpetron's construtor with three parameters:
 *
 * First = Number of neurons in the input layer, that is, number of attributes
 *
 * Second = Number of neurons in the output layer
 *
 * Third = std::vector<int> structure, where the size of vector is a number of  hidden layers
 *  and in each vector's position we put the number of neurons in that layer.
 *
 *  Example:
 *
 *      std::vector<int>layers(2); //Two hidden layers
 *
 *      layers[0] = 20; // twenty neurons in the first hidden layer
 *      layers[1] = 30; // thirty neurons in the second hidden layer
 *
 *
 */
MLP::MLP(int input, int output, vector<int> hidden){
    start_mlp(input,output,hidden);
}





//**********************************************************************************************************
/**
 * @brief MLP::MLP
 *
 * Superclass: Nets
 *
 * Type = Public
 *
 * Description
 *
 * Another Multilayer constructor without parameters
 */

MLP::MLP(){

    set_learning_decay(1);

    set_learning_rate(0.05);

    set_momentum(0.6);

    set_param_tanh(0.2,0.4);

    set_epochs(100);

    set_batch(1);

    set_activation_function_hidden_layer(ACT_FUNC_TYPE::SIGMOIDAL);

    set_activation_function_output_layer(ACT_FUNC_TYPE::SIGMOIDAL);

    set_loss_function(LOSS_FUNCTION::MEAN_SQUARE_ERROR);
}


//**********************************************************************************************************
/**
 * @brief MLP::start_mlp
 * @param input
 * @param output
 * @param hidden
 *
 * Type : Private
 *
 * Description:
 *
 * Start the Multilayer parameters
 *
 */

void MLP::start_mlp(int input, int output, vector<int> hidden){

    if(input <=0){

        cout<<"There Isn't Input neurons, ending process"<<endl;
        exit(0);
    }

    if(output <=0){

        cout<<"There Isn't Output neurons, ending process"<<endl;
        exit(0);
    }

    if(hidden.size() ==0){

        cout<<"There Isn't Hidden layers, ending process"<<endl;
        exit(0);
    }

    layers.push_back(Layer(hidden[0],input));
    for(int i = 1; i < (int)hidden.size(); i++)
        layers.push_back(Layer(hidden[i],hidden[i-1]));

    layers.push_back(Layer(output,hidden[hidden.size()-1]));
}


//**********************************************************************************************************
/**
 * @brief MLP::run_MLP
 * @param samples
 * @param out
 * @param _epochs
 * @param percent_validadion
 *
 * Type = Public
 *
 * Description:
 *
 *The parameters are:
 *
 * First = Examples for training the MLP, is a std::vector of std::vector structure (2D Matrix), for create there is a very simple way.
 *
 *      Example:
 *
 *          using namespace std;
 *
 *          vector<vector<float> >input_data; //You can use de Utility class for load a txt file, see more about that, in this documentation.
 *
 * Second  = The output means the result desirable, given the input data, use the same structure of input_data example(vector<vector<float> >out).
 *
 * Third = It's a number of epochs with yout network will train, is a int number.
 *
 * Fourth = The percentual of input data which you want use for a overfitting avaliation.
 *
 */

void MLP::run_MLP(vector<vector<float> > samples, vector<vector<float> > out,int _epochs, float percent_validadion){

    vector<vector<float> >  validation,validation_out;
    vector<float>error_validation_set;

    float val_error_tolerance=0, val_cost =0, train_error_zigzag=0, train_error_tolerance_zz=0,val_error_tolerance_zz =0 ;

    Utilitarios util;

    util.on_cross_validation(samples,out,validation,validation_out,percent_validadion);

    set_n_data_samples(samples.size());
    set_n_data_validation(validation.size());
    set_epochs(_epochs);
    set_last_cost_validation(0);
    set_last_hit_validation(0);


    util.shuffle_io_samples(samples,out);

    try {


        for(int i = 0; i < get_epochs() ; i++){

            forward(samples,out,1);

            if(i%get_learning_decay_batch() == 0)
                   if(get_learning_rate() > 0.0001)
                    set_learning_rate(get_learning_rate()*get_learning_decay());

            vector<vector<float> >aux_samples, aux_validation;

            predict(samples,aux_samples);

            set_last_hit_train(get_hit_rate(out,aux_samples,0.5));
            set_last_cost_train((this->*loss_function)(out,aux_samples));
            set_n_epochs_trained(i);

            error.push_back(get_last_cost_train()+get_weight_decay_cost());

            if(validation.size()>0 && (i > 1 && !(i%20) || !(i%get_batch()))){

                predict(validation,aux_validation);

                set_last_hit_validation(get_hit_rate(validation_out,aux_validation,0.5));

                set_last_cost_validation((this->*loss_function)(validation_out,aux_validation));


                if(val_cost != 0 && val_cost < get_last_cost_validation() && abs(train_error_zigzag) > abs(error[error.size()-1]) )
                    val_error_tolerance_zz +=1;
                else if(val_error_tolerance_zz > 0.3)
                    val_error_tolerance_zz -=0.3;

                val_cost = get_last_cost_validation();

                if(i >get_batch()){
                    if(get_last_hit_validation() < error_validation_set[error_validation_set.size()-1]){
                        val_error_tolerance += 1.0;
                    }
                    else{
                        if(val_error_tolerance >= 1 && get_last_hit_validation() > error_validation_set[error_validation_set.size()-1]){
                            val_error_tolerance -= 1;
                        }
                    }
                }

                error_validation_set.push_back(get_last_hit_validation());

                if(((val_error_tolerance >= 2) /*&& val_error_tolerance_zz >= 3) */||val_error_tolerance_zz > 5) && get_stop_early()){
                    cout<<"Stop Train Overfitting "<<endl;
                    break;
                }
            }

            if(!(i%get_batch_show_progress()))
                cout<<"Epochs "<<i<<" Cost: Train "<<error[error.size()-1]<<"  Val "<<get_last_cost_validation()<<" Hits: Train  "<<get_last_hit_train()<<" Val "<<get_last_hit_validation()<<" Learning "<<get_learning_rate()
                   <<" Paciente: Train "<<train_error_tolerance_zz<<" Val "<<val_error_tolerance_zz<<" Noise Cost "<<get_weight_decay_cost()<<endl;

            if(abs(train_error_zigzag) < abs(error[error.size()-1])){
                train_error_tolerance_zz +=1;
            }
            else{
                if(train_error_tolerance_zz > 0.3){
                    train_error_tolerance_zz -=0.3;
                }
            }
            if((get_min_loss_function_error() != -1 && get_min_loss_function_error() >= get_last_cost_train()+get_weight_decay_cost()))
                break;
           // if(train_error_tolerance_zz >= train_limit_zz && get_learning_decay() < 1){set_learning_rate(get_learning_rate()*0.9); train_limit_zz += train_limit_zz;}

           // if(error[error.size()-1] < get_error_minimum() ||  get_stop_early() && train_error_tolerance_zz >=20)
             //   break;

            train_error_zigzag = error[error.size()-1];

            if(get_last_hit_train() > 0.995)
                break;

            set_weight_decay_cost(0);
        }
    } catch(exception &e){

        cout<<"MLP problem in run "<<e.what()<<endl;
        return;
    }
}
//**********************************************************************************************************
/**
 * @brief MLP::forward
 * @param samples
 * @param out
 * @param tag
 *
 * Type = private
 *
 * Description: This method is responsable to accomplish the foward step in a Multilayer Perceptron
 */

void MLP::forward(vector<vector<float> > samples, vector<vector<float> >out, int tag){
    
    error.push_back(0);
    for(int ii = 0; ii < (int)samples.size(); ii++){

        for(int j = 0; j < (int)layers[0].neurons.size();j++){
            layers[0].input = samples[ii];
        }

        try{
            local_field_induced(layers);
        }catch(exception &e){
            cout<<"MLP Local field Induced Problem  "<<e.what()<<endl;

        }
        if(tag == 1){

            try {

                gradient_last_layer(&layers[layers.size()-1],out[ii]);

                gradient_hidden_layers(layers);
            } catch (exception &e) {

                cout<<"MLP Gradient Problem "<<e.what()<<endl;
            }

            try {

                if(ii % get_batch() == 0 || ii == (int)samples.size()-1){
                    update_weights(layers,true );
                }
                else{
                    update_weights(layers,false);

                }
            } catch (exception &e) {
                cout<<"MLP Update Weights Problem  "<<e.what()<<endl;
            }
        }
    }
}
//**********************************************************************************************************
/**
 * @brief MLP::local_field_induced
 * @param layer
 *
 * Type = private
 *
 * Description:
 *
 * Make a forward pass called local field induced in each net's layer.
 *
 */

void MLP::local_field_induced(vector<Layer> &layer){

#pragma omp paralell
    {
#pragma omp for
        for(int i = 0; i < (int)layer.size(); i++){

            if(i == layer.size()-1){

                for(int j = 0; j < (int)layer[i].neurons.size(); j++){

                    layer[i].neurons[j].vj = ((get_cross_product(layer[i].neurons[j].weight,layer[i].input)));


                    if(!get_softmax_output_layer())
                        layer[i].output[j] = ((this->*activation_function_out)(layer[i].neurons[j].vj + layer[i].neurons[j].bias));

                    layer[i].neurons[j].set_dropout_result(1);
                }
                if(get_softmax_output_layer())softmax(&layer[i]);

            }else{

                for(int j = 0; j < (int)layer[i].neurons.size(); j++){

                    layer[i].neurons[j].vj = 0;

                    if(!get_dropout_on()  || (get_dropout_on() && get_dropout_chance() > get_dropout_threshold())){

                        layer[i].neurons[j].set_dropout_result(1);

                        layer[i].neurons[j].vj  = (get_cross_product(layer[i].neurons[j].weight, layer[i].input));

                        layer[i].output[j] =  (this->*activation_function)(layer[i].neurons[j].vj + layer[i].neurons[j].bias);

                    }else
                        layer[i].neurons[j].set_dropout_result(0);
                }
            }
            if(!(i == layer.size()-1))
                layer[i+1].input = layer[i].output;
        }
    }
}

//**********************************************************************************************************
/**
 * @brief MLP::gradient_last_layer
 * @param layer
 * @param out
 *
 * Type =  private
 *
 * Description:
 *
 * It's the first step of Backpropagation Algorithm, calculing the gradient of error in the output layer.
 */

void MLP::gradient_last_layer(Layer *layer, vector<float> out){

    for(int i = 0; i < (int)layer->neurons.size();i++){

        if(LOSS_FUNC_TYPE == MEAN_SQUARE_ERROR  && !get_softmax_output_layer()){
            layer->neurons[i].erro =(out[i] - layer->output[i]);
            layer->gradient[i]  = ((layer->neurons[i].erro * (this->*derivate_function_out)(layer->output[i])));

        }
        else if(LOSS_FUNC_TYPE == CROSS_ENTROPY && !get_softmax_output_layer()){
            layer->neurons[i].erro = 0;
            for(int j = 0; j < (int)layer->input.size(); j++)
                layer->neurons[i].erro += layer->input[j] * (out[i] - layer->output[i]);
            if(layer->input.size() > 0)
                layer->neurons[i].erro /= layer->input.size();

            layer->gradient[i]  = ((layer->neurons[i].erro * (this->*derivate_function_out)(layer->output[i])));
        }

        else if(get_softmax_output_layer()){
            if(get_batch() > 0)
                layer->gradient[i] = ((derivate_softmax(out[i],layer->output[i])/get_batch()));
        }
    }
}
//**********************************************************************************************************
/**
 * @brief MLP::gradient_hidden_layers
 * @param layer
 *
 * Type : private
 *
 * Description:
 *
 * Get the gradient in all hidden layers (backward).
 *
 */

void MLP::gradient_hidden_layers(vector<Layer> &layer){

    for(int i = (int)layer.size()-2; i >= 0; i--){

        for(int j = 0; j < (int)layer[i].neurons.size(); j++){
            float s = 0;
            if(layer[i].neurons[j].get_dropout_result()){

                if(fabs((this->*derivate_function)(layer[i].output[j])) > 0){

                    for(int k = 0; k < (int)layer[i+1].neurons.size();k++)
                        if(layer[i+1].neurons[k].get_dropout_result())
                            s += layer[i+1].neurons[k].weight[j] * layer[i+1].gradient[k];

                    layer[i].gradient[j] = (s * (this->*derivate_function)(layer[i].output[j]));
                }
                else
                    layer[i].gradient[j]  = 0;
            }
        }
    }
}
//**********************************************************************************************************

/**
 * @brief MLP::update_weights
 * @param layer
 *
 * Type = private
 *
 * Description
 *
 * After calc the gradient for each neuron of each layer is time to update weights, so each connection (weights) is updated.
 */

void MLP::update_weights(vector<Layer> &layer, bool batch_on){

    float direction =1;
    if(FUNC_TYPE == 4)
        direction = -1;

    for(int i = 0; i < (int)layer.size(); i++){
#pragma omp paralell
        {
#pragma omp for
            for(int j = 0; j < (int)layer[i].neurons.size(); j++){

                if(layer[i].neurons[j].get_dropout_result()){

                    for(int k = 0; k < (int)layer[i].neurons[j].weight.size(); k++){

                        if(isnan(layer[i].gradient[j]) || isnan(layer[i].input[k])){
                            cout<<"Update weights gradient or input is NAN"<<endl;
                        }

                        float delta =  ( layer[i].gradient[j] * layer[i].input[k] );

                        layer[i].neurons[j].weight_batch[k] += (delta + (layer[i].neurons[j].old_delta[k] * get_momentum()));

                        if(batch_on){

                            float decay  = 0;//get_regularization_lambda() *   layer[i].neurons[j].weight[k];

                            if(abs(decay) > 1)
                                decay = 0;

                            layer[i].neurons[j].weight[k] +=  direction*get_learning_rate()*((layer[i].neurons[j].weight_batch[k])+decay);

                            if(isnan(layer[i].neurons[j].weight[k])){
                                cout<<"I have a NaN update weights"<<endl;
                            }
                            if(layer[i].neurons[j].weight.size() > 0)
                                set_weight_decay_cost(get_weight_decay_cost()+((get_regularization_lambda() *  layer[i].neurons[j].weight[k]* layer[i].neurons[j].weight[k]*0.5)/layer[i].neurons[j].weight.size()));

                            layer[i].neurons[j].weight_batch[k] = 0;
                        }
                        layer[i].neurons[j].old_delta[k] = delta;
                    }

                    layer[i].neurons[j].bias_batch += get_learning_rate() * layer[i].gradient[j];

                    if(batch_on){

                        layer[i].neurons[j].bias_batch  = (layer[i].neurons[j].bias_batch);

                        layer[i].neurons[j].bias += (direction)*(layer[i].neurons[j].bias_batch);

                        layer[i].neurons[j].bias_batch = 0;
                    }
                }
            }
        }
    }
}
//**********************************************************************************************************

/**
 * @brief MLP::save_net
 * @param net
 * @param path
 *
 * Type =  public
 *
 * Description :
 *
 * Save your net after the training. You can save and after keep training (there is a method here for this), or use to classification new data, as you wish.
 *
 *Parameters:
 *     First = The net itself.
 *
 *     Second =  The path where the method will save the txt file.
 */

void MLP::save_net(MLP *net, string path){

    ofstream file;
    file.open(path);
    if(!file.is_open()){
        cout<<"Erro ao abrir arquivo: "<<path<<endl;
        return;
    }

    file<<net->get_learning_rate()<<" ";
    file<<net->get_learning_decay()<<" ";
    file<<net->get_momentum()<<endl;

    //save layer
    file<<net->layers.size()<<endl;
    for(int i =0; i < (int)net->layers.size(); i++){
        file<<net->layers[i].neurons.size()<<" ";

        file<<net->layers[i].neurons[0].weight.size()<<" ";
    }
    for(int i =0; i < (int)net->layers.size(); i++){
        for(int j = 0; j < (int)net->layers[i].neurons.size(); j++){
            for(int k = 0; k <(int)net->layers[i].neurons[j].weight.size(); k++){
                file<<net->layers[i].neurons[j].weight[k]<<" ";

            }
            file<<net->layers[i].neurons[j].bias<<endl;
        }
    }

    file<<net->error.size()<<endl;
    for(int i = 0; i < (int)net->error.size(); i++)
        file<<error[i]<<" ";
    file<<endl;

    file.close();
}

//*************************************************************************************************
/**
 * @brief MLP::load_net
 * @param net
 * @param path
 *
 * Type =  static public
 *
 * Description:
 *
 * This method load some net which you save before.
 *
 * Parameters:
 *
 *  First = One instance of net, for example: MLP *net; //not use "new" operator because is is about static method.
 *
 *  Second =  String data with a path where you wish to save the network.
 *
 * Code snippet :
 *
 * MLP *mlp;
 *
 * MLP::load_net(mlp,"/home/user/Desktop/net_01.txt");
 *
 */

void MLP::load_net(MLP *net, string path){




    ifstream file;

    file.open(path);
    if(!file.is_open()){
        cout<<"Erro ao abrir arquivo: "<<path<<endl;
        return;
    }
    float x;
    file>>x;net->set_learning_rate(x);
    file>>x; net->set_learning_decay(x);
    file>>x; net->set_momentum(x);

    int size_net;
    file>>size_net;
    for(int i =0; i < size_net; i++){

        int n_neurons, connections;
        file>>n_neurons;
        file>>connections;
        net->layers.push_back(Layer(n_neurons,connections));
    }

    for(int i =0; i < (int)net->layers.size(); i++){
        for(int j = 0; j < (int)net->layers[i].neurons.size(); j++){
            for(int k = 0; k <(int)net->layers[i].neurons[j].weight.size(); k++){
                file>>net->layers[i].neurons[j].weight[k];

            }
            file>>net->layers[i].neurons[j].bias;
            net->layers[i].gradient[j] =0 ;
            net->layers[i].output[j] =0;
            //net->layers[i].neurons[j].gradient = 0;
            //net->layers[i].neurons[j].output = 0;
        }
    }
    int size_mse;
    file>>size_mse;

    net->error.resize(size_mse);
    for(int i = 0; i < (int)net->error.size(); i++){
        file>>net->error[i];

    }

    file.close();
}
//*********************************************************************************************************
/**
 * @brief MLP::test
 *
 * Type = public
 *
 * Description:
 *
 * This a test based in the book called: Data Mining: Concept and Techniques 3rd Edition, Autor: Jiawei Han, Micheline Kamber and Jian Pei.
 *
 * In this book's chapter 9 (9.2 page 405 to 406) is possible to follow some steps to sure if your MLP is correct. This test it was maintained because in each modification, we can see your correctness.
 *
 */

void MLP::test(){

    MLP *m = new MLP();

    m->set_learning_rate(0.9);

    m->set_momentum(0);

    m->layers.push_back(Layer(2,3));
    m->layers.push_back(Layer(1,2));

    m->layers[0].neurons.resize(2);

    m->layers[0].neurons[0].weight.resize(3);
    m->layers[0].neurons[1].weight.resize(3);

    m->layers[0].neurons[0].weight[0] = 0.2;
    m->layers[0].neurons[0].weight[1] = 0.4;
    m->layers[0].neurons[0].weight[2] = -0.5;

    m->layers[0].neurons[0].bias = -0.4;

    m->layers[0].neurons[1].weight[0] = -0.3;
    m->layers[0].neurons[1].weight[1] = 0.1;
    m->layers[0].neurons[1].weight[2] = 0.2;

    m->layers[0].neurons[1].bias = 0.2;

    m->layers[1].neurons.resize(1);

    m->layers[1].neurons[0].weight.resize(2);

    m->layers[1].neurons[0].weight[0] = -0.3;
    m->layers[1].neurons[0].weight[1] = -0.2;

    m->layers[1].neurons[0].bias = 0.1;

    vector<vector<float> >inputs;
    inputs.resize(1);
    inputs[0].resize(3);
    inputs[0][0] = 1;
    inputs[0][1] = 0;
    inputs[0][2] = 1;

    vector<vector<float> >outputs;
    outputs.resize(1);
    outputs[0].resize(1);
    outputs[0][0] = 1;

    m->forward(inputs, outputs, 1);

    m->print_net_data(m);
}
//**********************************************************************
/**
 * @brief MLP::print_net_data
 * @param net
 *
 * Type = public
 *
 * Description:
 *
 * Show the value of each connection (weight) of each neuron, in each layer, including the bias.
 *
 */

void MLP::print_net_data(MLP *net){

    cout<<"Camadas "<<net->layers.size()<<endl;

    // for(int i  = 0; i < (int)net->layers.size(); i++)
    //     cout<<"Camadas "<<i<<" Neuronios "<<net->layers[i].neurons.size()<<"  Conexoes  "<<net->layers[i].neurons[0].weight.size()<<endl;
   // cout<<"Camadas "<<0<<" Neuronios "<<net->layers[0].neurons.size()<<"  Conexoes  "<<net->layers[0].neurons[0].weight.size()<<endl;
   // cout<<"Camadas "<<1<<" Neuronios "<<net->layers[1].neurons.size()<<"  Conexoes  "<<net->layers[1].neurons[0].weight.size()<<endl;

    for(int i =0; i < (int)net->layers.size(); i++){

        for(int j = 0; j < (int)net->layers[i].neurons.size(); j++){
            for(int k = 0; k < (int)net->layers[i].neurons[j].weight.size(); k++){

                cout<<"Camada "<<i<<" Neuronio "<<j<<" Weight "<<net->layers[i].neurons[j].weight[k]<<" "<<k<<endl;

            }
            cout<<"Camada "<<i<<" Neuronio "<<j<<" bias "<<net->layers[i].neurons[j].bias<<endl;
            cout<<"Camada "<<i<<" Neuronio "<<j<<" Gradient "<<net->layers[i].gradient[j]<<endl;
            cout<<"Camada "<<i<<" Neuronio "<<j<<" out "<<net->layers[i].output[j]<<endl<<endl;

            //cout<<"Camada "<<i<<" Neuronio "<<j<<" Gradient "<<net->layers[i].neurons[j].gradient<<endl;
            // cout<<"Camada "<<i<<" Neuronio "<<j<<" out "<<net->layers[i].neurons[j].output<<endl<<endl;
        }
    }
}
//**********************************************************************
/**
 * @brief MLP::keep_training
 * @param _samples
 * @param out
 * @param epochs
 * @param percent_validation
 *
 * Type = public
 *
 * Description:
 *
 * After you train your net, if you want training with new data, or keep training with the same data, this method make this task for you.
 *
 *Note: This method was mentioned in the description of the save_net.
 *
 *
 * Parameters:
 *
 *  First = Training data -  vector<vector<float> >input_data; //Remember you can use Utilitary class for load txt files.
 *
 *  Second = Output desire, give the input data - vector<vector<float> >out;
 *
 *  Third = Number of epochs for train.
 *
 *  Foruth = Percetual to validation set to avoid overfitting.
 */


void MLP::keep_training(vector<vector<float> > _samples, vector<vector<float> > out, int epochs,float percent_validation){

    run_MLP(_samples,out,epochs,percent_validation);
}
//**********************************************************************

/**
 * @brief MLP::predict
 * @param samples
 * @param out
 *
 * Type = public
 *
 *
 * Description:
 *
 *  This method make a predict data, that is, given a input it make the output which define a class or cluster, or a continuous value.
 *
 *  Note: After the net was trained of course.
 *
 *
 */


void MLP::predict(vector<vector<float> > samples, vector<vector<float> > &out){

    out.resize(samples.size());
    for(int ii = 0; ii < (int)samples.size(); ii++){
        /* layers[0].neurons[j].input*/
        // for(int j = 0; j < (int)layers[0].neurons.size();j++)
        layers[0].input = samples[ii];
        local_field_induced(layers);
        out[ii].resize(layers[layers.size()-1].neurons.size());

        out[ii] = layers[layers.size()-1].output;
        //  for(int j = 0 ; j < (int)layers[layers.size()-1].neurons.size();j++)
        // out[ii][j] = (layers[layers.size()-1].output[j]); //(layers[layers.size()-1].neurons[j].output);
    }
}
//**********************************************************************

void MLP::gradient_level(MLP net){

    vector<float>layers(net.layers.size()), mean_output(net.layers.size());

    for(int i = 0; i < (int)net.layers.size(); i++){

        cout<<" Saida ";
        for(int j = 0; j < (int)net.layers[i].neurons.size(); j++){
            layers[i] += fabs(net.layers[i].gradient[j]/*net.layers[i].neurons[j].gradient*/);
            mean_output[i] += fabs(net.layers[i].output[j]/*net.layers[i].neurons[j].output*/);

        }

        layers[i] /= net.layers[i].neurons.size(); mean_output[i] /= net.layers[i].neurons.size();

        cout<<"Average Gradient Layer  "<<i<<" Magnitude "<<layers[i]<<" Average Out  "<<mean_output[i]<< "  "<<endl;
    }
}
//*********************************************************************************************************

void MLP::weights_changes(MLP net){

    vector<float>layers(net.layers.size()),bias(net.layers.size());//, mean_output(net.layers.size());

    for(int i = 0; i < net.layers.size(); i++){

        //cout<<" Saida ";
        for(int j = 0; j < net.layers[i].neurons.size(); j++){
            for(int k = 0; k < (int)net.layers[i].neurons[j].weight.size(); k++)
                layers[i] += fabs(net.layers[i].neurons[j].weight[k]);
            layers[i] /= net.layers[i].neurons[j].weight.size();
            bias[i] += net.layers[i].neurons[j].bias;
        }bias[i] /= net.layers[i].neurons.size();
        layers[i] /= net.layers[i].neurons.size();// mean_output[i] /= net.layers[i].neurons.size();

        cout<<"Average weight  "<<i<<" Value "<<layers[i]<<" bias "<<bias[i]<<endl;//" Average Out  "<<mean_output[i]<< "  "<<net.layers[i].neurons.size()<<endl;
    }
}
//*********************************************************************************************************

void MLP::relu_test(){

    vector<vector<float>>input, output, pesos1,pesos2,probs,vj,gradient_w2,gradient_w1, upd_w1, upd_w2,aux_out;
    vector<float>bias1, bias2, upd_bias, upd_bias2;

    Utilitarios util;

    string path = "/home/jeferson/Dropbox/Public/Projects/include/teste_relu/";

    util.loadTXTFile(input, path+"input.txt");
    util.loadTXTFile(output, path+"output.txt");
    util.loadTXTFile(pesos1, path+"relu_w1.txt");
    util.loadTXTFile(pesos2,path+"relu_w2.txt");
    util.loadTXTFile(probs, path+"relu_probs.txt");
    util.loadTXTFile(vj, path+"out_hide_one.txt");
    util.loadTXTFile(bias1,path+"relu_b.txt");
    util.loadTXTFile(bias2, path+"relu_b2.txt");
    util.loadTXTFile(gradient_w2,path+ "relu_gradient_w2.txt");
    util.loadTXTFile(gradient_w1,path+ "relu_gradient_w1.txt");
    util.loadTXTFile(upd_w1, path+"relu_upd_w1.txt");
    util.loadTXTFile(upd_w2, path+"relu_upd_w2.txt");
    util.loadTXTFile(upd_bias, path+"relu_upd_b.txt");
    util.loadTXTFile(upd_bias2, path+"relu_upd_b2.txt");

    //  util.shuffle_io_samples(input,output);

    // util.normalize(input);

    vector<int>hide(1); hide[0] =100;

    MLP mlp(input[0].size(), output[0].size(),hide );

    mlp.set_learning_decay(1);

    mlp.set_learning_rate(1);

    mlp.set_regularization_lambda(1e-3);

    mlp.set_batch(300);

    mlp.set_momentum(0);

    mlp.set_epochs(4000);

    mlp.set_activation_function_hidden_layer(mlp.ACT_FUNC_TYPE::ReLU);

    mlp.set_activation_function_output_layer(mlp.ACT_FUNC_TYPE::SOFTMAX);

    mlp.set_loss_function(mlp.LOSS_FUNCTION::CROSS_ENTROPY);

    mlp.set_stop_early(false);

    for(int i = 0; i < (int)mlp.layers[0].neurons.size();i++){

        mlp.layers[0].neurons[i].weight = pesos1[i];
        mlp.layers[0].neurons[i].bias = bias1[i];
    }

    for(int i = 0; i < (int)mlp.layers[1].neurons.size();i++){

        mlp.layers[1].neurons[i].weight = pesos2[i];
        mlp.layers[1].neurons[i].bias = bias2[i];
    }


    // mlp.run_MLP(input,output,8000,0);

    for(int k  =0; k < 4000; k++){

        for(int ii = 0; ii < (int)input.size(); ii++){

            mlp.layers[0].input = input[ii];

            mlp.local_field_induced(mlp.layers);

            mlp.gradient_last_layer(&mlp.layers[1],output[ii]);

            mlp.gradient_hidden_layers(mlp.layers);

            /* float diff1= 0, diff2=0,diff3=0,diff4=0;

            for(int i  = 0; i < (int)mlp.layers[0].output.size(); i++){
                diff1 += fabs(mlp.layers[0].output[i]-vj[ii][i]);
            }

            for(int i  = 0; i < (int)mlp.layers[1].output.size(); i++){
                diff2 += fabs(mlp.layers[1].output[i]-probs[ii][i]);
            }



            for(int i  = 0; i < (int)mlp.layers[1].gradient.size(); i++){
                diff3 += fabs(mlp.layers[1].gradient[i]-gradient_w2[ii][i]);
            }

            for(int i  = 0; i < (int)mlp.layers[0].gradient.size(); i++){
                diff4 += fabs(mlp.layers[0].gradient[i]-gradient_w1[ii][i] );
            }

            float diffTotal = diff1+diff2+diff3+diff4;
            if(diffTotal > 1e-4)
                cout<<"Epoca  "<<ii<<"  DIFF "<<diffTotal<<endl;*/

            if(ii == (int)input.size()-1)
                mlp.update_weights(mlp.layers,true);
            else
                mlp.update_weights(mlp.layers,false);

            //set_last_hit_train(get_hit_rate(output,aux_out,0.5));
            //set_last_cost_train((this->*loss_function)(output,aux_out));

            mlp.predict(input,aux_out);



        }
        /*  float diff_w1 = 0, diff_b = 0;
        for(int i = 0; i < (int)mlp.layers[0].neurons.size(); i++){
            for(int j = 0; j < (int)mlp.layers[0].neurons[i].weight.size(); j++){
                diff_w1 += fabs(mlp.layers[0].neurons[i].weight[j] - upd_w1[i][j]);
                // cout<<diff_w1<<"  "<<mlp.layers[0].neurons[i].weight[j]<<"  "<<upd_w1[i][j]<<endl;

            }
            diff_b += fabs(mlp.layers[0].neurons[i].bias-upd_bias[i]);
            //    cout<<mlp.layers[0].neurons[i].bias<<"  "<<upd_bias[i]<<endl;
        }
           cout<<"diff_w1  "<<diff_w1<<" diff_b "<<diff_b<<endl;

        float diff_w2 = 0, diff_b2 = 0;
        for(int i = 0; i < (int)mlp.layers[1].neurons.size(); i++){
            for(int j = 0; j < (int)mlp.layers[1].neurons[i].weight.size(); j++){
                diff_w2 += fabs(mlp.layers[1].neurons[i].weight[j] - upd_w2[i][j]);
                //        cout<<diff_w2<<"  "<<mlp.layers[1].neurons[i].weight[j]<<"  "<<upd_w2[i][j]<<endl;

            }
            diff_b2 += fabs(mlp.layers[1].neurons[i].bias-upd_bias2[i]);
            //      cout<<mlp.layers[0].neurons[i].bias<<"  "<<upd_bias2[i]<<endl;
        }
            cout<<"diff_w2  "<<diff_w2<<" diff_b2 "<<diff_b2<<endl;

        //   exit(0);*/

        if(!(k  % 50))
            cout<<k<<"  "<<mlp.get_hit_rate(output,aux_out,0.5)<<"  "<<mlp.get_cross_entropy(output,aux_out)<<endl;

    }



    cout<<"final"<<endl;
}

