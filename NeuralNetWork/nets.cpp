#include "nets.h"
#include <math.h>
#include <algorithm>
#include <iostream>

using namespace std;
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
    old_delta.resize(connections);

    random_device rd, rb; mt19937 gen(rd()),gen_bias(rb());

    normal_distribution <>d(0,0.1/*sqrt(weight.size())*/);
    normal_distribution <>b(0,0.01);

    for(int i = 0; i < (int)weight.size(); i++)
        weight[i] = d(gen);

    bias = b(gen_bias);

    bias_backup = 0;
    bias_batch =0;
    vj = 0;
    old_bias = 0;
    erro = 0;
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

/*float Neuron::get_gradient(){return gradient;}

/**
 * @brief Neuron::get_output
 * @return
 *
 * Description: Return the output got in the foward fase
 */

/*float Neuron::get_output(){return output;}

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

/*oid Neuron::set_gradient(float value){gradient = value;}

/**
 * @brief Neuron::set_output
 * @param value
 *
 * Description: Modifies the output value.
 */

/*void Neuron::set_output(float value){output = value;}

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




//##############################################################################################################

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

Layer::~Layer(){}

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

    input.resize(connections);
    gradient.resize(n_neurons);
    output.resize(n_neurons);

}


//##############################################################################################################

/**
 * @brief Nets::Nets
 *
 * Type =  public
 *
 * Description: Constructor of Nets class.
 */
Nets::Nets()
{
    set_activation_function_hidden_layer(ACT_FUNC_TYPE::SIGMOIDAL);

    set_activation_function_output_layer(ACT_FUNC_TYPE::SIGMOIDAL);

    set_loss_function(LOSS_FUNCTION::MEAN_SQUARE_ERROR);

    FUNC_TYPE = ACT_FUNC_TYPE::SIGMOIDAL;

    OUT_FUNC_TYPE = ACT_FUNC_TYPE::SIGMOIDAL;

    LOSS_FUNC_TYPE = LOSS_FUNCTION::MEAN_SQUARE_ERROR;

    set_softmax_output_layer(false);

    set_dropout_on(false);

    set_dropout_threshold(0.5);

    set_learning_decay(1);

    set_learning_rate(0.05);

    set_momentum(0.6);

    set_param_tanh(0.2,0.4);

    set_epochs(100);

    set_batch(1);

    set_regularization_lambda(0);

    set_learning_decay_batch(1);

    set_error_minimum(0.000001);

    set_weight_decay_cost(0);

    set_last_hit_train(0);

    set_last_hit_validation(0);

    set_stop_early(1);

    set_batch_show_progress(300);

    set_weight_decay_cost(0);

    set_min_loss_function_error(-1);
}

//**********************************************************************************************
Nets::~Nets(){}

//**********************************************************************************************
void Nets::set_min_loss_function_error(float value){min_loss_function_error = value;}
void Nets::set_dropout_threshold(float value){dropout_threshold = value;}
void Nets::set_dropout_on(bool value){dropout = value;}
void Nets::set_regularization_lambda(float value){lambda_l2 =value;}
void Nets::set_n_data_samples(int value){n_samples =value;}
void Nets::set_n_data_validation(int value){n_validation = value;}
void Nets::set_weight_decay_cost(float value){weight_decay_cost = value;}
void Nets::set_last_cost_train(float value){last_cost_train = value;}
void Nets::set_last_cost_validation(float value){last_cost_validation =value;}
void Nets::set_stop_early(bool value){stop_early = value;}

/**
 * @brief Nets::set_learning_decay
 * @param value
 *
 * Type =  public
 *
 * Description:
 *
 * Set the decrease of learning rate of each epoch.
 */

void Nets::set_learning_decay(float value){learning_decay = value;}

/**
 * @brief Nets::set_learning_rate
 * @param value
 *
 * Set the learning rate value.
 */

void Nets::set_learning_rate(float value){learning_rate = value;}

/**
 * @brief Nets::set_momentum
 * @param value
 *
 * Type = public
 *
 * Description: Set the momentum value.
 */

void Nets::set_momentum(float value){alpha = value;}

/**
 * @brief Nets::set_epochs
 * @param value
 *
 * Type = public
 *
 * Description = Set the number of epochs wich the algorithm will run.
 */


void Nets::set_epochs(int value){epochs = value;}

/**
 * @brief Nets::set_activation_function
 * @param FUNCTION
 *
 * Type = public
 *
 * Description:
 *
 * Set the type of activation function on the hidden layers, it is about a enum structure, the type are:
 *
 * 1º ACT_FUNC_TYPE::SIGMOIDAL
 * 2º ACT_FUNC_TYPE::HYPERBOLIC
 * 3º ACT_FUNC_TYPE::SOFTPLUS
 * 4º ACT_FUNC_TYPE::ReLU
 */


void Nets::set_activation_function_hidden_layer(ACT_FUNC_TYPE FUNCTION){

    if(FUNCTION == (ACT_FUNC_TYPE::SIGMOIDAL)){
        activation_function = &Nets::sigmoidal;
        derivate_function = &Nets::derivate_sigmoidal;

    }
    else if(FUNCTION == ACT_FUNC_TYPE::HYPERBOLIC){
        activation_function = &Nets::tangent_hyperbolic;
        derivate_function = &Nets::derivate_tanh_hyperbolic;
    }

    else if(FUNCTION == ACT_FUNC_TYPE::SOFTPLUS){

        activation_function = &Nets::softplus;
        derivate_function = &Nets::derivate_softplus;
    }
    else if(FUNCTION == ACT_FUNC_TYPE::ReLU){

        activation_function = &Nets::rectifier;
        derivate_function = &Nets::derivate_rectifier;
    }

    FUNC_TYPE = FUNCTION;

    if(activation_function == NULL || derivate_function ==NULL){cout<<"Activation Function Hidden Layer == NULL"<<endl; exit(0);}
}


void Nets::set_activation_function_output_layer(ACT_FUNC_TYPE FUNCTION){

    if(FUNCTION == (ACT_FUNC_TYPE::SIGMOIDAL)){
        activation_function_out = &Nets::sigmoidal;
        derivate_function_out = &Nets::derivate_sigmoidal;

        set_softmax_output_layer(false);

        OUT_FUNC_TYPE = ACT_FUNC_TYPE::SIGMOIDAL;

    }
    else if(FUNCTION == ACT_FUNC_TYPE::HYPERBOLIC){
        activation_function_out = &Nets::tangent_hyperbolic;
        derivate_function_out = &Nets::derivate_tanh_hyperbolic;

        set_softmax_output_layer(false);

        OUT_FUNC_TYPE =ACT_FUNC_TYPE::HYPERBOLIC;
    }

    else if(FUNCTION == ACT_FUNC_TYPE::SOFTPLUS){

        activation_function_out = &Nets::softplus;
        derivate_function_out = &Nets::derivate_softplus;

        set_softmax_output_layer(false);

        OUT_FUNC_TYPE =ACT_FUNC_TYPE::SOFTPLUS;
    }
    else if(FUNCTION == ACT_FUNC_TYPE::ReLU){

        activation_function_out = &Nets::rectifier;
        derivate_function_out = &Nets::derivate_rectifier;

        set_softmax_output_layer(false);

        OUT_FUNC_TYPE =  ACT_FUNC_TYPE::ReLU;
    }
    else if(FUNCTION == ACT_FUNC_TYPE::SOFTMAX){

        set_softmax_output_layer(true);

        OUT_FUNC_TYPE =ACT_FUNC_TYPE::SOFTMAX;

    }

    if(activation_function == NULL || derivate_function ==NULL){cout<<"Activation Function Output Layer == NULL"<<endl; exit(0);}
}

/**
 * @brief Nets::set_param_tanh
 * @param a
 * @param b
 *
 * Type = public
 *
 * Description: Set the alpha and beta parameters when we use the ACT_FUNC_TYPE::HYBERPOLIC.
 */

void Nets::set_param_tanh(float a, float b){ tanh_a = a; tanh_b =b;}

/**
 * @brief Nets::set_batch
 * @param value
 *
 * Type =  public
 *
 * Description: Set the number batch of examples which we will run before update weights (connections).
 *
 */

void Nets::set_batch(int value){if(value > 0)batch_data = value;else batch_data=1;}

/**
 * @brief Nets::set_softmax_output_layer
 * @param value
 *
 * Type =  public
 *
 * Description: Case the output layer be softmax turn on this method.
 */

void Nets::set_batch_show_progress(int value){batch_show_progress = value;}
void Nets::set_learning_decay_batch(int value){learning_decay_batch = value;}

void Nets::set_softmax_output_layer(bool value){softmax_activation = value;}

void Nets::set_error_minimum(float value){error_minimum = value;}

void Nets::set_last_hit_train(float value){last_hit_train = value;}

void Nets::set_last_hit_validation(float value){last_hit_validation =value;}

void Nets::set_n_epochs_trained(int value){n_epochs_treined = value;}

//**********************************************************************************************

/**
 * @brief Nets::get_error_train
 * @return
 *
 * Type =  public
 *
 * Description: Return error in each epoch, this error can be Mean Square Error or Cross Entropy, always over whole data set.
 * The structure of return it is std::vector.
 */

vector<float> Nets::get_error_train(){return error ;}

/**
 * @brief Nets::get_batch
 * @return
 *
 * Type = public
 *
 * Description:
 */

int Nets::get_batch(){return batch_data;}

/**
 * @brief Nets::get_epochs
 * @return
 *
 *
 *
 */

bool Nets::get_softmax_output_layer(){return softmax_activation;}

bool Nets::get_dropout_on(){return dropout;}

bool Nets::get_stop_early(){return stop_early;}

int Nets::get_n_data_samples(){return n_samples;}

int Nets::get_n_data_validation(){return n_validation;}

int Nets::get_epochs(){ return epochs;}

int Nets::get_learning_decay_batch(){return learning_decay_batch;}

int Nets::get_n_epochs_trained(){return n_epochs_treined;}

int Nets::get_batch_show_progress(){return batch_show_progress;}

float Nets::get_regularization_lambda(){return lambda_l2;}

float Nets::get_learning_decay(){return learning_decay;}

float Nets::get_learning_rate(){return learning_rate;}

float Nets::get_momentum(){return alpha;}

float Nets::get_tanh_a(){return tanh_a;}

float Nets::get_tanh_b(){return tanh_b;}

float Nets::get_dropout_threshold(){return dropout_threshold;}

float Nets::get_dropout_chance(){return (rand()%100 +1)/100.0; }

float Nets::get_error_minimum(){return error_minimum;}

float Nets::get_weight_decay_cost(){return weight_decay_cost;}

float Nets::get_last_hit_train(){return last_hit_train;}

float Nets::get_last_hit_validation(){return last_hit_validation;}

float Nets::get_last_cost_train(){return last_cost_train;}

float Nets::get_last_cost_validation(){return last_cost_validation;}

float Nets::get_min_loss_function_error(){return min_loss_function_error;}

//**********************************************************************************************************
float Nets::get_mse(vector<vector<float> > desire,vector<vector<float> > get){

    if(get.size() == 0)
        return 0;

    float a=0;
    for(int i = 0; i < (int)get.size(); i++){
        float aa = 0;
        for(int j = 0; j < (int)get[j].size();j++){
            aa+= pow(desire[i][j]- get[i][j],2)/2;
        }a += aa;
    }return a/get.size();
}
//**********************************************************************************************************
float Nets::get_cross_entropy(vector<vector<float> > desire, vector<vector<float> > get){
    float t=0;
    for(int i = 0; i < (int)desire.size(); i++){
        for(int j = 0; j < (int)desire[i].size(); j++){
            if(get[i][j] >= 1)
                get[i][j] = 0.999999;
            if(get[i][j] < 0.00000001)
                get[i][j] = 0.00000001;

            if(fabs(desire[i][j]) > 0)
                t+= -log(get[i][j]);

            // t+= (desire[i][j]*log(get[i][j])) + ((1-desire[i][j])*log(1-get[i][j]));
        }

        if(isnan((t)/desire.size()) || isnan(t) || -isnan(t) || isinf(t) || -isinf(t) || isinf((t)/desire.size())){
            cout<<"É no cross entropy"<<endl;
            return 0.000001;
        }
    }return (t)/desire.size();
}

//**********************************************************************************************************
float  Nets::get_treatment(float value){

    if(isnan(value))
        return  numeric_limits<float>::min();
    if(isinf(value))
        return numeric_limits<float>::max()-10;

    return value;
}

//**********************************************************************************************************

float Nets::get_cross_product(vector<float> a, vector<float> b){

    float value = 0;
    for(int i = 0; i < (int)a.size();i++)
        value += a[i]*b[i];
    return value;
}
//**********************************************************************************************************

void Nets::set_loss_function(LOSS_FUNCTION FUNCTION){

    if(FUNCTION == LOSS_FUNCTION::MEAN_SQUARE_ERROR){

        loss_function = &Nets::get_mse;
        
      //  loss_function_train = &Nets::get_square_error_train;

        LOSS_FUNC_TYPE = LOSS_FUNCTION::MEAN_SQUARE_ERROR;
    }
    else if(FUNCTION == LOSS_FUNCTION::CROSS_ENTROPY){

        loss_function = &Nets::get_cross_entropy;
        
      //  loss_function_train = &Nets::get_cross_entropy_train;

        LOSS_FUNC_TYPE = LOSS_FUNCTION::CROSS_ENTROPY;
    }
}
//**********************************************************************

float Nets::get_hit_rate(vector<vector<float> > desire, vector<vector<float> > predict,float trunc){

    float hit_rate = 0.0;

    if(!get_softmax_output_layer()){

        for(int i = 0; i < (int)desire.size(); i++){
            for(int j = 0; j < (int)desire[i].size();j++){

                if(predict[i][j] > trunc)
                    predict[i][j] = 1.0;
                else
                    predict[i][j] = 0.0;

                if(desire[i][j] > trunc)
                    desire[i][j] = 1.0;
                else
                    desire[i][j] = 0.0;
            }
        }

        for(int i = 0; i < (int)desire.size(); i++){
            int sum= 0;
            for(int j = 0; j < (int)desire[i].size(); j++){
                if(fabs(desire[i][j] != predict[i][j])){
                    sum = 10; break;
                }
            }
            if(sum < 1)
                hit_rate+=1.0;
        }

    }else{

        for(int i = 0; i < (int)desire.size(); i++){

            int x =0,y=0; float max = 0;
            for(int j = 0; j < (int)desire[i].size();j++){

                if(predict[i][j] > max ){
                    max = predict[i][j]; x = i; y =j;
                }
            }
            if(desire[x][y]==1 && predict[x][y] >= trunc)
                hit_rate+=1.0;
        }
    }
    return float(hit_rate /desire.size());
}


//**********************************************************************************************

float Nets::softplus(float vj){return log(1+exp(vj));}

float Nets::derivate_softplus(float vj){ return 1 / (1 + exp(-vj));}

float Nets::rectifier(float vj){if(vj <= 0)return 0; return vj;}

float Nets::derivate_rectifier(float vj){if(vj <= 0)return 0; return 1;}

float Nets::sigmoidal(float vj){  return 1 / (1+(exp(-vj))); }

float Nets::derivate_sigmoidal(float vj){return  (vj * (1-vj));}

float Nets::tangent_hyperbolic(float vj){ return (get_tanh_a()*tanh(get_tanh_b()*vj));}

float Nets::derivate_tanh_hyperbolic(float vj){return ((get_tanh_a()/get_tanh_b())*(get_tanh_a() - vj)*(get_tanh_a() + vj));}


//**********************************************************************************************************
float Nets::softmax_sum(Layer *layer){
    float max = -1000000000, loge =0;

    for(int i = 0; i < (int)layer->output.size(); i++)
        if(max < (layer->neurons[i].vj + layer->neurons[i].bias))
            max =  layer->neurons[i].vj + layer->neurons[i].bias;

    for(int i = 0; i < (int)layer->output.size(); i++)
        loge+=  exp((layer->neurons[i].vj + layer->neurons[i].bias)-max);

    return max + log(loge);
}
//**********************************************************************************************************
void Nets::softmax(Layer *layer){

    float max = softmax_sum(layer);

    for(int i = 0; i < (int)layer->output.size(); i++)
        layer->output[i]  = get_treatment(exp((layer->neurons[i].vj + layer->neurons[i].bias)-max));
    //neurons[i].output = exp((neurons[i].vj + neurons[i].bias)-max);
}
//**********************************************************************************************************
float Nets::derivate_softmax(float desire, float get){

    if(desire == 1)
        return get -1;
    else
        return get;
}

//**********************************************************************************************************

float Nets::get_cross_entropy_train(float d, float g){
    
    if(d <= 1 && d >= 0 && g <= 1 && g >= 0 )
        return 0;
     return 1;           
}
