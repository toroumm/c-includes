    #ifndef NETS_H
#define NETS_H

#include <vector>

using namespace std;

//**********************************************************************************************

class Neuron{

public:

    Neuron();
    Neuron(int connections);
    ~Neuron();

    void start_neuron(int connections);

    bool get_dropout_result();

    float get_gradient();
    float get_output();
    float get_vj();
    float get_error();
    float get_bias();
    float get_weight(int index);


    vector<float>get_weights();

    void set_bias(float value);
    void set_error(float value);
    void set_gradient(float value);
    void set_output(float value);
    void set_vj(float value);
    void set_weight(int index,float value);
    void set_dropout_result(bool value);

  vector<float>weight,old_delta,weight_batch;

  bool  dropout_result;

  float bias,bias_batch,erro, old_bias,vj, bias_backup;

private:

};

//**********************************************************************************************


class Layer{

public:
    Layer();
    Layer(int neurons, int connections);
    ~Layer();
    
    Neuron get_one_neuron(int index);
    vector<Neuron> get_neurons();
    
    void start_layer(int n_neurons, int connections);
        
    vector<Neuron>neurons;

    vector<float>input;
    vector<float>output;
    vector<float>gradient;

    
protected:



private:    


};


//**********************************************************************************************

class Nets
{
public:
    Nets();
    ~Nets();

    enum ACT_FUNC_TYPE{ SIGMOIDAL =1, HYPERBOLIC =0, SOFTPLUS = 2, SOFTMAX =3, ReLU = 4};
    enum LOSS_FUNCTION{MEAN_SQUARE_ERROR =1, CROSS_ENTROPY =2 };


    ACT_FUNC_TYPE FUNC_TYPE;
    ACT_FUNC_TYPE OUT_FUNC_TYPE;
    LOSS_FUNCTION LOSS_FUNC_TYPE;

    void set_activation_function_output_layer(ACT_FUNC_TYPE FUNCTION);
    void set_activation_function_hidden_layer(ACT_FUNC_TYPE FUNCTION);
    void set_loss_function(LOSS_FUNCTION FUNCTION);

    void set_batch(int value);
    void set_learning_rate(float value);
    void set_learning_decay(float value);
    void set_momentum(float value);
    void set_epochs(int value);
    void set_param_tanh(float a = 0.1, float b = 0.2);
    void set_softmax_output_layer(bool value);
    void set_dropout_on(bool value);
    void set_dropout_threshold(float value);
    void set_regularization_lambda(float value);
    void set_n_data_samples(int value);
    void set_n_data_validation(int value);
    void set_learning_decay_batch(int value);
    void set_error_minimum(float value);
    void set_last_cost_train(float value);
    void set_last_cost_validation(float value);
    void set_n_epochs_trained(int value);
    void set_stop_early(bool value);
    void set_batch_show_progress(int value);
    void set_min_loss_function_error(float value);

    bool get_dropout_on();
    bool get_softmax_output_layer();
    bool get_stop_early();

    int get_batch();
    int get_epochs();
    int get_n_data_samples();
    int get_n_data_validation();
    int get_learning_decay_batch();
    int get_n_epochs_trained();
    int get_batch_show_progress();

    float get_learning_rate();
    float get_momentum();
    float get_learning_decay();
    float get_tanh_a();
    float get_tanh_b();
    float get_hit_rate(vector<vector<float> > desire, vector<vector<float> > predict, float trunc = 0.5);
    float get_dropout_threshold();
    float get_dropout_chance();
    float get_regularization_lambda();
    float get_error_minimum();
    float get_last_hit_train();
    float get_last_hit_validation();
    float get_last_cost_train();
    float get_last_cost_validation();
    float get_min_loss_function_error();

    float get_cross_product(vector<float>a,vector<float>b);
    float get_treatment(float value);
    

    vector<float> get_error_train();


protected:

    void set_weight_decay_cost(float value);
    void set_last_hit_train(float);
    void set_last_hit_validation(float);


    float get_weight_decay_cost();
    float get_mse(vector<vector<float> >samples,vector<vector<float> >out);
    float get_cross_entropy(vector<vector<float> >samples,vector<vector<float> >out);
    float get_cross_entropy_train(float d, float g);
    float get_square_error_train(float d, float g);
    


    //Activation Functio
    float tangent_hyperbolic(float vj);
    float derivate_tanh_hyperbolic(float vj);
    float sigmoidal(float vj);
    float derivate_sigmoidal(float vj);
    float softplus(float vj); //ReLu
    float derivate_softplus(float vj);
    float softmax_sum(Layer *layer);

    float rectifier(float vj);
    float derivate_rectifier(float vj);


    float derivate_softmax(float desire, float get);

    void softmax(Layer *layer);
    void set_cost_error_function();

    float (Nets::*activation_function_out)(float);
    float (Nets::*derivate_function_out)(float);
    float (Nets::*activation_function)(float);
    float (Nets::*derivate_function)(float);

    float (Nets::*loss_function)(vector<vector<float> >, vector<vector<float> >);
    float (Nets::*loss_function_train)(float d, float g);
    

    vector<float>error;

   bool dropout;
   float dropout_threshold;



private:

    bool softmax_activation, stop_early;

    int epochs,batch_data, n_samples, n_validation, learning_decay_batch, n_epochs_treined, batch_show_progress;

    float learning_rate, alpha, learning_decay, tanh_a, tanh_b,lambda_l2, error_minimum, weight_decay_cost, min_loss_function_error;
    float last_hit_train, last_hit_validation, last_cost_train, last_cost_validation, cost_train, cost_validation;



};

#endif // NETS_H
