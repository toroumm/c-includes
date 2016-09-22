#ifndef NEURON_H
#define NEURON_H


#include <vector>


using namespace std;

//###########################################################################################################
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

  vector<float>weight,old,input,delta, old_delta, weight_batch, weight_backup;

  bool  dropout_result;

  float bias,bias_batch,erro, old_bias,gradient,output,vj, bias_backup;

private:

};


//###########################################################################################################
class Neuron_RBM:public Neuron{

public:
    vector<float> momentum,batch, batch_momentum;
    float pos_prob, neg_prob, neg_state, pos_state, bias, bias_momentum, pos_corr, neg_corr,batch_bias;

    Neuron_RBM(int connections);
    Neuron_RBM();

};

//###########################################################################################################

class Neuron_MLP:public Neuron{

public:
    Neuron_MLP(int connections);
    Neuron_MLP();

};


//###########################################################################################################


#endif //NEURON_H
