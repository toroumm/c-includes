    #ifndef DNN_H
#define DNN_H

#include "mlp_oo.h"
#include <iostream>
#include <vector>
#include <string>


class DNN: public MLP
{
public:

    DNN();
    DNN(int input, int output, vector<int>_hidden);

    void start_net(int input, int output, vector<int> _hidden, vector<int> _epochs);

    void set_hidden_pre_train_learning_rate(vector<float>_learning);
    void set_hidden_pre_train_momentum(vector<float>_momentum);
    void set_percent_validation(float value);

    void set_hidden_pre_train_dropout_on(bool value);
    void set_hidden_pre_train_dropout_threshold(float value);
    void set_hidden_pre_train_epochs(vector<int>epochs);
    void set_hidden_pre_train_batch(int value);
    void set_hidden_pre_train_act_function(ACT_FUNC_TYPE value);
    void set_hidden_pre_train_act_function_output(ACT_FUNC_TYPE value);
    void set_hidden_pre_train_lambda_weight_decay(float value);
    void set_net_type(string value);

    void set_hidden_layer(vector<int>layers);

    void run_DNN(vector<vector<float> > _samples, vector<vector<float> > _out, int epochs, float _percent_validation);

    string get_net_type();
    bool get_hidden_pre_train_dropout_on();
    int get_hidden_pre_train_batch();

    int get_hidden_epochs_index(int index);

    float get_percent_validation();
    float get_hidden_learning_rate_index(int index);
    float get_hidden_momentum_index(int index);
    float get_hidden_dropout_threshold();
    float get_hidden_pre_train_lambda_weight_decay();

    ACT_FUNC_TYPE get_hidden_pre_train_function();
    ACT_FUNC_TYPE get_hidden_pre_train_function_output();

    vector<float>get_hidden_pre_train_momentum();
    vector<float>get_hidden_pre_train_learning_rate();
    vector<int> get_hidden_pre_train_epochs();

private:


protected:

    ACT_FUNC_TYPE pre_train_function, pre_train_function_out;

    vector<int> hidden_layers,hidden_epochs;
    vector<float>hidden_learning_rate, hidden_momentum;

    bool dropout_hidden_on;
    int batch_hidden;
    float percent_validation, dropout_hidden_threshold,lamda_hidden_weight_decay;
    string net_type;

    void start_dnn(int input,int output, vector<int>_hidden, vector<int>_epochs);
};

#endif // DNN_H
