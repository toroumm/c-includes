#include "dnn.h"

DNN::DNN(){}

DNN::DNN(int input, int output, vector<int> _hidden){

    string erro= "DNN (constructor) :  ";
    if(input <= 0){

        cout<<erro+"Problem with size of input "<<endl; return;
    }
    if(output <= 0){

        cout<<erro+"Problem with size of output"<<endl; return;
    }

    start_mlp(input,output,_hidden);

    set_activation_function_hidden_layer(ReLU);

    set_loss_function(CROSS_ENTROPY);
}

//*************************************************************************************
void DNN::set_hidden_pre_train_dropout_on(bool value){dropout_hidden_on = value;}
void DNN::set_hidden_pre_train_dropout_threshold(float value){ dropout_hidden_threshold =value;}
void DNN::set_hidden_pre_train_batch(int value){if(value > 0)batch_hidden = value;}

void DNN::set_hidden_pre_train_epochs(vector<int>epochs){hidden_epochs = epochs;}
void DNN::set_hidden_pre_train_act_function(ACT_FUNC_TYPE value){pre_train_function =value;}
void DNN::set_hidden_pre_train_act_function_output(ACT_FUNC_TYPE value){pre_train_function_out =value;}

void DNN::set_hidden_pre_train_learning_rate(vector<float> _learning){hidden_learning_rate = _learning;}
void DNN::set_hidden_pre_train_momentum(vector<float> _momentum){hidden_momentum = _momentum;}

void DNN::set_percent_validation(float value){percent_validation = value;}
void DNN::set_hidden_layer(vector<int>layers){hidden_layers = layers;}
void DNN::set_hidden_pre_train_lambda_weight_decay(float value){lamda_hidden_weight_decay = value;}
void DNN::set_net_type(string value){net_type = value;}

bool DNN::get_hidden_pre_train_dropout_on(){return dropout_hidden_on;}
int DNN::get_hidden_pre_train_batch(){return batch_hidden;}

int DNN::get_hidden_epochs_index(int index){return hidden_epochs[index];}
float DNN::get_percent_validation(){return percent_validation;}
float DNN::get_hidden_learning_rate_index(int index){return hidden_learning_rate[index];}
float DNN::get_hidden_momentum_index(int index){return hidden_momentum[index];}
float DNN::get_hidden_dropout_threshold(){return dropout_hidden_threshold;}
float DNN::get_hidden_pre_train_lambda_weight_decay(){return lamda_hidden_weight_decay;}
string DNN::get_net_type(){return net_type;}

DNN::ACT_FUNC_TYPE DNN::get_hidden_pre_train_function(){return pre_train_function;}
DNN::ACT_FUNC_TYPE DNN::get_hidden_pre_train_function_output(){return pre_train_function_out;}

vector<float>DNN::get_hidden_pre_train_momentum(){return hidden_momentum;}
vector<float>DNN::get_hidden_pre_train_learning_rate(){return hidden_learning_rate;}
vector<int>DNN::get_hidden_pre_train_epochs(){return hidden_epochs;}


//*************************************************************************************
void DNN::run_DNN(vector<vector<float> > _samples, vector<vector<float> > _out, int epochs, float _percent_validation){

    run_MLP(_samples,_out,epochs,_percent_validation);
}

//************************************************************************************
