#include "dbn.h"
#include "../Utilitarios/utilitarios.h"
#include <iostream>
#include <math.h>

DBN::DBN(){

    set_gibbs_steps(1);
    set_batch(1);
    set_threshold_rbm_overfit(0.099);
    set_percent_validation(0.1);
    set_gaussian_bernoulli_rbm(true);
    set_net_type("DBN");
}

DBN::DBN(int input, int output, vector<int> _hidden, vector<int> _epochs){

    start_net(input, output,_hidden,_epochs);

    set_gibbs_steps(1);
    set_threshold_rbm_overfit(0.099);
    set_gaussian_bernoulli_rbm(true);
    set_net_type("DBN");
}

void DBN::keep_training(vector<vector<float> > samples, vector<vector<float> > out, int epochs, float per_validation){keep_training(samples,out,epochs,per_validation);}

void DBN::set_gibbs_steps(int value){gibbs_steps = value;}

void DBN::set_threshold_rbm_overfit(float value){threshold_rbm_overfitting = value;}

void DBN::set_gaussian_bernoulli_rbm(bool value){gauss_bernoulli = value;}

bool DBN::get_gaussian_bernoulli(){return gauss_bernoulli;}

int DBN::get_gibbs_steps(){return gibbs_steps;}

float DBN::get_threshold_rbm_overfit(){return threshold_rbm_overfitting;}

//**************************************************************************************
void DBN::convert_layer_RBM_to_MLP(Layer *mlp_layer, RBM rbm){

    for(int i = 0; i < (int)mlp_layer->neurons.size();i++){

        for(int j = 0; j < (int)mlp_layer->neurons[i].weight.size(); j++){
           mlp_layer->neurons[i].weight[j] = rbm.hidden[i].weight[j];

        }

        if(isnan(rbm.hidden[i].bias))
            mlp_layer->neurons[i].bias = 0;
        else
            mlp_layer->neurons[i].bias = rbm.hidden[i].bias;

        mlp_layer->neurons[i].old_bias = 0;
    }
}
//**************************************************************************************
void DBN::convert_layer_MLP_to_RBM(RBM *rbm, Layer mlp_layer ){

    for(int i = 0; i < (int)mlp_layer.neurons.size(); i++){

        for(int j = 0; j < (int)mlp_layer.neurons[i].weight.size(); j++)
            rbm->hidden[i].weight[j] = mlp_layer.neurons[i].weight[j];


        if(isnan(mlp_layer.neurons[i].bias))
            rbm->hidden[i].bias = 0;
        else
            rbm->hidden[i].bias = mlp_layer.neurons[i].bias;
    }
}
//**************************************************************************************
void DBN::keep_training_hidden_layers(MLP *mlp, vector<int> epochs, vector<float>momentum,vector<float>learning_rate, vector<vector<float> >samples,vector<vector<float> >out){

    if(epochs.size() != mlp->layers.size()-1){

        cout<<"error: number of hidden layer for training is different from MLP"<<endl;
        return;
    }
    vector<vector<float> >validation,validation_out;
    Utilitarios util;

    for(int i = 0; i < (int)epochs.size(); i++){

        util.normalize(samples);

        util.on_cross_validation(samples,out,validation,validation_out,get_percent_validation());

        RBM rbm(samples,mlp->layers[i].neurons.size());

        if(validation.size() > 0)
            rbm.set_validation(validation);

        convert_layer_MLP_to_RBM(&rbm,mlp->layers[i]);

        rbm.set_batch(get_hidden_pre_train_batch());

        rbm.set_learning_rate(learning_rate[i]);

        rbm.set_momentum(momentum[i]);

        rbm.set_gibbs_steps(get_gibbs_steps());

        rbm.set_threshold_rbm_overfit(get_threshold_rbm_overfit());

        rbm.set_gaussian_bernoulli(get_gaussian_bernoulli());

        rbm.set_regularization_lambda(get_hidden_pre_train_lambda_weight_decay());

        rbm.run(get_hidden_epochs_index(i));

        convert_layer_RBM_to_MLP(&mlp->layers[i],rbm);

        rbm.predict(samples,samples);

        if(validation.size() > 0){

            rbm.predict(validation,validation);

            util.mergeVector(samples,validation,out,validation_out);
        }
    }
}
//**************************************************************************************
void DBN::build_dbn(vector<vector<float> > _samples,vector<vector<float> > out, float  percent_validation,float threshold_rbm_overfit, int sampling ){

    Utilitarios util;
    vector<vector<float> > validation,validation_out;

    util.on_cross_validation(_samples,out,validation,validation_out,percent_validation);

    util.shuffle_io_samples(_samples,out);

    for(int i = 0; i < (int)layers.size()-1; i++){

        try{

        vector<vector<float> >aux1,aux2;

        RBM rbm(_samples,layers[i].neurons.size());

        if(get_hidden_pre_train_learning_rate().size() > 0)
            rbm.set_learning_rate(get_hidden_learning_rate_index(i));
        if(get_hidden_pre_train_momentum().size() > 0)
            rbm.set_momentum(get_hidden_momentum_index(i));

        rbm.set_validation(validation);

        rbm.set_batch(get_hidden_pre_train_batch());

        rbm.set_gibbs_steps(sampling);

        rbm.set_threshold_rbm_overfit(threshold_rbm_overfit);

        if(i ==0)
            rbm.set_gaussian_bernoulli(get_gaussian_bernoulli());
        else
            rbm.set_gaussian_bernoulli(true);

        rbm.run(get_hidden_epochs_index(i));

        convert_layer_RBM_to_MLP(&this->layers[i], rbm);

        rbm.predict(_samples,aux1);

        rbm.predict(validation,aux2);

         _samples.clear(); validation.clear();

         _samples = aux1; validation = aux2;

        }catch(exception *e){

            cout<<"DBN RUN Error  "<<e->what()<<endl;
            return;
        }
     }
}
//**************************************************************************************
void DBN::run_DBN(vector<vector<float> > _samples, vector<vector<float> > _out){


    if(_samples.size() != _out.size()){

        cout<<"Run DBN: Size's problem with samples and output (they are diferent!)"<<endl;
        return;
    }


    try {
        build_dbn(_samples,_out,get_percent_validation(),get_threshold_rbm_overfit(), get_gibbs_steps());

    } catch (exception *e) {

        cout<<"Build DBN error  "<<e->what()<<endl;
    }

    set_learning_rate(get_hidden_learning_rate_index(get_hidden_pre_train_learning_rate().size()-1));
    
    set_momentum(get_hidden_momentum_index(get_hidden_pre_train_momentum().size()-1));

    try{
        run_MLP(_samples,_out,get_hidden_epochs_index(get_hidden_pre_train_epochs().size()-1),get_percent_validation());
    }catch(exception *e){

        cout<<"Final Adustment error  "<<e->what()<<endl;
        return;
    }
}
//**************************************************************************************

