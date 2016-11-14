#include "dae.h"
#include "../Utilitarios/utilitarios.h"

DAE::DAE(){}

//****************************************************************************
DAE::DAE(int input, int output, vector<int> _hidden, vector<int> _epochs){

    start_net(input,output, _hidden, _epochs);

    int pre_train_epochs = 0;
    for(int i  = 0; i < (int)_epochs.size()-1; i++)
        pre_train_epochs+= _epochs[i];
    if(pre_train_epochs > 0)
        set_net_type("DAE");
    else
        set_net_type("DNN");


}
//****************************************************************************



//****************************************************************************
void DAE::run_DAE(vector<vector<float> > _samples, vector<vector<float> > _out){

    build_dae(_samples,get_batch());

    set_learning_rate(get_hidden_learning_rate_index(get_hidden_pre_train_learning_rate().size()-1));

    set_momentum(get_hidden_momentum_index(get_hidden_pre_train_momentum().size()-1));

    set_dropout_on(get_dropout_on());

    set_dropout_threshold(get_dropout_threshold());

    run_MLP(_samples,_out,get_hidden_epochs_index(get_hidden_pre_train_epochs().size()-1),get_percent_validation());
}
//****************************************************************************

void DAE::build_dae(vector<vector<float> > _samples, int batch){

    Utilitarios util;
    vector<vector<float> > validation,validation_out;

    util.on_cross_validation(_samples,_samples,validation,validation_out,0);

    util.shuffle_io_samples(_samples,_samples);

    for(int i = 0; i < (int)layers.size()-1; i++){

        vector<int>hide(1); hide[0] = layers[i].neurons.size();

        MLP mlp(_samples[0].size(),_samples[0].size(),hide);

        if(get_hidden_pre_train_learning_rate().size() > 0)
            mlp.set_learning_rate(get_hidden_learning_rate_index(i));
        if(get_hidden_pre_train_momentum().size() > 0)
            mlp.set_momentum(get_hidden_momentum_index(i));

        mlp.set_batch(get_hidden_pre_train_batch());

        mlp.set_activation_function_hidden_layer(get_hidden_pre_train_function());

        mlp.set_activation_function_output_layer(get_hidden_pre_train_function_output());

        mlp.set_loss_function(mlp.LOSS_FUNCTION::MEAN_SQUARE_ERROR);

        mlp.set_dropout_on(get_hidden_pre_train_dropout_on());

        mlp.set_dropout_threshold(0.5);

        mlp.run_MLP(_samples,_samples,get_hidden_epochs_index(i),0);

        layers[i] = mlp.layers[0];

        mlp.predict(_samples,_samples);

        mlp.predict(validation,validation);
    }
}
//****************************************************************************

