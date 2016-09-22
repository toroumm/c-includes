#ifndef DBN_H
#define DBN_H

#include "dnn.h"
#include "rbm.h"
#include <vector>
#include <string>

using namespace std;

class DBN: public DNN
{
public:
    DBN();
    DBN(int input,int output, vector<int>_hidden, vector<int>_epochs);

    void run_DBN(vector<vector<float> >_samples, vector<vector<float> >_out);

    void keep_training_hidden_layers(MLP *mlp, vector<int> epochs, vector<float>momentum, vector<float>learning_rate, vector<vector<float> >samples, vector<vector<float> > out);

    void keep_training(vector<vector<float> >desire, vector<vector<float> >predict, int epochs, float per_validation);
    void set_threshold_rbm_overfit(float value);
    void set_gibbs_steps(int value);
    void set_gaussian_bernoulli_rbm(bool value);
    void set_validation_rbm(vector<vector<float> >_samples, vector<vector<float> > _out);

    bool get_gaussian_bernoulli();
    int get_gibbs_steps();
    float get_threshold_rbm_overfit();

protected:

private:

    bool gauss_bernoulli;

    int gibbs_steps;

    float threshold_rbm_overfitting;

    void convert_layer_RBM_to_MLP(Layer *mlp_layer, RBM rbm);
    void convert_layer_MLP_to_RBM(RBM *rbm, Layer mlp_layer);
    void build_dbn(vector<vector<float> >_samples, vector<vector<float> >out, float percent_validation = 0.1, float threshold_rbm_overfit = 0.00999, int sampling =1);

};

#endif // DBN_H
