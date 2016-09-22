#ifndef RBM_H
#define RBM_H

#include <iomanip>
#include <iostream>
#include <fstream>
#include <cstddef>

#include <vector>
#include <string>
#include <iostream>
#include "nets.h"


using namespace std;

class Neuron_RBM:public Neuron{

public:
    vector<float> momentum,batch, batch_momentum;
    float pos_prob, neg_prob, neg_state, pos_state, bias, bias_momentum, pos_corr, neg_corr,batch_bias;

    Neuron_RBM(int connections);
    Neuron_RBM();

};



class RBM: public Nets{


public:

    RBM(vector<vector<float> >_samples, int n_neurons);
    RBM(){}


    void run(int epochs=100);

    void predict(vector<vector<float> > _samples, vector<vector<float> > &back);

    void reconstruction_samples(vector<vector<float> > _samples, vector<vector<float> > &back);    

    void set_sigma(float value);

    void set_gaussian_bernoulli(bool value);

    void set_gibbs_steps(int value);

    void set_threshold_rbm_overfit(float value);

    void set_data_batch(int value);

    void set_validation(vector<vector<float> >_samples);

    void set_train_data(vector<vector<float> >_samples);

    float get_precision();

    float get_sigma();

    float get_free_energy_samples();

    float get_free_energy_validation();

    static void load_net(RBM *net,string path);

    void save_net(RBM *net, string path);

    vector<Neuron_RBM>hidden,inputs;


private:

    float  free_energy_samples, free_energy_val,sigma, threshold_rbm_overfit;

    int gibbs_steps;

    bool gaussian_bernoulli;

    vector<float>precision;

    vector<vector<float> >samples,validation,pcd_v,pcd_h;
    
    void cdn(vector<vector<float> >_samples, int sampling = 1);

    void pcd(vector<vector<float> >_samples, int sampling = 1);

    
    void get_gibs_sampling(int iter);

    float get_free_energy_calc(vector<vector<float> >_samples);

    void set_free_energy_samples(float value);

    void set_free_energy_validation(float value);

    void update_weights(bool batch_update);


    void update_weight_batch(int batch);

    void weight_decay();

    void save_layer(string path);

    void check_samples(vector<vector<float> >&samples);


};



#endif // RBM_H
