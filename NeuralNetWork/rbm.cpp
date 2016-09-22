#include "rbm.h"
#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <random>
#include <fstream>

//***************************************************************************
Neuron_RBM::Neuron_RBM(int connections){

    start_neuron(connections);

    batch.resize(connections);
    momentum.resize(connections);
    bias_momentum = 0;
    batch_bias = 0;

    random_device rd; mt19937 gen(rd());
    normal_distribution <>d(0,1);

    if(connections > 0)
        bias =  fabs(d(gen)*0.01); //float(((rand()%200 + 1)-100)/1000.0);
    else
        bias = 0;
}

Neuron_RBM::Neuron_RBM(){}
//***************************************************************************

RBM::RBM(vector<vector<float> > _samples, int n_neurons){

    if(_samples.size() ==0){

        cout<<"RBM (constructor): Samples size cannot be zero"<<endl; return;
    }


    if(n_neurons ==0){

        cout<<"RBM (constructor): Number of neurons cannot be zero or negative "<<endl; return;
    }

    int connections = _samples[0].size();

    for(int i = 0; i < n_neurons;i++)
        hidden.push_back(Neuron_RBM(connections));

    for(int i = 0; i < connections;i++)
        inputs.push_back(Neuron_RBM(0));

    samples = _samples;

    set_sigma(0.001);
    set_gaussian_bernoulli(true);
    set_gibbs_steps(1);
    set_threshold_rbm_overfit(0.099);
    set_regularization_lambda(0.000);

}
//***************************************************************************
void RBM::check_samples(vector<vector<float> > &samples){

    for(int i =0;i <(int)samples.size();i++){

        for(int j= 0; j < (int)samples[i].size();j++){
            cout<<isnan(samples[i][j])<<endl;
            if(isnan(samples[i][j]) == 1)
                samples[i][j] =1;
        }
    }
}
//***************************************************************************
float RBM::get_free_energy_calc(vector<vector<float> > _samples){

    if(_samples.size() ==0)
        return 0;

    float a = 0, b =0;
    for(int i = 0; i < (int)_samples.size(); i++){
        float aa = 0, bb =0;
        for(int j = 0; j < (int) hidden.size(); j++){
            float bbb = 0;
            for(int k = 0; k < (int)inputs.size(); k++)
                bbb +=hidden[j].weight[k] * _samples[i][k];
            bb += log(1 + exp(sigmoidal(bb+hidden[j].bias)));
            // bb += log(1 + exp((this->*activation_function)(bb+hidden[j].bias)));
        }b += (bb / hidden.size());

        for(int j = 0; j < (int)inputs.size(); j++)
            aa += _samples[i][j] * inputs[j].bias;
        a+= aa/inputs.size();
    }
    return (((-1)*a) - b)/_samples.size();
}

float RBM::get_precision(){return precision[precision.size()-1]/(samples.size()*samples[0].size());}

float RBM::get_free_energy_samples(){return free_energy_samples;}

float RBM::get_sigma(){return sigma;}

float RBM::get_free_energy_validation(){return free_energy_val;}

//***************************************************************************
void RBM::set_train_data(vector<vector<float> > _samples){samples = _samples;}

void RBM::set_free_energy_samples(float value){free_energy_samples = value;}

void RBM::set_free_energy_validation(float value){free_energy_val = value;}

void RBM::set_sigma(float value){sigma = value;}

void RBM::set_gaussian_bernoulli(bool value){gaussian_bernoulli =value;}

void RBM::set_validation(vector<vector<float> > _samples){validation = _samples;}

void RBM::set_gibbs_steps(int value){gibbs_steps = value;}

void RBM::set_threshold_rbm_overfit(float value){threshold_rbm_overfit =value;}

//***************************************************************************
void RBM::predict(vector<vector<float> > _samples, vector<vector<float> > &back){

    back.clear();

    back.resize(_samples.size());
    for(int i = 0; i  < (int)_samples.size(); i++){

        for(int j = 0; j < (int)hidden.size(); j++){
            hidden[j].pos_prob = 0;
            for(int k = 0; k < (int)_samples[i].size(); k++)
                hidden[j].pos_prob += hidden[j].weight[k]*_samples[i][k];

            back[i].push_back(sigmoidal(hidden[j].pos_prob + hidden[j].bias));
        }
    }
}
//***************************************************************************

void RBM::reconstruction_samples(vector<vector<float> > _samples, vector<vector<float> > &back){

    back.clear();
    back.resize(_samples.size());

    default_random_engine gear;

    random_device rd; mt19937 m(rd());

    normal_distribution<>d(0,sigma);

    for(int i = 0; i  < (int)_samples.size(); i++){

        for(int j = 0; j < (int)hidden.size(); j++){
            hidden[j].pos_prob = 0;
            for(int k = 0; k < (int)_samples[i].size(); k++)
                hidden[j].pos_prob += hidden[j].weight[k]*_samples[i][k];
            hidden[j].pos_prob = sigmoidal(hidden[j].pos_prob + hidden[j].bias);

            bernoulli_distribution dist(hidden[j].pos_prob);

            if(dist(gear))
                hidden[j].pos_state = 1;
            else
                hidden[j].pos_state = 0;
        }

        for(int j = 0; j < (int)inputs.size(); j++){
            inputs[j].neg_prob =0;
            for(int k = 0; k < (int)hidden.size(); k++)
                inputs[j].neg_prob +=  hidden[k].weight[j]*hidden[k].pos_state;

            if(gaussian_bernoulli){
                inputs[j].neg_prob /= inputs.size();

                back[i].push_back((inputs[j].neg_prob + inputs[j].bias));
            }
            else
                back[i].push_back(sigmoidal(inputs[j].neg_prob + inputs[j].bias));
        }
    }
}
//***************************************************************************

void RBM::update_weights(bool batch_update){

    //atualizacao de pesos
#pragma omp parallel
    {
#pragma omp for
        for(int j = 0; j < (int)hidden.size(); j++){
            for(int k = 0; k < (int)inputs.size(); k++){

                float a = ((hidden[j].pos_state * inputs[k].pos_prob) - (inputs[k].neg_prob * hidden[j].neg_state));

                float decay  = get_regularization_lambda() * hidden[j].weight[k] * hidden[j].weight[k];
                 hidden[j].weight[k] +=  (get_momentum() * hidden[j].momentum[k])  + (get_learning_rate() * a) - decay;

                 if(isnan(hidden[j].weight[k]) || isinf(hidden[j].weight[k]))
                        hidden[j].weight[k] = 0;
                 hidden[j].momentum[k] =a;
            }
            hidden[j].bias += (get_momentum()*hidden[j].bias_momentum) + (get_learning_rate()* (hidden[j].pos_prob - hidden[j].neg_prob));

            hidden[j].bias_momentum = (hidden[j].pos_prob - hidden[j].neg_prob);

            if(isnan(hidden[j].bias) || isinf(hidden[j].bias))
                hidden[j].bias = 0;
        }
   }

#pragma omp parallel
    {
#pragma omp for
        for(int j = 0; j < (int)inputs.size(); j++){

           inputs[j].bias +=   (get_momentum()* inputs[j].bias_momentum) + (get_learning_rate() * (inputs[j].pos_prob - inputs[j].neg_prob));

            inputs[j].bias_momentum  = ((inputs[j].pos_prob - inputs[j].neg_prob));

            if(isnan(inputs[j].bias) || isinf(inputs[j].bias))
                inputs[j].bias = 0;
        }
    }
}

//***************************************************************************
void RBM::pcd(vector<vector<float> > _samples, int sampling){

}
//***************************************************************************


void RBM::cdn(vector<vector<float> >_samples, int sampling){

    precision.push_back(0);
    for(int i = 0; i  < (int)_samples.size(); i++){

        for(int j = 0; j < (int)inputs.size(); j++)
            inputs[j].pos_prob = _samples[i][j];
        //Calculo p(h = 1 | v) pos = positivo e neg = negativo

        default_random_engine gear;
#pragma omp parallel
        {
#pragma omp for
            for(int j = 0; j < (int)hidden.size(); j++){
                hidden[j].pos_prob = 0;

                for(int k = 0; k < (int)_samples[i].size(); k++)
                    hidden[j].pos_prob += hidden[j].weight[k]*_samples[i][k];

                 hidden[j].pos_prob = sigmoidal(hidden[j].pos_prob + hidden[j].bias);

                bernoulli_distribution dis(hidden[j].pos_prob);
                if(dis(gear))
                    hidden[j].pos_state = 1;
                else
                    hidden[j].pos_state = 0;

                hidden[j].neg_state = hidden[j].pos_state;
                hidden[j].neg_prob =  hidden[j].pos_prob;
            }
        }
        try {
            get_gibs_sampling(sampling);
        } catch (exception *e) {
            cout<<"Gibbs Sampling  "<<e->what()<<endl;
        }


        for(int j = 0; j<(int)inputs.size(); j++)
            precision[precision.size()-1] += fabs(inputs[j].pos_prob - inputs[j].neg_prob);

        try {
          update_weights(!(i%get_batch()));
        } catch (exception *e) {

            cout<<"Update Weights  "<<e->what()<<endl;
        }

    }
}


//***************************************************************************
void RBM::get_gibs_sampling(int iter){

    for(int i  = 0; i < iter; i++){
        //Calculo p(v' | h)

        normal_distribution<>d(0,sigma);
#pragma omp parallel
        {
#pragma omp for

            for(int j = 0; j < (int)inputs.size(); j++){
                inputs[j].neg_prob =0;

                for(int k = 0; k < (int)hidden.size(); k++){
                    inputs[j].neg_prob += hidden[k].weight[j]*hidden[k].neg_state;
                }
                if(gaussian_bernoulli){

                    inputs[j].neg_prob/=hidden.size();

                    inputs[j].neg_prob += inputs[j].bias;

                    random_device rd; mt19937 m(rd());

                    inputs[j].neg_state = inputs[j].neg_prob +d(m)*sigma;

                }else{

                    inputs[j].neg_prob = sigmoidal(inputs[j].neg_prob+ inputs[j].bias);

                    default_random_engine gear;

                    bernoulli_distribution dis(inputs[j].neg_prob);

                    if(dis(gear))
                        inputs[j].neg_state =1;
                    else
                        inputs[j].neg_prob = 0;
                }
            }
       }

        //Calculo p(h' = 1 | v')
        default_random_engine gear;
#pragma omp parallel
        {

#pragma omp for
            for(int j = 0; j < (int)hidden.size(); j++){
                hidden[j].neg_prob = 0;

                for(int k = 0; k < (int)inputs.size(); k++){
                    hidden[j].neg_prob += hidden[j].weight[k] * inputs[k].neg_state;

                }

                hidden[j].neg_prob = sigmoidal(hidden[j].neg_prob + hidden[j].bias );

                bernoulli_distribution dis(hidden[j].neg_prob);

                if(dis(gear))
                    hidden[j].neg_state =1;
                else
                    hidden[j].neg_state =0;
            }
        }
    }

}
//***************************************************************************

void RBM::run(int epochs){

    int c_status = 0,c_energy=0;

    float energy = 0;

    while(epochs > 0){

        try {
            cdn(samples,gibbs_steps);
        } catch (exception *e) {
            cout<<"Constrative Divergence "<<e->what()<<endl;
            break;
        }
        epochs--;

        set_free_energy_samples(get_free_energy_calc(samples));

        set_free_energy_validation(get_free_energy_calc(validation));

      /*  cout<<epochs<<"  "<<get_free_energy_samples()<<" "<<get_free_energy_validation()<<" "<<get_free_energy_samples()-get_free_energy_validation()<<
             " Precision "<<precision[precision.size()-1]<<"  % mean  "<<1-(precision[precision.size()-1]/(samples.size()*samples[0].size()))<<
             " Status "<<c_status<<"  "<<c_energy<<"  "<<get_learning_rate()<<endl;*/

        energy = get_free_energy_samples();
        if(!(epochs % 20))
            set_learning_rate(get_learning_rate()*get_learning_decay());
        if(validation.size() > 0){
            if(fabs(get_free_energy_samples()-get_free_energy_validation())> threshold_rbm_overfit){
                c_status++;
                if(c_status >=50)
                    break;
            }else
                c_status =1;
        }
    }
}

//***************************************************************************
void RBM::save_layer(string path){

    ofstream filePesos, fileBias;
    filePesos.open(path+"_pesos"+".txt");
    fileBias.open(path+"_bias"+".txt");
    if(!filePesos.is_open()){
        cout<<"Erro ao abrir arquivo "<<endl;
        return;
    }

    if(!fileBias.is_open()){
        cout<<"Erro ao abrir arquivo "<<endl;
        return;
    }

    for(int i = 0; i < (int)hidden.size(); i++){
        for(int j =0; j < (int)hidden[i].weight.size();j++){
            filePesos<<hidden[i].weight[j]<<" ";
        }filePesos<<endl;
        fileBias<<hidden[i].bias<<endl;
    }
    filePesos.close();
    fileBias.close();
}
//***************************************************************************

void RBM::save_net( RBM *net,string path){

    ofstream file;
    file.open(path);

    if(!file.is_open()){
        cout<<"Erro ao abrir arquivo: "<<path<<endl;
        return;
    }

    file<<net->get_learning_rate()<<" ";
    file<<net->get_momentum()<<" "<<endl;

    file<<net->inputs.size()<<" ";
    file<<net->hidden.size()<<endl;

    //saving inputs neurons
    for(int i = 0; i < (int)net->inputs.size(); i++){

        file<<net->inputs[i].pos_prob<<" "<<net->inputs[i].neg_prob<<" ";
        file<<net->inputs[i].pos_state<<" "<<net->inputs[i].neg_state<<" "<<endl;
        file<<inputs[i].momentum.size()<<" ";
        for(int j = 0; j < (int)inputs[i].momentum.size();j++)
            file<<net->inputs[i].momentum[j]<<" ";
        file<<net->inputs[i].bias_momentum<<endl;

    }

    //saving hidden neurons
    for(int i = 0; i < (int)net->hidden.size(); i++){

        for(int j = 0; j < (int)net->hidden[i].weight.size(); j++)
            file<<net->hidden[i].weight[j]<<" ";

        file<<net->hidden[i].pos_prob<<" "<<net->hidden[i].neg_prob<<" "<<endl;
        file<<hidden[i].momentum.size()<<" ";
        for(int j = 0; j < (int)hidden[i].momentum.size();j++)
            file<<net->hidden[i].momentum[j]<<" ";
        file<<net->hidden[i].bias_momentum<<endl;
    }

    file<<net->samples.size()<<" "<<net->validation.size()<<endl;

    //saving train set
    for(int i = 0; i < (int)net->samples.size(); i++){
        file<<net->samples[i].size()<<" ";
        for(int j = 0; j < (int)net->samples[i].size(); j++)
            file<<net->samples[i][j]<<" ";

        file<<endl;
    }
    //saving validation set
    for(int i = 0; i < (int)net->validation.size(); i++){
        file<<net->validation[i].size()<<" ";
        for(int j = 0; j < (int)net->validation[i].size(); j++)
            file<<net->validation[i][j]<<" ";

        file<<endl;
    }
    file.close();
}
//***************************************************************************
void RBM::load_net(RBM *net, string path){


    ifstream file;
    file.open(path);
    if(!file.is_open()){
        cout<<"Erro ao abrir arquivo: "<<path<<endl;
        return;
    }
    float x;
    file>>x; net->set_learning_rate(x);
    file>>x; net->set_momentum(x);

    int in_size, hidden_size;

    file>>in_size;
    file>>hidden_size;

    net->inputs.resize(in_size);
    net->hidden.resize(hidden_size);

    for(int i = 0; i < (int)net->inputs.size(); i++){

        file>>net->inputs[i].pos_prob;
        file>>net->inputs[i].neg_prob;
        file>>net->inputs[i].pos_state;
        file>>net->inputs[i].neg_state;
        int t; file>>t;
        net->inputs[i].momentum.resize(t);
        for(int j = 0; j < (int)net->inputs[i].momentum.size();j++)
            file>>net->inputs[i].momentum[j];
        file>>net->inputs[i].bias_momentum;
    }

    for(int i = 0; i < (int)net->hidden.size(); i++){
        net->hidden[i].weight.resize(net->inputs.size());
        for(int j = 0; j < (int)net->hidden[i].weight.size(); j++)
            file>>net->hidden[i].weight[j];

        file>>net->hidden[i].pos_prob;
        file>>net->hidden[i].neg_prob;
        file>>net->hidden[i].pos_state;
        file>>net->hidden[i].neg_state;
        int t; file>>t;
        net->hidden[i].momentum.resize(t);
        for(int j = 0; j < (int)net->hidden[i].momentum.size();j++)
            file>>net->hidden[i].momentum[j];
        file>>net->hidden[i].bias_momentum;
    }

    int samples_size, validation_size;

    file>>samples_size;
    file>>validation_size;

    net->samples.resize(samples_size);
    net->validation.resize(validation_size);

    for(int i = 0; i < (int)net->samples.size(); i++){
        int size; file>>size;
        net->samples[i].resize(size);
        for(int j = 0; j < (int)net->samples[i].size(); j++)
            file>>net->samples[i][j];
    }

    for(int i = 0; i < (int)net->validation.size(); i++){
        int size; file>>size;
        net->validation[i].resize(size);
        for(int j = 0; j < (int)net->validation[i].size(); j++)
            file>>net->validation[i][j];
    }
    file.close();
}
//***************************************************************************


