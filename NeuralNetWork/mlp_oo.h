    #ifndef MLP_OO_H
#define MLP_OO_H

#include <vector>
#include <string>
#include "nets.h"

using namespace std;

//**********************************************************************
/*class Neuron_MLP:public Neuron{

public:
    Neuron_MLP(int connections);
    Neuron_MLP();

};*/
//**********************************************************************
/*class Layer_MLP{

public:

    Layer_MLP(int n_Neuron_MLP, int connections);

    vector<Neuron>neurons;
};*/


//**********************************************************************
class MLP: public Nets{

public:


    MLP();
    MLP(int input, int output, vector<int>hidden);
   // MLP(const MLP &mlp, int asd);


    void run_MLP(vector<vector<float> > samples, vector<vector<float> > out, int _epochs = 100, float percent_validadion = 0.1);

    void predict(vector<vector<float> > samples, vector<vector<float> > &out);

    void save_net(MLP *net, string path);

    void keep_training(vector<vector<float> >_samples, vector<vector<float> >out, int epochs = 100, float percent_validation = 0.1);

    void print_net_data(MLP *net);


     void relu_test();
    static void test();
    static void load_net(MLP *net, string path);

    vector<Layer>layers;

 protected:


    void start_mlp(int input, int output, vector<int> hidden);

    void gradient_level(MLP net);
    void weights_changes(MLP net);

private:


    void local_field_induced(vector<Layer> &layer);

    void forward(vector<vector<float> >samples, vector<vector<float> >out, int tag);

    void gradient_last_layer(Layer *layer,  vector<float>out);

    void gradient_hidden_layers(vector<Layer> &layer);

    void update_weights(vector<Layer> &layer, bool batch_on);
};
//**********************************************************************

#endif // MLP_OO_H
