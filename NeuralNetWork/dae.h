#ifndef DAE_H
#define DAE_H

#include "dnn.h"
#include <vector>
#include <iostream>
#include <string>

using namespace std;

class DAE:public DNN{

public:

    DAE();
    DAE(int input,int output, vector<int>_hidden, vector<int>_epochs);

    void start_dae(int input, int output, vector<int> _hidden, vector<int> _epochs);

    void run_DAE(vector<vector<float> >_samples, vector<vector<float> >_out);

protected:

private:


    void build_dae(vector<vector<float> >_samples, int batch = 10 );

};

#endif // DAE_H
