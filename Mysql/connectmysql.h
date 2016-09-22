#ifndef CONNECTMYSQL_H
#define CONNECTMYSQL_H


#include "mysql_connection.h"
#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>
#include <cppconn/prepared_statement.h>
#include "../NeuralNetWork/dnn.h"
#include <vector>
#include <string>

using namespace sql;
using namespace std;

class ConnectMysql
{
public:
    ConnectMysql();
    ConnectMysql(string _user, string _password, string _tcp, string _database);

    int search_net(vector<int>hidden_layer, vector<float>momento, vector<float>aprendizado,vector<int>epocas, vector<string>conf);

    int search_net(DNN dnn);

    int execute_query(string sql_comand);

    void insert_net_data(int base,string experiemento,vector<int>hidden_layer, vector<float>momento,
                        vector<float>aprendizado,vector<int>epocas, vector<string>conf,vector<vector<float> >results);

    void insert_experiments(DNN dnn, int base, int fold, int teste_size, float teste_hit);


    string replace(string x);

    bool insert_values(string sql_comand);


 private:
    
    string user, password, tcp, database;
    
    Connection *init_connetion();
    
    /*
     * vector conf :
     * Tipo rede = DAE ou DBN ou...
     * Descrecimo Aprendizado
     * Batch MLP
     * Batch RBM
     * Amostragem Gibbs
     * Funcao Custo
     * Funcao Ativacao
     * Funcao Ativacao output
     *
     * */

};

#endif // CONNECTMYSQL_H
