#include "connectmysql.h"
#include "../Utilitarios/utilitarios.h"
#include "../NeuralNetWork/dnn.h"
#include "../NeuralNetWork/nets.h"
#include "../NeuralNetWork/mlp_oo.h"




Connection * ConnectMysql::init_connetion(){

    Driver *driver;
    Connection *conexao;

    driver  = get_driver_instance();

    conexao = driver->connect(tcp,user,password);

    conexao->setSchema(database);

    return conexao;

}

ConnectMysql::ConnectMysql(){


    user = "jeferson"; password = "gedr0123";tcp  ="tcp://127.0.0.1:3306"; database = "bd_resultado_fmri";
}

ConnectMysql::ConnectMysql(string _user, string _password, string _tcp, string _database){

    user = _user; password = _password; tcp = _tcp; database = _database;
}

//*************************************************************************************************************************************************1

int ConnectMysql::execute_query(string sql_comand){

    try{

        Statement *stmt;
        ResultSet *res;

        Connection *conexao = init_connetion();

        stmt = conexao->createStatement();

        res = stmt->executeQuery(sql_comand);

        int net_id;
        while(res->next()){ net_id = res->getInt(1); }

        delete conexao;
        delete stmt;
        delete res;

        return net_id;

    }catch(sql::SQLException &e){
        cout << "# ERR: SQLException in " << __FILE__;
        cout << "(" << __FUNCTION__ << ") on line "
             << __LINE__ << endl;
        cout << "# ERR: " << e.what();
        cout << " (MySQL error code: " << e.getErrorCode();
        cout << ", SQLState: " << e.getSQLState() << " )" << endl;
    }
    return -1;
}

//*************************************************************************************************************************************************1

bool ConnectMysql::insert_values(string sql_comand){

    try {

        PreparedStatement *pstmt;

        Connection *conexao = init_connetion();

        pstmt = conexao->prepareStatement(sql_comand);

        pstmt->execute();

        //delete driver;
        delete conexao;
        delete pstmt;

        return 1;

    } catch(sql::SQLException &e){
        cout << "# ERR: SQLException in " << __FILE__;
        cout << "(" << __FUNCTION__ << ") on line "
             << __LINE__ << endl;
        cout << "# ERR: " << e.what();
        cout << " (MySQL error code: " << e.getErrorCode();
        cout << ", SQLState: " << e.getSQLState() << " )" << endl;
    }
    return 0;
}

//*************************************************************************************************************************************************1

int ConnectMysql::search_net(vector<int> hidden_layer, vector<float> momento, vector<float> aprendizado, vector<int> epocas, vector<string> conf){

    vector<string>comand(5);

    comand[0] =(to_string(hidden_layer.size())); //numero de camadas

    for(int i = 0; i < (int)hidden_layer.size();i++){ // neuronios por camada
        comand[1]+= to_string(hidden_layer[i]);
        if(i < ((int)hidden_layer.size()-1) )comand[1] +="-";
    }

    for(int i = 0; i < (int)aprendizado.size();i++){ //taxas de aprendizado
        comand[2]+=to_string( aprendizado[i]);
        if(i < ((int)aprendizado.size()-1) )comand[2] +="-";
    }

    for(int i = 0; i < (int)momento.size();i++){ //taxas de momento
        comand[3]+=to_string( momento[i]);
        if(i < ((int)momento.size()-1) )comand[3] +="-";
    }

    for(int i = 0; i < (int)epocas.size();i++){ //numero de epocas
        comand[4]+= to_string(epocas[i]);
        if(i < ((int)epocas.size()-1) )comand[4] +="-";
    }

    for(int i = 0; i < (int)conf.size();i++){ //configuracao
        comand.push_back(conf[i]);
    }

    string sql_comand;
    sql_comand+= "select  search_net(";
    for(int i = 0; i < (int)comand.size(); i++){

        sql_comand += "'";
        sql_comand += comand[i];
        sql_comand += "'";

        if(i < (int)comand.size()-1)
            sql_comand+=",";
    }
    sql_comand += ")";

    cout<<sql_comand<<endl;

    exit(0);

    return execute_query(sql_comand);
}

//*************************************************************************************************************************************************1

int ConnectMysql::search_net(DNN dnn){

    string sql_comando = "select search_net_regularizacao(";

    sql_comando += to_string(dnn.layers.size()-1);

    sql_comando+= ",";
    sql_comando+= "'";

    for(int i = 0; i < (int)dnn.layers.size()-1; i++){
        sql_comando += to_string(dnn.layers[i].neurons.size());

        if(! (i == (int)dnn.layers.size()-2))
            sql_comando+= "-";
    }
    sql_comando+= "'";
    sql_comando+= ",";
    sql_comando+= "'";

    for(int i  = 0; i < (int)dnn.get_hidden_pre_train_learning_rate().size(); i++){

        sql_comando += to_string(dnn.get_hidden_learning_rate_index(i));

        if(! (i == (int)dnn.get_hidden_pre_train_learning_rate().size()-1))
            sql_comando+= "-";
    }
    sql_comando+= "'";
    sql_comando+= ",";
    sql_comando+= "'";

    for(int i  = 0; i < (int)dnn.get_hidden_pre_train_momentum().size(); i++){

        sql_comando += to_string(dnn.get_hidden_momentum_index(i));

        if(! (i == (int)dnn.get_hidden_pre_train_momentum().size()-1))
            sql_comando+= "-";
    }
    sql_comando+= "'";
    sql_comando+= ",";
    sql_comando+= "'";

    for(int i  = 0; i < (int)dnn.get_hidden_pre_train_epochs().size(); i++){

        sql_comando += to_string(dnn.get_hidden_epochs_index(i));

        if(! (i == (int)dnn.get_hidden_pre_train_epochs().size()-1))
            sql_comando+= "-";
    }
    sql_comando+= "'";
    sql_comando+= ",";

    sql_comando += "'"+dnn.get_net_type()+"'" + ",";

    sql_comando += "'"+to_string(dnn.get_learning_decay()) + "',";

    sql_comando += "'"+to_string(dnn.get_hidden_pre_train_batch())+ "',";

    sql_comando += "'"+to_string(dnn.get_batch())+"',";

    sql_comando += "'"+to_string(0)+"',"; // amostragem gibbs 10


    switch(dnn.LOSS_FUNC_TYPE){

    case 1:
        sql_comando += "'MSE',";
        break;
    case 2:
        sql_comando += "'CS',";
        break;
    default:
        break;

    }

    switch (dnn.FUNC_TYPE) {
    case 1:
        sql_comando += "'SIGMOIDE',";
        break;
    case 0:
        sql_comando += "'HYOERBOLIC',";
        break;
    case 2:
        sql_comando += "'SOFTPLUS',";
        break;
    case 3:
        sql_comando += "'SOFTMAX',";
        break;
    case 4:
        sql_comando += "'ReLU',";
        break;

    default:
        break;
    }

    switch (dnn.OUT_FUNC_TYPE) {
    case 1:
        sql_comando += "'SIGMOIDE',";
        break;
    case 0:
        sql_comando += "'HYOERBOLIC',";
        break;
    case 2:
        sql_comando += "'SOFTPLUS',";
        break;
    case 3:
        sql_comando += "'SOFTMAX',";
        break;
    case 4:
        sql_comando += "'ReLU',";
        break;

    default:
        break;
    }


    sql_comando += "'"+to_string(dnn.get_regularization_lambda())+"',";

    sql_comando += "'"+to_string(dnn.get_hidden_pre_train_dropout_on())+"',";

    sql_comando += "'"+to_string(dnn.get_dropout_on())+"',";

    sql_comando += "'"+to_string(dnn.get_hidden_dropout_threshold())+"',";

    sql_comando += "'"+to_string(dnn.get_dropout_threshold())+"',";

    sql_comando += "'"+to_string(dnn.get_learning_decay_batch())+"'"; // 17

    sql_comando += ");";

    return execute_query(sql_comando);
}

//*************************************************************************************************************************************************1

void ConnectMysql::insert_net_data( int base,string experimento,vector<int>hidden_layer, vector<float>momento,
                                    vector<float>aprendizado,vector<int>epocas, vector<string>conf,vector<vector<float> >results){


    int net_id = search_net(hidden_layer,momento,aprendizado,epocas,conf);

    for(int i = 0; i < (int)results.size(); i++){

        string sql_comand = "insert into tb_resultados_10_folds () values(null,'"+experimento+"',"+to_string(net_id)+","+to_string(base)+","+to_string(i+1)+",";

        for(int j = 0; j < (int)results[i].size(); j++){

            sql_comand += to_string(results[i][j]);

            if(j < (int)results[i].size()-1)
                sql_comand += ",";
        }
        sql_comand += ")";

        insert_values(sql_comand);
    }

}
//*************************************************************************************************************************************************1

void ConnectMysql::insert_experiments(DNN dnn, int base, int fold, int teste_size, float teste_hit){

    int net = search_net(dnn);

    int numero_teste = execute_query("select search_numero_teste("+to_string(net)+","+to_string(base)+","+to_string(fold)+");");

    string sql_comand = "insert into tb_experimental values(null,"+to_string(net)+","+to_string(base)+","+to_string(fold)+","+to_string(numero_teste)+",";
    sql_comand += to_string(dnn.get_n_data_samples())+","+to_string(dnn.get_n_data_validation())+","+to_string(teste_size)+",";
    sql_comand += replace(to_string(dnn.get_last_hit_train()))+","+replace(to_string(dnn.get_last_hit_validation()))+","+replace(to_string(teste_hit))+",";
    sql_comand += replace(to_string(dnn.get_last_cost_train()))+","+replace(to_string(dnn.get_last_cost_validation()))+","+to_string(dnn.get_n_epochs_trained())+");";

    insert_values(sql_comand);
}

//*************************************************************************************************************************************************1
string ConnectMysql::replace(string x){

    std::replace(x.begin(),x.end(),',','.');
    return x;
}

