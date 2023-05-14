#include "utils.hpp"

double communication, computation, total_time;

Time_Calculator mytime;
My_Buffer dt_alice(50000000), dt_bob(50000000);

NetIO * io_server;
NetIO * io_client_alice;
NetIO * io_client_bob;
NetIO * io_client;

void load_train_data(Mat& train_data, Mat& train_label){
    ifstream infile( "../data/mnist_train.csv" );
        int count1=0, count2=0;
        int i=0;
        while(infile) {

            string s;
            if (!getline(infile,s))
                break;
            istringstream ss(s);
            int temp;
            char c;


            //read label
            ss>>temp;
            ss>>c;
            if(temp == 0 && count1<=N/2) {
                train_label(i) = 0;
                count1++;

                //read data (last entry 1)
                for(int j=0; j<D-1; j++) {
                    ss>>train_data(i,j);
                    ss>>c;
                }

                train_data(i,D-1) = 1;
                i++;
            }


            if(temp != 0 && count2<=N/2) {
                train_label(i) = 1;
                count2++;

                //read data (last entry 1)
                for(int j=0; j<D-1; j++) {
                    ss>>train_data(i,j);
                    ss>>c;
                }

                train_data(i,D-1) = 1;
                i++;
            }


            if(i>=N)
                break;
        }

        //train_data.conservativeResize(i, D);
        //train_label.conservativeResize(i,1);
    infile.close();
}

int main(int argc, char** argv) {
    srand((unsigned)time(NULL));

    int port, party;
    parse_party_and_port(argv, &party, &port);
    setup(party, io_server, io_client_alice, io_client_bob, io_client);

    srand ( unsigned ( time(NULL) ) );
    
    total_time = 0.0;
    communication = 0.0;
    computation = 0.0;

    cout << "reading data......\n";

    Mat train_data(N,D), train_label(N,1);

    load_train_data(train_data, train_label);
    
    vector<int> perm = random_perm();
    
    io_client_alice->send_data(&perm[0], sizeof(int) * perm.size());
    io_client_alice->flush();
    io_client_bob->send_data(&perm[0], sizeof(int) * perm.size());
    io_client_bob->flush();
    
    int hahaha1;
    io_client_alice->recv_data(&hahaha1, sizeof(int));
	io_client_bob->recv_data(&hahaha1, sizeof(int));
	
    
    Mat x_batch(B,D), y_batch(B,1), z0_batch(D,1), z1_batch(B,1), tau0_batch(B,1), tau1_batch(D, 1);
    Mat z0(IT, D), z1(IT, B), tau0(IT * B, 1), tau1(IT, D);
    
    PRF prf_ALICE, prf_BOB;
    
	receive_keys(&prf_ALICE, io_client_alice, 1);
	receive_keys(&prf_BOB, io_client_bob, 1);
	
	distribute_mat_same(train_data, &prf_ALICE, &prf_BOB, &dt_alice, &dt_bob);
	distribute_mat_same(train_label, &prf_ALICE, &prf_BOB, &dt_alice, &dt_bob);
	dt_alice.send(io_client_alice);
	dt_bob.send(io_client_bob);
	
    add_mat(&prf_ALICE, &prf_BOB, z0);
    add_mat(&prf_ALICE, &prf_BOB, z1);
	
    int start = 0;
    
    for (int i = 0; i < IT; ++i, start += B) {
    
    	next_batch(x_batch,start,perm,train_data);
        next_batch(y_batch,start,perm,train_label);
        
        z0_batch = (z0.row(i)).transpose();
        z1_batch = (z1.row(i)).transpose();
        //cout << "calculation...\n";
        
        tau0_batch = x_batch * z0_batch;
        tau1_batch = x_batch.transpose() * z1_batch;
        
        
        //cout << "distribute mat...\n";
       
		tau0.block<B,1>(i * B, 0) = tau0_batch;
		tau1.row(i) = tau1_batch.transpose();
        
    }
    
	distribute_mat_same(tau0, &prf_ALICE, &prf_BOB, &dt_alice, &dt_bob);
	distribute_mat_same(tau1, &prf_ALICE, &prf_BOB, &dt_alice, &dt_bob);
	dt_alice.send(io_client_alice);
	dt_bob.send(io_client_bob);

	
	if (party != EVE) {
		delete io_server;
	}
	if (party != BOB) {
		delete io_client_alice;
	}
	if (party != ALICE) {
		delete io_client_bob;
	}
}
