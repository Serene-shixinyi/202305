#include "utils.hpp"

double total_time;
double COMM, COMM_client;

double offline;

Time_Calculator mytime;
My_Buffer dt;

NetIO * io_server;
NetIO * io_client_alice;
NetIO * io_client_bob;
NetIO * io_client;

PRF prf_client[B], prf_client_same;

void load_test_data(Mat& test_data, Mat& test_label) {
    ifstream infile2( "../data/mnist_test.csv" );
    int i=0;

    cout<<"load testing data.......\n";

    while(infile2) {

        string s;
        if (!getline(infile2,s))
            break;
        istringstream ss(s);
        int temp;
        char c;

        //read label
        ss>>temp;
        ss>>c;

        //if(temp == 0 || temp == 1){
        test_label(i) = (temp!=0);


        //read data (last entry 1)
        for(int j=0; j<D-1; j++) {
            ss>>test_data(i,j);
            ss>>c;
        }

        test_data(i,D-1) = 1;
        i++;
        //}

    }

    test_data.conservativeResize(i, D);
    test_label.conservativeResize(i,1);

    infile2.close();

    return;
}


double test_model(Mat& W0, Mat& W1, Mat& x, Mat& y) {
    Mat y_,W;
    double temp1;
    long int temp2,temp3;

    W = W0+W1;
    y_ = x*W;

    int count = 0;

    for(int i=0; i<y.rows(); i++) {
        temp3 = (long int)y_(i);
        //temp3 = (temp3<<4);

        //if(temp3>conv<long int>(p)/2)
        //	temp3 = temp3-conv<long int>(p);

        temp1 = temp3/(double)pow(2,L+8);
        //temp1 = conv<long int>(y_[i][0])/(double)pow(2,L);
        temp2 = (long int)y(i);

        //if(temp2>conv<long int>(p)/2)
        //	temp2 = temp2-conv<long int>(p);

        if(temp1>0.5 && temp2 == 1) {
            count++;
        } else if(temp1<0.5 && temp2 == 0) {
            count++;
        }
    }
    return count/(double)y.rows();


}

Mat secure_inner_product(Mat x, Mat z, Mat y, int party) {
    
    Mat alpha0(x.rows(), x.cols()), alpha1(x.rows(), x.cols());
	
    if(party == ALICE) alpha0 = x - z;
    else alpha1 = x - z;

    if(party == ALICE) {
        COMM += send_mat(alpha0, io_server);
        COMM += receive_mat(alpha1, io_server);

    } else {
        COMM += receive_mat(alpha0, io_server);
        COMM += send_mat(alpha1, io_server);
    }

    Mat result = y * (alpha0 + alpha1);

    return result;
}

int main(int argc, char** argv) {
    //assert(N % B == 0);

    //setup connection
    int port, party;
    parse_party_and_port(argv, &party, &port);
    setup(party, io_server, io_client_alice, io_client_bob, io_client);

    Mat test_data(testN,D), test_label(testN,1);

    if(party == ALICE && EVALUATE) {
        load_test_data(test_data, test_label);
    }
	
	
    Mat W(D,1);
    W.setZero();

    Mat x_batch(B,D), y_batch(B,1), z0_batch(D,1), z1_batch(B,1), tau0_batch(B,1), tau1_batch(B, D);
    Mat train_data(N, D), train_label(N, 1);
    Mat z0(IT, D), z1(IT, B), tau0(IT * B, 1), tau1(IT * B, D);

    Mat update0(B, 1);
    Mat omega(B, 1);

    Mat update1(B, D);

    Mat delta(D, 1);
    
    vector<int> perm(N * Ep, 0);
    io_client->recv_data(&perm[0], sizeof(int) * perm.size());
    
	
	int hahaha1 = 666, hahaha2;
	if(party == ALICE) {
		io_server->send_data(&hahaha1, sizeof(int));
		io_server->flush();
		io_server->recv_data(&hahaha2, sizeof(int));
	}
	else if(party == BOB) {
		io_server->recv_data(&hahaha2, sizeof(int));
		io_server->send_data(&hahaha1, sizeof(int));
		io_server->flush();
	}
	io_client->send_data(&hahaha1, sizeof(int));
	io_client->flush();
	
    total_time = 0.0;
    COMM = 0.0;
    COMM_client = 0.0;

    
    int t1 = mytime.start();
    
    PRF *prf_client_temp = new PRF[N];
    PRF *prf_client_same = new PRF[IT];
    PRF *prf_client = new PRF[IT * B];
	COMM_client += send_keys(prf_client_same, IT, &dt);
    COMM_client += send_keys(prf_client_temp, N, &dt);
    COMM_client += send_keys(prf_client, IT * B, &dt);
    dt.send(io_client);
    
	COMM_client += merge_mat(train_data, prf_client_temp, io_client, party);
	COMM_client += merge_mat(train_label, prf_client_temp, io_client, party);
	
	for (int i = 0; i < IT; ++i) {
		prf_client_same[i].cal(z0_batch);
		z0.row(i) = z0_batch.transpose();
		for (int j = 0; j < B; ++j) {
			Mat z1j(1, 1);
			prf_client[i * B + j].cal(z1j);
			z1(i, j) = z1j(0, 0);
			
		}
	}
	
	COMM_client += merge_mat(tau0, prf_client, io_client, party);
	COMM_client += merge_mat(tau1, prf_client, io_client, party);
	
	
	
	delete [] prf_client_temp;
	delete [] prf_client_same;
	delete [] prf_client;
	
	total_time = mytime.end(t1);
	
    cout<<"start training..."<<endl;
	

    //start training
    for (int it = 0, start = 0; it < IT; ++it, start+=B) {
        
            int t1 = mytime.start();
        	next_batch(x_batch,start,perm,train_data);
            next_batch(y_batch,start,perm,train_label);
            
            tau0_batch = tau0.block<B, 1>(it * B, 0);
            tau1_batch = tau1.block<B, D>(it * B, 0);
            
            update0 = secure_inner_product(W, (z0.row(it)).transpose(), x_batch, party);

            omega = update0 + tau0_batch - y_batch * (unsigned long int)(1<<(L+8));

            delta = secure_inner_product(omega, (z1.row(it)).transpose(), x_batch.transpose(), party);
            for (int i = 0; i < B; ++i) delta += tau1_batch.row(i).transpose();

            for (int i=0; i<delta.rows(); i++) {
                delta(i) = (long int)delta(i)/ ((long int)(1<<23)*B);
            }

            W = W - delta;

            total_time += mytime.end(t1);


            if (EVALUATE && it % 5 == 4) {
            	NetIO * io = io_server;
                vector<long int> W_temp(W.cols()*W.rows());
                Mat W1(D,1);
				//cout << "EVALUATE...\n";
                if(party==ALICE) {

                    io->recv_data(&W_temp[0],sizeof(unsigned long int)*W_temp.size());

                    for(int j=0; j<W1.rows(); j++) {
                        for(int k=0; k<W1.cols(); k++)
                            W1(j,k) = W_temp[j*W1.cols()+k];
                    }

                    double res = test_model(W, W1, test_data, test_label);
                    cout<<res<<endl;
                } else {
                    for(int j=0; j<W.rows(); j++) {
                        for(int k=0; k<W.cols(); k++)
                            W_temp[j*W.cols()+k] = W(j,k);
                    }

                    io->send_data(&W_temp[0],sizeof(unsigned long int)*W_temp.size());
                    io->flush();
                }
            }
        
    }



    cout<<"total time:"<<total_time<<"s!!"<<endl;
    cout<<"communication between servers:"<< COMM/1024/1024 << " MB" << endl;
    cout<<"communication with clients:"<<2.0 * COMM_client/1024/1024 << " MB" << endl;

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
