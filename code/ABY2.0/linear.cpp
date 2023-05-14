#include "utils.hpp"

double total_time, online_time, offline_time;
double COMM, COMM_client, COMM1;

Time_Calculator mytime;

NetIO * io_server;
NetIO * io_client_alice;
NetIO * io_client_bob;
NetIO * io_client;

Mat reconstruct(Mat A, int party){
	Mat tempA(A.rows(), A.cols());
    if(party == ALICE) {
    	send_mat(A, io_server, COMM);
    	receive_mat(tempA, io_server, COMM);
    }
    else {
    	receive_mat(tempA, io_server, COMM);
    	send_mat(A, io_server, COMM);
    }
    return A + tempA;
}

void load_test_data(Mat& test_data, Mat& test_label){
	ifstream infile2( "../data/mnist_test.csv" );
	int i=0;
	
	//cout<<"load testing data.......\n";
	
	while(infile2){
		
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
			for(int j=0;j<D-1;j++){
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


double test_model(Mat& W0, Mat& W1, Mat& x, Mat& y){
	Mat y_,W;
	double temp1;
	long int temp2,temp3;
	
	W = W0+W1;
	y_ = x*W;
	
	int count = 0;
	
	for(int i=0;i<y.rows();i++){
		temp3 = (long int)y_(i);
		//temp3 = (temp3<<4);
		
		//if(temp3>conv<long int>(p)/2)
		//	temp3 = temp3-conv<long int>(p);
		
		temp1 = temp3/(double)pow(2,L+8);
		//temp1 = conv<long int>(y_[i][0])/(double)pow(2,L);
		temp2 = (long int)y(i);
		
		//if(temp2>conv<long int>(p)/2)
		//	temp2 = temp2-conv<long int>(p);
		
		if(temp1>0.5 && temp2 == 1){
			count++;
		}
		else if(temp1<0.5 && temp2 == 0){
			count++;
		}
	}	
	
	//cout << "rows = " << (int)y.rows() << endl;
	return count/(double)y.rows();
}

void sync_server() {
//	cerr << "start sync_server\n";
	int t1 = 666, t2;
	io_client->send_data(&t1, sizeof(int));
	io_client->flush();
	io_client->recv_data(&t2, sizeof(int));
//	cerr << "end sync_server\n";
}

void sync_between_servers(int party) {
	int syn1 = 666, syn2;
	if(party == ALICE) {
		io_server->send_data(&syn1, sizeof(int));
		io_server->flush();
		io_server->recv_data(&syn2, sizeof(int));
	}
	else if(party == BOB) {
		io_server->recv_data(&syn2, sizeof(int));
		io_server->send_data(&syn1, sizeof(int));
		io_server->flush();
	}
}

int main(int argc, char** argv){
	//setup connection
	int port, party;
	parse_party_and_port(argv, &party, &port);
	setup(party, io_server, io_client_alice, io_client_bob, io_client);

	vector<int> perm(N * Ep, 0);

	io_client->recv_data(&perm[0], sizeof(int) * perm.size());

	total_time = 0.0;
	offline_time = 0.0;
	online_time = 0.0;

    COMM = 0.0;
    COMM_client = 0.0;
    COMM1 = 0.0;
	

    int t1;

	Mat test_data(testN,D), test_label(testN,1);
	
	if(party == ALICE){
		load_test_data(test_data, test_label);
	}
	
	
	Mat W(D,1);
	W.setZero();
	
	Mat Dx(N, D), DW(D, 1), Dy(B, 1);
	Mat dx(N, D), dW(IT, D), dz(IT, B), dy(IT, B), dg(IT, D);
	Mat train_data(N,D), train_label(N,1);
	Mat dx_batch(B, D), dy_batch(B, 1), Dx_batch(B, D), Dz(B, 1), dW_batch(D, 1), y_batch(B, 1);
	
	sync_between_servers(party);
	sync_server();
	
	if(CLIENT_AIDED) t1 = mytime.start();
	
	receive_mat(dx, io_client, COMM1);
	receive_mat(dW, io_client, COMM1);
	receive_mat(dz, io_client, COMM1);
	receive_mat(dy, io_client, COMM1);
	receive_mat(dg, io_client, COMM1);
	if(CLIENT_AIDED){
		offline_time += mytime.end(t1);
		COMM_client += COMM1;
	}
	
	cerr << "triples received\n";
	
	t1 = mytime.start();
	receive_mat(train_data, io_client, COMM_client);
	receive_mat(train_label, io_client, COMM_client);
	Dx = reconstruct(train_data + dx, party);
	online_time += mytime.end(t1);
	
	
	sync_between_servers(party);
	cerr << "start training...\n";
	
	int start = 0;

	
	//start training
	for (int i = 0; i < IT; i++, start += B){
		
		t1 = mytime.start();
		
		next_batch(dx_batch, start, perm, dx);
		next_batch(Dx_batch, start, perm, Dx);
		next_batch(y_batch, start, perm, train_label);
		
		//cerr << "0...\n";
		dW_batch = (dW.row(i)).transpose();
		DW = reconstruct(W + dW_batch, party);
		Mat temp(B, 1), temp_g(D, 1);
		if(party == ALICE) temp = - Dx_batch * dW_batch - dx_batch * DW + (dz.row(i)).transpose();
		else temp = Dx_batch * DW - Dx_batch * dW_batch - dx_batch * DW + (dz.row(i)).transpose();
		
		temp = temp - y_batch * (unsigned long int)(1 << (L + 8));
		
		//cerr << "1...\n";
		dy_batch = (dy.row(i)).transpose();
		Dy = reconstruct(temp + dy_batch, party);
		
		if(party == ALICE) temp_g = - Dx_batch.transpose() * dy_batch - dx_batch.transpose() * Dy + (dg.row(i)).transpose();
		else temp_g =  Dx_batch.transpose() * Dy - Dx_batch.transpose() * dy_batch - dx_batch.transpose() * Dy + (dg.row(i)).transpose();
		
		if(DEBUG) {
			Mat now_x = Dx_batch - reconstruct(dx_batch, party);
			Mat now_W = reconstruct(W, party);
			Mat now_z = now_x * now_W - reconstruct(y_batch, party) * (unsigned long int)(1 << (L + 8));
			Mat temp_z = reconstruct(temp, party);
			Mat now_g = now_x.transpose() * now_z;
			Mat temp_temp_g = reconstruct(temp_g, party);
			for (int j = 0; j < B; ++j) if(now_z(j, 0) != temp_z(j, 0)) {
				cerr << "IT" << i << " z" << j << ": " << now_z(j, 0) << " " << temp_z(j, 0) << "\n";
				assert(0);
			}
			for (int j = 0; j < D; ++j) if(now_g(j, 0) != temp_temp_g(j, 0)) {
				cerr << "IT" << i << " g" << j << ": " << now_g(j, 0) << " " << temp_temp_g(j, 0) << "\n";
				assert(0);
			}
		}
		
		//cerr << "2...\n";
		for (int j = 0; j < D; ++j) 
			W(j, 0) = (long int) W(j, 0) - (long int)temp_g(j, 0) / ((long int)(1<<23) * B);
		
		online_time += mytime.end(t1);
		
		if(EVALUATE && i%5 == 0){
			vector<long int> W_temp(W.cols()*W.rows());
			int tempi = 10;
			Mat W1(D,1);
			
			if(party==ALICE){
			
				io_server->send_data(&tempi,4);
				io_server->flush();
				io_server->recv_data(&W_temp[0],sizeof(unsigned long int)*W_temp.size());
				
				
				for(int j=0;j<W1.rows();j++){
					for(int k=0;k<W1.cols();k++)
						W1(j,k) = W_temp[j*W1.cols()+k];
				}
				
				
				double res = test_model(W,W1,test_data, test_label);
				cout<<res<<endl;
			}
			else{
				for(int j=0;j<W.rows();j++){
					for(int k=0;k<W.cols();k++)
						W_temp[j*W.cols()+k] = W(j,k);
				}
				
				io_server->recv_data(&tempi,4);
				io_server->send_data(&W_temp[0],sizeof(unsigned long int)*W_temp.size());
				io_server->flush();
				
			}
			
		}
		
	}
	
	total_time = online_time + offline_time;
	
	cout << "total time:" << total_time << "s" << endl;
	cout << "online time:" << online_time << "s" << endl;
	cout << "offline time:" << offline_time << "s" << endl;
	cout << "comm between servers: " << (COMM / 1024 / 1024) << "MB" << endl;
	cout << "comm with clients: " << (COMM_client / 512 / 1024) << "MB" << endl;
	cout << "\n";
	
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
