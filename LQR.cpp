// LQR.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

using namespace std;
using namespace Eigen;

// Set Global Variables
double max_linear_velocity = 3.0;
double max_angular_velocity = M_PI / 2;
double dt = 1.0;	// smapling time

MatrixXd getB(double yaw, double dt);
MatrixXd stateSpaceModel(MatrixXd A, MatrixXd x_revious, MatrixXd B, MatrixXd u_previous);
MatrixXd LQR(MatrixXd x_actual, MatrixXd x_f, MatrixXd Q, MatrixXd R, MatrixXd A, MatrixXd B, double dt);

int main()
{
	MatrixXd actual_state(3, 1);
	MatrixXd desired_state(3, 1);
	MatrixXd state_error(3,1);
	MatrixXd optimal_input;
	MatrixXd B(3, 2);
	actual_state << 0.0, 0.0, 0.0;			// Robot Starts at point: (0, 0) with 0 heading
	desired_state << 2.0, 2.0, M_PI / 2;	// Robot Desired Position: point (2, 2) facing North

	double state_error_magnitude;

	// State Space Model
	MatrixXd A = MatrixXd::Identity(3, 3);

	// Control Input Cost Matrix
	MatrixXd R(2, 2);
	R << 0.01, 0.0,
		0.0, 0.01;
	// State Cost Matrix
	MatrixXd Q(3, 3);
	Q << 0.639, 0.000, 0.000,
		0.000, 1.000, 0.000,
		0.000, 0.000, 1.000;

	// Launch the Robot and Controller
	for (int i = 0; i < 100; i++) {
		cout << "iteration: " << i << "seconds\n";
		cout << "Current State:\n" << actual_state << endl;
		cout << "Desired State:\n" << desired_state << endl;

		state_error = actual_state - desired_state;
		state_error_magnitude = state_error.norm();
		cout << "State Error (Normal): " << state_error_magnitude << endl;

		B = getB(actual_state(2, 0), dt);
		optimal_input = LQR(actual_state, desired_state, Q, R, A, B, dt);
		cout << "Control Input: \n" << optimal_input << endl;

		actual_state = stateSpaceModel(A, actual_state, B, optimal_input);

		if (state_error_magnitude < 0.01) {
			cout << "The Goal Position has been reached!" << endl;
			break;
		}

		cout << "--------------------------\n";
	}

	system("pause");
    return 0;
}


MatrixXd getB(double yaw, double dt) {
	MatrixXd B(3,2);
	B << cos(yaw) * dt, 0.0,
		sin(yaw) * dt, 0.0,
		0.0, dt;
	return B;
}

MatrixXd stateSpaceModel(MatrixXd A, MatrixXd x_previous, MatrixXd B, MatrixXd u_previous) {
	MatrixXd state_estimate(x_previous.rows(), x_previous.cols());
	// Clip linear Velocity
	if (u_previous(0, 0) > max_linear_velocity)
		u_previous(0, 0) = max_linear_velocity;
	else if (u_previous(0, 0) < -max_linear_velocity)
		u_previous(0, 0) = -max_linear_velocity;
	
	// Clip Angular Velocity
	if (u_previous(1, 0) > max_angular_velocity)
		u_previous(1, 0) = max_angular_velocity;
	else if (u_previous(1, 0) < -max_angular_velocity)
		u_previous(1, 0) = -max_angular_velocity;

	state_estimate = A * x_previous + B * u_previous;
	return state_estimate;
}
MatrixXd LQR(MatrixXd x_actual, MatrixXd x_f, MatrixXd Q, MatrixXd R, MatrixXd A, MatrixXd B, double dt) {
	MatrixXd error = x_actual - x_f;

	const int N = 50; // algorithm parameter
	MatrixXd* P = new MatrixXd[N + 1];
	MatrixXd* K = new MatrixXd[N];		// Optimal Feedback Gain
	MatrixXd* u = new MatrixXd[N];		// Controller Output

	MatrixXd QF = Q;

	// Dynamic Programming
	P[N] = QF;
	for (int i = N; i > 0; i--) {
		//cout << i << endl;
		P[i - 1] = Q + A.transpose() * P[i] * A - (A.transpose() * P[i] * B) * 
			(R + B.transpose() * P[i] * B).inverse() * (B.transpose() * P[i] * A);
	}
	//cout << "okay";
	for (int i = 0; i < N; i++) {
		K[i] = -(R + B.transpose() * P[i+1] * B).inverse() * B.transpose() * P[i+1] * A;
		u[i] = K[i] * error;
	}
	//cout << "okay\n";
	MatrixXd ans = u[N - 1];
	delete[] P;
	delete[] K;
	delete[] u;
	return ans;
}