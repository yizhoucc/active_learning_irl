import numpy as np 
from scipy.optimize import minimize 
import scipy

dt=0.1
tau=0.5
tau_a=np.exp(-dt/tau)

A=np.array([[1.,dt],[0.,1-tau_a]])
B=np.array([[0.],[tau_a]])
Q=np.diag([1.,1])
R=np.ones((1,1))*0.01


A,B,Q,R


def lqg(A,B,Q,R):
    P = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
    K = np.matrix(-np.linalg.inv(R)*(B.T*P)*A)
    return K

def dynamic(x, u, A,B):
    x_next=A@x+B@u
    return x_next


x=np.ones((2,1))
u=np.ones((1,1))




x=np.array([[10],[0]])
u=np.ones((1,1))*0

xs,us=[],[]
t=0
while x[0]>1 and  t<88:
    u=lqg(A,B,Q,R)*x
    x=dynamic(x,u,A, B)
    xs.append(x)
    us.append(u)

from plot_ult import *
plt.plot([t[0,0] for t in xs])




import numpy as np

np.set_printoptions(precision=3,suppress=True)
 
# Optional Variables
max_linear_velocity = 2.0 # meters per second
max_angular_velocity = 1.5708 # radians per second
 
     
def state_space_model(A, state_t_minus_1, B, control_input_t_minus_1):

    # make sure control with in range
    control_input_t_minus_1[0] = np.clip(control_input_t_minus_1[0],
                                                                            -max_linear_velocity,
                                                                            max_linear_velocity)
    # control_input_t_minus_1[1] = np.clip(control_input_t_minus_1[1], -max_angular_velocity,max_angular_velocity)
    
    # prediction
    state_estimate_t = (A @ state_t_minus_1) + (B @ control_input_t_minus_1) 
             
    return state_estimate_t


def lqr(actual_state_x, desired_state_xf, Q, R, A, B):
    """
    Discrete-time linear quadratic regulator for a nonlinear system.
 
    Compute the optimal control inputs given a nonlinear system, cost matrices, 
    current state, and a final state.
     
    Compute the control variables that minimize the cumulative cost.
 
    Solve for P using the dynamic programming method.
 
    :param actual_state_x: The current state of the system 
        3x1 NumPy Array given the state is [x,y,yaw angle] --->
        [meters, meters, radians]
    :param desired_state_xf: The desired state of the system
        3x1 NumPy Array given the state is [x,y,yaw angle] --->
        [meters, meters, radians]   
    :param Q: The state cost matrix
        3x3 NumPy Array
    :param R: The input cost matrix
        2x2 NumPy Array
    :param dt: The size of the timestep in seconds -> float
 
    :return: u_star: Optimal action u for the current state 
        2x1 NumPy Array given the control input vector is
        [linear velocity of the car, angular velocity of the car]
        [meters per second, radians per second]
    """
    
    # We want the system to stabilize at desired_state_xf.
    x_error = actual_state_x - desired_state_xf
 
    # Solutions to discrete LQR problems are obtained using the dynamic 
    # programming method.
    # The optimal solution is obtained recursively, starting at the last 
    # timestep and working backwards.
    # You can play with this number
    N = 50
 
    # Create a list of N + 1 elements
    P = [None] * (N + 1)
     
    Qf = Q
 
    # LQR via Dynamic Programming
    P[N] = Qf
 
    # For i = N, ..., 1
    for i in range(N, 0, -1):
 
        # Discrete-time Algebraic Riccati equation to calculate the optimal 
        # state cost matrix
        P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
            R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)      
 
    # Create a list of N elements
    K = [None] * N
    u = [None] * N
 
    # For i = 0, ..., N - 1
    for i in range(N):
 
        # Calculate the optimal feedback gain K
        K[i] = -np.linalg.pinv(R + B.T @ P[i+1] @ B) @ B.T @ P[i+1] @ A
 
        u[i] = K[i] @ x_error
 
    # Optimal control input is u_star
    u_star = u[N-1]
 
    return u_star
 

def main():
    
    dt = 0.1
    tau=0.5
    tau_a=np.exp(-dt/tau)

     
    # Actual state
    # relative dist, relative angle, linear v, angular w
    # actual_state_x = np.array([10,0.7,0,0]) 
    actual_state_x = np.array([10,10]) 
 
    # Desired state [x,y,yaw angle]
    # desired_state_xf = np.array([0,0,0,0])  
    desired_state_xf = np.array([0,0.])  
     
    # A matrix
    # A = np.array([  [1.0,  dt],
                    # [ 0,  tau_a]])
    A=np.array([[1.,dt],[0.,1-tau_a]])
    B=np.array([[0.],[tau_a]])
    
 
    # R matrix
    # The control input cost matrix
    # Experiment with different R matrices
    # This matrix penalizes actuator effort (i.e. rotation of the 
    # motors on the wheels that drive the linear velocity and angular velocity).
    # The R matrix has the same number of rows as the number of control
    # inputs and same number of columns as the number of 
    # control inputs.
    # This matrix has positive values along the diagonal and 0s elsewhere.
    # We can target control inputs where we want low actuator effort 
    # by making the corresponding value of R large. 
    R = np.array([[0.01,   0],  # Penalty for linear velocity effort
                [  0, 0.01]]) # Penalty for angular velocity effort
 
    # Q matrix
    # The state cost matrix.
    # Experiment with different Q matrices.
    # Q helps us weigh the relative importance of each state in the 
    # state vector (X, Y, YAW ANGLE). 
    # Q is a square matrix that has the same number of rows as 
    # there are states.
    # Q penalizes bad performance.
    # Q has positive values along the diagonal and zeros elsewhere.
    # Q enables us to target states where we want low error by making the 
    # corresponding value of Q large.
    Q = np.array([[0.639, 0],  # Penalize X position error 
                [0, 1.0]]) # Penalize YAW ANGLE heading error 
                   
    # Launch the robot, and have it move to the desired goal destination
    for i in range(100):
        print(f'iteration = {i} seconds')
        print(f'Current State = {actual_state_x}')
        print(f'Desired State = {desired_state_xf}')
         
        state_error = actual_state_x - desired_state_xf
        state_error_magnitude = np.linalg.norm(state_error)     
        print(f'State Error Magnitude = {state_error_magnitude}')
         

        # LQR returns the optimal control input
        optimal_control_input = lqr(actual_state_x, 
                                    desired_state_xf, 
                                    Q, R, A, B) 
         
        print(f'Control Input = {optimal_control_input}')
                                     
         
        # We apply the optimal control to the robot
        # so we can get a new actual (estimated) state.
        actual_state_x = state_space_model(A, actual_state_x, B, 
                                        optimal_control_input)  
 
        # Stop as soon as we reach the goal
        # Feel free to change this threshold value.
        if state_error_magnitude < 0.1:
            print("\nGoal Has Been Reached Successfully!")
            break
             
        print()
 
# Entry point for the program
main()






A_k_minus_1 = np.array([[1.0,  0,   0],
                                                [  0,1.0,   0],
                                                [  0,  0, 1.0]])
 

process_noise_v_k_minus_1 = np.array([0.01,0.01,0.003])

Q_k = np.array([[1.0,   0,   0],
                                [  0, 1.0,   0],
                                [  0,   0, 1.0]])
                 
H_k = np.array([[1.0,  0,   0],
                                [  0,1.0,   0],
                                [  0,  0, 1.0]])
                         
R_k = np.array([[1.0,   0,    0],
                                [  0, 1.0,    0],
                                [  0,    0, 1.0]])  
                 

sensor_noise_w_k = np.array([0.07,0.07,0.04])
 

def ekf(z_k_observation_vector, state_estimate_k_minus_1, 
        control_vector_k_minus_1, P_k_minus_1, dk):

    state_estimate_k = A_k_minus_1 @ (
            state_estimate_k_minus_1) + (
            getB(state_estimate_k_minus_1[2],dk)) @ (
            control_vector_k_minus_1) + (
            process_noise_v_k_minus_1)
             
    print(f'State Estimate Before EKF={state_estimate_k}')
             
    # Predict the state covariance estimate based on the previous
    # covariance and some noise
    P_k = A_k_minus_1 @ P_k_minus_1 @ A_k_minus_1.T + (
            Q_k)
         
    ################### Update (Correct) ##########################
    # Calculate the difference between the actual sensor measurements
    # at time k minus what the measurement model predicted 
    # the sensor measurements would be for the current timestep k.
    measurement_residual_y_k = z_k_observation_vector - (
            (H_k @ state_estimate_k) + (
            sensor_noise_w_k))
 
    print(f'Observation={z_k_observation_vector}')
             
    # Calculate the measurement residual covariance
    S_k = H_k @ P_k @ H_k.T + R_k
         
    # Calculate the near-optimal Kalman gain
    # We use pseudoinverse since some of the matrices might be
    # non-square or singular.
    K_k = P_k @ H_k.T @ np.linalg.pinv(S_k)
         
    # Calculate an updated state estimate for time k
    state_estimate_k = state_estimate_k + (K_k @ measurement_residual_y_k)
     
    # Update the state covariance estimate for time k
    P_k = P_k - (K_k @ H_k @ P_k)
     
    # Print the best (near-optimal) estimate of the current state of the robot
    print(f'State Estimate After EKF={state_estimate_k}')
 
    # Return the updated state and covariance estimates
    return state_estimate_k, P_k
