function [J, grad] = nnCostFunction(nn_params, ...  %%% ... line continuation?
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
  %%% corresponds to lecture 9 Neural Networks: Learning Backpropogation
  %%% intuition --> Implementation note

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); % 25 * 401
Theta2_grad = zeros(size(Theta2)); % 10 * 26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%


%%% K = 10; num_labels

%%% Part 1


X_add1 = [ones(m,1) X];

%%% old
activation2 = sigmoid(X_add1 * Theta1');  %  Old: 5000 * 401 * 401 * 25
activation2 = [ones(m,1) activation2]; %  5000 * 26
Htheta = sigmoid(activation2 * Theta2'); % 5000 * 10, suit Htheta(i,k) 


%%%%%% recode y
%%% Previous coding: 
% y_new = zeros(m,num_labels);
% for i = 1 : m
% y_new(i,y(i)) = 1;
% end
% y_new(i,k)  % y_k^(i)  scaler
% Htheta(i,k) % h_{\theta}(X^(i))_k

%%% an equivalent but much simpler way to achieve that, according to course
%%% lab resource
%%% https://www.coursera.org/learn/machine-learning/resources/Uuxg6

%% Nov 19:  y_matrix(i,k) returns the binary True of False classfication response 
%% if the ith obervation belongs to the kth digit class, y^{(i)}_j = 1, otherwise = 0. 
eyebase = eye(num_labels);
y_matrix = eyebase(y,:); 
% eye(num_labels)(y,:) would return a warning


%%% Cost function J(theta) 
% lecture 9 Cost function
% main part without regularization
for i = 1 : m
    for k = 1 : num_labels
        J = J + ( - y_matrix(i,k) * log(Htheta(i,k)) - (1 - y_matrix(i,k)) * log(1 - Htheta(i,k)));
    end
end

J = 1/m * J;  

% -------------------------------------------------------------

%%% ex4 1.4 Regularized Cost Function

% Testimate: According to the ex4 lab, we can assume the neural network has 3
% layers (L = 3), i.e., no need to generalize this for now. 

% L = 3, the outer sum^2_1
regTerm_Theta1 = 0;
regTerm_Theta2 = 0;

for j = 1 : hidden_layer_size
    for k = 1 : input_layer_size
        regTerm_Theta1 = regTerm_Theta1 + Theta1(j,k+1)^2;
    end
end

for j = 1 : num_labels
    for k = 1 : hidden_layer_size
        regTerm_Theta2 = regTerm_Theta2 + Theta2(j,k+1)^2;
    end
end

J = J + lambda/(2*m) * (regTerm_Theta1 + regTerm_Theta2);


% Note that you should not be regularizing the terms that correspond to the bias. 
% For the matrices Theta1 and Theta2, this corresponds to the first column of each matrix. 


% -------------------------------------------------------------


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.


%%% Lab pre-step

delta_3 = zeros(num_labels, 1); % 10 * 1

BigDelta_2 = zeros(size(Theta2_grad));
BigDelta_1 = zeros(size(Theta1_grad));

% for loop for step 1 to 4
for t = 1 : m
    for k = 1 : num_labels

        %%% Lab step 1:
        % denote a^{(l)} (the lth layer) as a_l. (al normally used as elemental 'fen liang') 

        a_1 = X(t,:)'; %  column vector
        a_1 = [1;a_1]; % 401 * 1
        z_2 = Theta1 * a_1; % 25 * 401 * 401 * 1 = 25 * 1
        a_2 = sigmoid(z_2); 
        a_2 = [1;a_2]; % 26 * 1
        z_3 = Theta2 * a_2; % 10 * 26 * 26 * 1 = 10 * 1
        a_3 = sigmoid(z_3); % 10 * 1


        %%% Lab step 2:               
        delta_3(k) = a_3(k) - y_matrix(t,k); 
    end
    
        %%% Lab step 3:
        % delta_2 = zeros(hidden_layer_size,1);        
        %% 1st try, a little different than lab note p.9 in terms of notation (ignore bias unit in Theta instead of delta)      
        delta_2 = Theta2(:,2:end)' * delta_3 .* sigmoidGradient(z_2);  % 25 * 10 * 10 * 1 .* 25 * 1
        % ignoring the Theta2 bias units

  
        %%% Lab step 4:

        BigDelta_2 = BigDelta_2 + delta_3 * a_2';  % 10 * 1 * 1 * 26 = 10 * 26  num_labels * hidden_layer_size
        BigDelta_1 = BigDelta_1 + delta_2 * a_1';  % 25 * 1 * 1 * 401 = 25 * 401  hidden_layer_size * input_layer_size
    
end


% -------------------------------------------------------------


%%% Lab step 5:

Theta1_grad = BigDelta_1 / m;

Theta2_grad = BigDelta_2 / m;



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%%% Lecture 8 the euqations before Model Representation II may help
Theta1_grad = Theta1_grad + [zeros(hidden_layer_size,1) lambda/m * Theta1(:,2:end)];

Theta2_grad = Theta2_grad + [zeros(num_labels,1) lambda/m * Theta2(:,2:end)];


% =========================================================================


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
