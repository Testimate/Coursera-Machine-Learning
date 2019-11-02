function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    
    %$$$$$$$$$%%%  This is the Wrong Answer: 
    
    %%% theta(1) = theta(1) - alpha /m * sum(X * theta - y); 
    %%% theta(2) = theta(2) - alpha /m * sum((X * theta - y) .* X(:,2));

    %%% The reason this is wrong is that it fails to update 2-vector theta simultaneously
    %%% Which do not mean that theta0 and theta1 have to be updated at the same time 
    %%% but in that the iteration expression of update for both theta0 and theta1 depend on theta, which should not be changed after updating theta(1):
 
   
    
    %% This is the correct way. Note that this way is 'simultaneously'
    
    %% sums of m products! additional pair of () to prioritize .* over sum
     theta0 = theta(1) - alpha /m * sum ((X * theta - y).* X(:,1)); 
     theta1 = theta(2) - alpha /m * sum ((X * theta - y).* X(:,2));
     theta = [theta0; theta1];

    %% Of course, the optimal way shall be
    % theta = theta - alpha /m * sum (X * theta - y)*[1 1].* X;

    % ============================================================

    % Save the cost J in every iteration,     
    J_history(iter) = computeCost(X, y, theta);

end

end
