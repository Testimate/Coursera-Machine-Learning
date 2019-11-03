function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta


% hTheta = sigmoid(X * theta); %%% hTheta: m*1. theta: p*1 

% J = -1/m * (y' * log(hTheta) + (1-y)' * log(1-hTheta) ); %%% SIGMA of a multiplication: vectorize
% grad(1) = 1/m * X(:,1)' * (hTheta - y); % 1*m * m*1
% grad(2) = 1/m * X(:,2)' * (hTheta - y);
% grad(3) = 1/m * X(:,3)' * (hTheta - y);
%%% vectorization expression,  see reading: Simplified Cost Function and Gradient Descent

%% The above code, though is correct, has problem with general p other than 3.


%% Nov. 2nd 2019

hTheta = sigmoid(X * theta); %%% hTheta: m*1. theta: p*1 

J = -1/m * (y' * log(hTheta) + (1-y)' * log(1-hTheta) ); 
 
grad = 1/m * X' * (hTheta - y); %%% p * m * m * 1 


% =============================================================

end
