function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m, 1) X]; %% 5000 * 401  add a column with 1s to the left as intercept

%% #col in our current theta matrix = #nodes on our current layer
%% #row in our current theta matrix = #nodes on our next layer
%% Theta1: 25 * 401
z2 = Theta1 * X'; %% a1, the input layer should contain 400 nodes + 1 
% where 400 is the column number in X, so a1 = X', cols of Theta is 401.


%% 25 * 401 * 401 * 5000 = 25 * 5000
a2 = sigmoid(z2); %% sigmoid not change dimension
a2 = [ones(1,size(X,1));a2]; % 26 * 5000, add a row with 1s at top as bias unit
z3 = Theta2 * a2; % 10 * 26 * 26 * 5000 = 10 * 5000
a3 = sigmoid(z3); % Output layer
[~,p]  = max(a3,[],1); % a3 5000 columns; use max(A, [], 1) to obtain the max for each column







% =========================================================================


end
