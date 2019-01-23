function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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



prediction = sigmoid(X*theta);
red= log(prediction);

valuered= red' * -y;

blue = log(1-prediction);
valueblue=blue' * (1-y);

value= valuered - valueblue;

resultunregulize = (1/m)*value;

theta(1)=0;

squaretheta = theta' * theta;
regul = (lambda/(2*m))*squaretheta;

J = resultunregulize + regul;

gauche = prediction - y;
unregulize  = (1/m) * X' * gauche;

#egularized gradient term as theta scaled by (lambda / m).

grad = unregulize + (lambda*theta)/m;



% =============================================================

end
