function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h = X*theta;

n = length(theta);

SUMATORIA1 = ((h-y).^2)'*ones(m,1);

if n == 1,
  SUMATORIA2 = (theta.^2)'*0;
else,
  SUMATORIA2 = (theta.^2)'*[0; ones(n-1,1)];
end

J = 1/(2*m) * SUMATORIA1 + lambda/(2*m) * SUMATORIA2;

SUMATORIA3 = X'*(h-y);

aux_theta = theta;

aux_theta(1) = 0;

grad = 1/m * SUMATORIA3 + lambda/m * aux_theta;


% =========================================================================

grad = grad(:);

end
