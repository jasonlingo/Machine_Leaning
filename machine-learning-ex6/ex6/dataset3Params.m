function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
steps = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];
bestC = 0.01;
bestSigma = 0.01;
model = svmTrain(Xval, yval, bestC, @(x1, x2) gaussianKernel(x1, x2, bestSigma));
predictions = svmPredict(model, Xval);
bestMean = mean(double(predictions ~= yval));

for i = steps
    for j = steps
       model = svmTrain(X, y, i, @(x1, x2) gaussianKernel(x1, x2, j));
       predictions = svmPredict(model, Xval);
       err = mean(double(predictions ~= yval));
       if (err < bestMean)
           bestMean = err;
           bestC = i;
           bestSigma = j;
       end
    end
end

C = bestC;
sigma = bestSigma;






% =========================================================================

end
