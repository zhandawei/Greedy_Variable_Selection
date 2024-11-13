clearvars;clc;close all;
% objective function
fun_name = 'Ellipsoid';
% number of variables
num_vari = 100;
% lower and upper bounds
lower_bound = -5.12*ones(1,num_vari);
upper_bound = 5.12*ones(1,num_vari);
% dimension of subspaces
sub_vari = 5;
% number of initial design points
num_initial = 200;
% number of maximum evaluations
max_evaluation = 1000;
% initial design
sample_x = lhsdesign(num_initial,num_vari,'criterion','maximin','iterations',1000).*(upper_bound-lower_bound)+lower_bound;
sample_y = feval(fun_name,sample_x);
evaluation =  size(sample_x,1);
iteration = 1;
[fmin,ind]= min(sample_y);
best_x = sample_x(ind,:);
fprintf('Gr-VS on %d-D %s, run: %d, iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,1,iteration-1,evaluation,fmin);
while evaluation < max_evaluation
    % train GP models
    GP_model = GP_train(sample_x,sample_y,lower_bound,upper_bound,1,0.01,100);
    infill_x  = best_x;
    remain_dim = 1:num_vari;
    % optimizes the sub-variables one by one using the greedy approach
    for kk = 1:sub_vari
        optimal_x = zeros(1,length(remain_dim));
        ECI = zeros(1,length(remain_dim));   
        % calcaute ECI values of remaining varibles
       for ii = 1:length(remain_dim)
            [optimal_x(ii),ECI(ii)]= Optimizer_GA(@(x)-Infill_ECI(x,GP_model,fmin,infill_x,remain_dim(ii)),1,lower_bound(remain_dim(ii)),upper_bound(remain_dim(ii)),10,20);
       end
       % find the max ECI value and its corresponding index value
        [max_ECI,index] = max(-ECI);
        % replace the variable corresponding to the index value 
        infill_x(remain_dim(index)) = optimal_x(index);
        % delete the index value corresponding to the max ECI  
        remain_dim(index) = [];
    end  
    % evaluate the new solution
    infill_y = feval(fun_name,infill_x);
    iteration = iteration + 1;
    % add the updated solution to the sample set
    sample_x = [sample_x;infill_x];
    sample_y = [sample_y;infill_y];
    [fmin,ind]= min(sample_y);
    best_x = sample_x(ind,:);
    evaluation = evaluation + size(infill_x,1);
    fprintf('Gr-VS on %d-D %s, run: %d, iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,1,iteration-1,evaluation,fmin); 
end


    


