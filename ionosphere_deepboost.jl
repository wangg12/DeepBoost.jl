workspace()

include("./deepboost_C.jl")

cd(dirname(@__FILE__()))

using DataFrames
using MLBase

srand(2333)
ionosphere = readtable("../datasets/ionoSphere/ionosphere.data.csv",
                      separator = ',', header = false)

ds = ionosphere
labels = unique(ds[:,length(ds)]) # length == 2
# convert label to (1, -1)
ds_nrow = nrow(ds)
ds[:,length(ds)] = Vector{Int64}(map((x) -> x==labels[1] ? 1:-1, ds[:,length(ds)]))

# convert the dataframe to Example type
dataset = Vector{Example}()
for i_example = 1:ds_nrow
  value = Vector()
  for i_feature = 1:length(ds)-1
    push!(value,Float64(ds[i_example,i_feature]))
  end
  label = ds[i_example,length(ds)]
  weight = 1./ds_nrow
  example = Example(value, label, weight)
  push!(dataset, example)
end

# 10-fold cross-validation: partite data to 10 folds, each fold have equal p/n samples
ds_folds = collect(MLBase.StratifiedRandomSub(ds[length(ds)], ceil(ds_nrow/10), 10))

# cross-validation test:
# first fold - test fold
# second fold - validation fold
# rest folds - training fold
test_set = dataset[ds_folds[1]]
val_set = dataset[ds_folds[2]]
train_index = Int64[]
for i = 3:10
  train_index = append!(train_index,ds_folds[i])
end
train_set = dataset[train_index]

# parameter setting
beta_vals = Float64[2^(-6.), 2^(-5.), 2^(-4.), 2^(-3.), 2(-1.), 1.]
lambda_vals = Float64[0.0001, 0.005, 0.01, 0.05, 0.1, 0.5]
# model selection by cross-validation
best_val_err = 1
best_lambda = 0
best_beta = 0
best_dpb_model = Model()

beta = 1.0
lambda = 0.5
dpb_model = Train(train_set, Model(), 5, 20, beta, lambda, "e", true)

# for beta in beta_vals
#   for lambda in lambda_vals
#     # train the model
#     println("beta:", beta, ", lambda: ", lambda, ". Training...")
#     dpb_model = Train(train_set, Model(), 5,
#             100, beta, lambda, "l", true)
#     # Evaluate the model on validation set
#     val_err, avg_tree_size, num_trees = Evaluate(val_set, dpb_model)
#     # record the parameters and model of best performance on validation set
#     if val_err < best_val_err
#       best_val_err = val_err
#       best_lambda = lambda
#       best_beta = beta
#       best_dpb_model = dpb_model
#     end
#   end
# end
# # caculate the error on test set
# dpb_test_err, avg_tree_size, num_trees = Evaluate(test_set, best_dpb_model)
