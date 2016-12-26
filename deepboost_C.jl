include("./types.jl")
include("./tree.jl")
include("./boost.jl")
# using Juno
# using Gallium


# Train a deepboost model on the given examples, using
# numIter iterations (which not necessarily means numIter trees)
function Train(train_examples::Vector{Example}, model::Model, tree_depth::Int64,
        num_iter::Int64, beta::Float64, lambda::Float64, loss_type::String, verbose::Bool)
# model: Model, need return
# train_examples: pointer, since it is a Vector, change is in-place, no need to return
	# Train the model
	for iter in 1:num_iter
		(train_examples, model) = AddTreeToModel(train_examples, model, loss_type,
                                             beta, lambda, tree_depth)
    # println("successfully run for ", iter, "iterations.")
    #Juno.@step gcd(12, 4)
    # @enter gcd(5, 20)
    println(model[1].weight)
		if verbose
   		(error, avg_tree_size, num_trees) = EvaluateModel(train_examples, model)
			println("Iteration: ", iter, ", error: ", error,
			        ", avg tree size: ", avg_tree_size,
			        ", num trees: ", num_trees, "\n")

		end
	end

  return model
end


# Classify examples using model
function Predict(examples::Vector{Example}, model::Model)
	# TODO::initiate labels
	labels  = zeros(Int64, length(examples)) # Vector{Label::Int64}
	for i in 1:length(examples)
		labels[i] = ClassifyExampleModel(examples[i], model)
  end
	return labels
end


# Compute the error of model on examples. Also compute the number of trees in
# model and their average size.
function Evaluate(examples::Vector{Example}, model::Model)
# error:
# avg_tree_size
# num_trees
  (error, avg_tree_size, num_trees) = EvaluateModel(examples, model)
  return (error, avg_tree_size, num_trees)
end
