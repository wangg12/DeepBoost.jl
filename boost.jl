# workspace()

# include("./types.jl")
# include("./tree.jl")


# Return the optimal weight to add to a tree that will maximally decrease the
# objective. line 12 ~16 in algorithm
function ComputeEta(wgtd_error::Float64, tree_size::Float64, alpha::Float64,
                    beta::Float64, lambda::Float64,
                    the_normalizer::Float64, num_examples::Int64, num_features::Int64)
  wgtd_error = max(wgtd_error, kTolerance)  # Helps with division by zero.
  error_term = (1. - wgtd_error) * exp(alpha) - wgtd_error * exp(-alpha)
  complexity_penalty = ComplexityPenalty(Int64(tree_size), beta, lambda,
                                  the_normalizer, num_examples, num_features)
  ratio = complexity_penalty / wgtd_error
  eta = 0.
  if abs(error_term) <= 2.0 * complexity_penalty
    eta = -alpha;
  elseif error_term > 2.0 * complexity_penalty
    eta = log(-ratio + sqrt(ratio * ratio + (1 - wgtd_error)/wgtd_error))
  else
    eta = log(ratio + sqrt(ratio * ratio + (1 - wgtd_error)/wgtd_error))
  end
  return eta
end

# Either add a new tree to model or update the weight of an existing tree in
# model. The tree and weight are selected via approximate coordinate descent on
# the objective, where the "approximate" indicates that we do not search all
# trees but instead grow trees greedily.
function AddTreeToModel(examples::Vector{Example}, model::Model, loss_type::String,
                        beta::Float64, lambda::Float64, tree_depth::Int64)
# NB(wg): we return (examples, model)
# TODO(usyed): examples is passed by non-const reference because the example
# weights need to be changed. This is bad style.
# NB(wg):weights of examples  should be changed, the Vector should do this, need check
  # Initialize normalizer
  normalizer = 0. # NB:static float, why static?
  if isempty(model)
    if loss_type == "e"
      normalizer = exp(1) * length(examples)
    elseif loss_type == "l"
      normalizer = length(examples) / (log(2) * (1 + exp(-1)))
    end
  end
  # by yy: for distribution is initialized with uniform distribution
  # if isempty(model)
  #   normlizer = 1.
  # end

  # InitializeTreeData(examples, normalizer) # NB(need change)
  num_features = length(examples[1].values) # length of feature, num_examples should >= 1
  num_examples = length(examples)
  the_normalizer = normalizer


  best_old_tree_idx = -1
  best_wgtd_error = wgtd_error = gradient = best_gradient = 0.0

  # Find best old tree
  old_tree_is_best = false
  for i in 1:length(model)
    alpha = model[i].weight
    if abs(alpha) < kTolerance
      continue  # Skip zeroed-out weights.
    end
    old_tree = model[i].tree
    wgtd_error = EvaluateTreeWgtd(examples, old_tree)
    sign_edge = (wgtd_error >= 0.5) ? 1 : -1
    gradient = Gradient(wgtd_error, length(old_tree), alpha, sign_edge, beta, lambda,
                        the_normalizer, num_examples, num_features)
    if abs(gradient) >= abs(best_gradient)
      best_gradient = gradient
      best_wgtd_error = wgtd_error
      best_old_tree_idx = i
      old_tree_is_best = true
    end
  end
  # println("successfully find the best old tree!")

  # Find best new tree
  new_tree = TrainTree(examples, beta, lambda, tree_depth, normalizer)
  wgtd_error = EvaluateTreeWgtd(examples, new_tree)
  gradient = Gradient(wgtd_error, length(new_tree), 0., -1, beta, lambda,
                      the_normalizer, num_examples, num_features)
  # println("Complete calculate the gradient !")
  if isempty(model) || abs(gradient) > abs(best_gradient)
    best_gradient = gradient
    best_wgtd_error = wgtd_error
    old_tree_is_best = false
  end
  # println("successfully find the best new tree!")

  # Update model weights
  alpha = 0.
  tree = Tree() # vector of Nodes
  if old_tree_is_best
    alpha = model[best_old_tree_idx].weight
    tree = model[best_old_tree_idx].tree
  else
    alpha = 0.
    tree = new_tree
  end
  eta = ComputeEta(best_wgtd_error, Float64(length(tree)), alpha, beta, lambda,
                  the_normalizer, num_examples, num_features)
  if old_tree_is_best
    println("before update, weight is ", model[best_old_tree_idx].weight)
    model[best_old_tree_idx].weight += eta
    println("after update, weight is ", model[best_old_tree_idx].weight)
  else
    push!(model, classifier(eta, new_tree))
    println("the weight of new classifier is ", eta)
  end

  # Update examples weights and compute normalizer
  old_normalizer = normalizer
  normalizer = 0.
  for example in examples
    u = eta * example.label * ClassifyExampleTree(example, tree); # ??
    if loss_type == "e"
      example.weight *= exp(-u)
    elseif loss_type == "l"
      z = (1 - log(2) * example.weight * old_normalizer) /
                      (log(2) * example.weight * old_normalizer)
      example.weight = 1 / (log(2) * (1 + z * exp(u)))
    end
    normalizer += example.weight
  end

  # Renormalize example weights
  # TODO(usyed): Two loops is inefficient.
  for example in examples
    example.weight /= normalizer
  end
  return (examples, model)
end


# Classify example with model.
function ClassifyExampleModel(example::Example, model::Model)
  score = 0.
  for wgtd_tree in model
    score += wgtd_tree.weight * ClassifyExampleTree(example, wgtd_tree.tree)
  end
  if score < 0.
    return -1
  else
    return 1
  end
end


# Compute the error of model on examples. Also compute the number of trees in
# model and their average size.
# return (error, avg_tree_size, num_trees)
function EvaluateModel(examples::Vector{Example}, model::Model)
# error: pointer float
# avg_tree_size: pointer float
# num_trees: pointer int
  incorrect = 0.
  for example in examples
    if example.label != ClassifyExampleModel(example, model)
      incorrect += 1
    end
  end
  num_trees = 0.
  sum_tree_size = 0.
  println("the size of model is ", length(model))
  for  wgtd_tree in model
    if abs(wgtd_tree.weight) >= kTolerance
      num_trees += 1
      sum_tree_size += length(wgtd_tree.tree)
    end
    # num_trees += 1
    # sum_tree_size += length(wgtd_tree.tree)
  end
  error = incorrect / length(examples)
  avg_tree_size = sum_tree_size / num_trees
  return (error, avg_tree_size, num_trees)
end
