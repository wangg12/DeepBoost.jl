# workspace()

# include("./types.jl")
#  Global variables are bad style.
# g_num_features = 0 # Int64
# g_num_examples = 0 # Int64
# g_the_normalizer = 0.0  # Float64
# g_is_initialized = false
#
# # Initialize some global variables.
# function InitializeTreeData(examples::Vector{Example}, normalizer::Float64)
# # leave it here, but avoid using it
#   global g_num_examples = length(examples)
#   # length of feature, num_examples should >= 1
#   global g_num_features = length(examples[1].values)
#   global g_the_normalizer = normalizer # the global normalizer
#   global g_is_initialized = true # globally initialized
#   return (g_num_examples, g_num_features, g_the_normalizer, g_is_initialized)
# end


# Return root node for a tree.
function MakeRootNode(examples::Vector{Example})
  # return Node root
  root = Node()
  root.examples = examples
  root.positive_weight = root.negative_weight = 0
  for example in examples
    if example.label == 1
      root.positive_weight += example.weight
    else   # label == -1
      root.negative_weight += example.weight
    end
  end
  root.leaf = true
  root.depth = 0
  return root
end


# Return a tree trained on examples.
function TrainTree(examples::Vector{Example}, beta::Float64, lambda::Float64,
                  tree_depth::Int64, normalizer::Float64)
  tree = Tree() # vector of Nodes
  push!(tree, MakeRootNode(examples))
  node_id = Int64(1) # julia starts from 1
  # println("length(tree) is ", length(tree))
  # for node_id in 1:length(tree)
  while node_id <= length(tree) # by yy: for the previous range is static
    # reference: any problems?
    # fine: need to be updated in the end, by function return
    node = tree[node_id]
    best_split_feature = Int64(0)
    best_split_value = 0.
    best_delta_gradient = 0.

    # following three are global in cpp, we need to calculate here
    num_features = length(examples[1].values) # length of feature, num_examples should >= 1
    num_examples = length(examples) # only used in ComplexityPenalty
    the_normalizer = normalizer # the normalizer, only used in ComplexityPenalty

    # search for the best split feature and the best split value
    for split_feature in 1:num_features
      value_to_weights = MakeValueToWeightsMap(node, split_feature)
      # this is a little different from the cpp code,
      # since cpp achieves changes of multiple by pointers in the input params/args
      (split_value, delta_gradient) = BestSplitValue(value_to_weights, node,
                                          length(tree), beta, lambda,
                                          the_normalizer,
                                          num_examples, num_features)
      if delta_gradient > best_delta_gradient + kTolerance
        best_delta_gradient = delta_gradient
        best_split_feature = split_feature
        best_split_value = split_value
      end
    end
    # add nodes to the tree if the depth doesn't exceed the limit
    # and the best delta gradient is not zero
    if (node.depth < tree_depth) && (best_delta_gradient > kTolerance)
      node, tree = MakeChildNodes(best_split_feature, best_split_value, node, tree)
    end
    # println("Run for ", node_id, " iterations to add the nodes.")
    node_id += 1
  end # end of train a new tree
  return tree
end


# Given a value-to-weights map for a feature (constructed by MakeValueToWeightsMap()),
# determine the best split value for the feature and
# the improvement in the gradient of the objective if we split on that value.
# Note that delta_gradient <= 0 indicates that we should not split on this feature.
function BestSplitValue(value_to_weights::Dict{Float64, Vector{Float64}},
                    node::Node, tree_size::Int64, beta::Float64, lambda::Float64,
                    the_normalizer::Float64, num_examples::Int64, num_features::Int64)
# used in TrainTree
# split_value: float pointer, no need to be in the input params
# delta_gradient: float pointer, no need to be in input params
# in julia, there is no pointer, so we return them to store the changed values
  split_value = 0. # by yy: split_value should be initialized, not sure if 0 is
                   # reasonable
  delta_gradient = 0.
  left_positive_weight = 0.
  left_negative_weight = 0.
  right_positive_weight = node.positive_weight
  right_negative_weight = node.negative_weight

  old_error = min(left_positive_weight + right_positive_weight,
                  left_negative_weight + right_negative_weight)
  old_gradient = Gradient(old_error, tree_size, 0., -1, beta, lambda,
                          the_normalizer, num_examples, num_features)
  for elem in value_to_weights # elem now is (key, value) Tuple
    left_positive_weight  += elem[2][1]
    right_positive_weight -= elem[2][1]
    left_negative_weight  += elem[2][2]
    right_negative_weight -= elem[2][2]

    new_error = min(left_positive_weight, left_negative_weight) +
                min(right_positive_weight, right_negative_weight)
    new_gradient = Gradient(new_error, tree_size + 2, 0., -1, beta, lambda,
                            the_normalizer, num_examples, num_features)
    if abs(new_gradient) - abs(old_gradient) > delta_gradient + kTolerance
      delta_gradient = abs(new_gradient) - abs(old_gradient)
      split_value = elem[1]
    end
  end
  return (split_value, delta_gradient)
end



# Return the (sub)gradient of the objective with respect to a tree.
function Gradient(wgtd_error::Float64, tree_size::Int64, alpha::Float64,
                  sign_edge::Int64, beta::Float64, lambda::Float64,
                  the_normalizer::Float64, num_examples::Int64,
                  num_features::Int64)
  # TODO(usyed): Can we make some mild assumptions and get rid of sign_edge?
  # const float
  complexity_penalty = ComplexityPenalty(tree_size, beta, lambda,
                                         the_normalizer,
                                         num_examples, num_features)
  edge = wgtd_error - 0.5 # const float
  sign_alpha = (alpha >= 0) ? 1 : -1 # const int
  if abs(alpha) > kTolerance
    return edge + sign_alpha * complexity_penalty
  elseif abs(edge) <= complexity_penalty
    return 0
  else
    return edge - sign_edge * complexity_penalty
  end
end


# Return complexity penalty.
function ComplexityPenalty(tree_size::Int64, beta::Float64, lambda::Float64,
                  the_normalizer::Float64, num_examples::Int64, num_features::Int64)
# we need to handle these global variables
# the_normalizer
# num_examples
# num_features
  rademacher = sqrt((((2. * tree_size+1.) * (log(num_features+2.)/log(2.))*log(num_examples)) / num_examples))
  return ((lambda*rademacher+beta)*num_examples) / (2*the_normalizer)
end





# Return a map from each value of feature to a pair of weights. The first
# weight in the pair is the total weight of positive examples at node that have
# that value for feature, and the second weight in the pair is the total weight
# of negative examples at node that have that value for feature. This map is
# used to determine the best split feature/value.
# map<Value::Float64, pair<Weight::Float64, Weight::Float64>>
function MakeValueToWeightsMap(node::Node, feature::Int64)
  # map<Value, pair<Weight, Weight>>
  # modified here by yy: for tuples are immutable
  value_to_weights = Dict{Float64, Vector{Float64}}()
  for example in node.examples
    if ~haskey(value_to_weights, example.values[feature])
      value_to_weights[example.values[feature]] = Vector{Float64}([0.,0.])
    end
    if example.label == 1
      value_to_weights[example.values[feature]][1] += example.weight
    else   # label = -1
      value_to_weights[example.values[feature]][2] += example.weight
    end
  end
  return value_to_weights
end



# Make child nodes using split feature/value and add them to the tree. Also
# update info in the parent node, like child pointers.
function MakeChildNodes(split_feature::Int64, split_value::Float64, parent::Node,
                    tree::Tree)
# parent: pointer Node
# tree: pointer Tree
  parent.split_feature = split_feature
  parent.split_value = split_value
  parent.leaf = false
  # println(tree[1].leaf) # To check whether the tree has been modified
  # Node left_child, right_child
  left_child = Node()
  right_child = Node()
  # Initialize the variables in child nodes
  left_child.examples = Array{Example, 1}()
  right_child.examples = Array{Example, 1}()
  left_child.depth = right_child.depth = parent.depth + 1
  left_child.leaf = right_child.leaf = true
  left_child.positive_weight = 0.
  left_child.negative_weight = 0.
  right_child.positive_weight = 0.
  right_child.negative_weight = 0.

  for example in parent.examples
    # Node* child, how to handle this?
    child = Node()
    if example.values[split_feature] <= split_value
      child = left_child
    else
      child = right_child
    end
    # TODO(usyed): Moving examples around is inefficient.
    push!(child.examples, example)
    if example.label == 1
      child.positive_weight += example.weight
    else #   label == -1
      child.negative_weight += example.weight
    end
  end
  # update the info of parent
  # by yy: modified to adapt to julia style
  parent.left_child_id = length(tree) + 1
  parent.right_child_id = length(tree) + 2
  push!(tree, left_child)
  push!(tree, right_child)
  return (parent, tree)
end



# Given an example and a tree, classify the example with the tree.
# NB: This function assumes that if an example has a feature value that is
# _less than or equal to_ a node's split value then the example should be sent
# to the left child, and otherwise sent to the right child.
function ClassifyExampleTree(example::Example, tree::Tree)
# return label of an example
# check the tree
  # for i in 1:length(tree)
  #   println("child info of node ", i, " is: left -> ", tree[i].left_child_id,
  #           " right -> ", tree[i].right_child_id,
  #           " leaf ? -> ", tree[i].leaf )
  # end
  node = tree[1] # 0-->1; const Node pointer, fine to do in this way in julia
  while node.leaf == false
    if example.values[node.split_feature] <= node.split_value
      node = tree[node.left_child_id]
    else
      node = tree[node.right_child_id]
    end
    # println(node.leaf)
  end
  # println("Complete reaching the leaf")
  if node.positive_weight >= node.negative_weight
    return 1
  else
    return -1
  end
end




# Given a set of examples and a tree,
# return the weighted error of tree on the examples.
function EvaluateTreeWgtd(examples::Vector{Example}, tree::Tree)
  wgtd_error = 0.
  for example in examples
    # println("...")
    if ClassifyExampleTree(example, tree) != example.label
      wgtd_error += example.weight
    end
  end
  return wgtd_error
end
