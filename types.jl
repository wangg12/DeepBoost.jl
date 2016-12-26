# workspace() # clear workspace by providing a clean workspace

# Used in many places as the minimum possible difference between two distinct
# numbers. Helps make code stable, tests predictable, etc.
kTolerance = Float64(1e-7)

typealias Feature Int64
typealias Label Int64
typealias NodeId Int64
typealias Value Float64
typealias Weight Float64

# An example consists of a vector of feature values, a label and a weight.
# Note that this is a dense feature representation; the value of every
# feature is contained in the vector, listed in a canonical order.
type Example
  values::Vector # Float64 Vector
  label::Label # Int64
  weight::Weight # Float64 probability (distribution)
  # construct function
  Example() = new()
  Example(values::Vector, # Float64 Vector
          label::Label, # Int64
          weight::Weight) = new(values, label, weight)
end

# a tree node
type Node
  examples::Array{Example, 1} # examples at this node
  split_feature::Int64 # Int64, split feature
  split_value::Float64 # Float64, split value
  left_child_id::Int64 # pointer to left child, if any
  right_child_id::Int64 # pointer to right child id, if any
  positive_weight::Float64 # Total weight of positive examples at this node.
  negative_weight::Float64 # total weight of negative examples at this node
  leaf::Bool # is this node a leaf?
  depth::Int64 # depth of the node in the tree. Root node has depth 0.
  Node() = new()
  Node(examples::Array{Example, 1}, # examples at this node
    split_feature::Int64, # Int64, split feature
    split_value::Float64, # Float64, split value
    left_child_id::Int64, # pointer to left child, if any
    right_child_id::Int64, # pointer to right child id, if any
    positive_weight::Float64, # Total weight of positive examples at this node.
    negative_weight::Float64, # total weight of negative examples at this node
    leaf::Bool, # is this node a leaf?
    depth::Int64) = new(examples, split_feature,split_value,
                        left_child_id, right_child_id,
                        positive_weight, negative_weight,
                        leaf, depth)
end



# a tree is a vector of nodes
typealias Tree Vector{Node}

type classifier
    weight::Weight
    tree::Tree
    classifier() = new()
    classifier(weight, tree) = new(weight, tree)
end

# a model is a vector of (weight, tree) pairs, i.e. a weighted combination of trees.
# by yy: for tuple is immutable
typealias Model Vector{classifier}


# a = Example([1.0,2.0,3.0],1,2)
# b = Node([a,a,a],2, 1.0, 2, 5, 6.0, 6.0, false,2)
# c = Node()
# d = Tree([b,b])
# e = Model([(1.2,d),(1.3,d)])
