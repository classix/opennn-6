#include "multi_perceptron_layer.h"

namespace opennn
{

  /// Default constructor.
  /// It creates a empty layer object, with no perceptrons.
  /// This constructor also initializes the rest of class members to their default values.

  MultiPerceptronLayer::MultiPerceptronLayer() : Layer()
  {
    set();

    layer_type = Type::MultiPerceptron;
  }


  /// Layer architecture constructor.
  /// It creates a layer object with given numbers of inputs and perceptrons.
  /// The parameters are initialized at random.
  /// This constructor also initializes the rest of class members to their default values.
  /// @param new_inputs_number Number of inputs in the layer.
  /// @param new_neurons_number Number of perceptrons in the layer.

  MultiPerceptronLayer::MultiPerceptronLayer(const Index& new_inputs_number, const Index& new_neurons_number, const vector<size_t>& new_regCols, const vector<vector<size_t>>& new_catCols,
    const MultiPerceptronLayer::ActivationFunction& new_activation_function_reg, const MultiPerceptronLayer::ActivationFunction& new_activation_function_class) : Layer()
  {
    set(new_inputs_number, new_neurons_number, new_regCols, new_catCols, new_activation_function_reg, new_activation_function_class);

    layer_type = Type::MultiPerceptron;

    layer_name = "perceptron_layer";
  }


  /// Destructor.
  /// This destructor does not delete any pointer.

  MultiPerceptronLayer::~MultiPerceptronLayer()
  {
  }


  /// Returns the number of inputs to the layer.

  Index MultiPerceptronLayer::get_inputs_number() const
  {
    return synaptic_weights.dimension(0);
  }


  /// Returns the number of neurons in the layer.

  Index MultiPerceptronLayer::get_neurons_number() const
  {
    return biases.size();
  }


  Index MultiPerceptronLayer::get_biases_number() const
  {
    return biases.size();
  }


  /// Returns the number of layer's synaptic weights

  Index MultiPerceptronLayer::get_synaptic_weights_number() const
  {
    return synaptic_weights.size();
  }


  /// Returns the number of parameters(biases and synaptic weights) of the layer.

  Index MultiPerceptronLayer::get_parameters_number() const
  {
    return biases.size() + synaptic_weights.size();
  }


  /// Returns the biases from all the perceptrons in the layer.
  /// The format is a vector of real values.
  /// The size of this vector is the number of neurons in the layer.

  const Tensor<type, 2>& MultiPerceptronLayer::get_biases() const
  {
    return biases;
  }


  /// Returns the synaptic weights from the perceptrons.
  /// The format is a matrix of real values.
  /// The number of rows is the number of neurons in the layer.
  /// The number of columns is the number of inputs to the layer.

  const Tensor<type, 2>& MultiPerceptronLayer::get_synaptic_weights() const
  {
    return synaptic_weights;
  }


  Tensor<type, 2> MultiPerceptronLayer::get_synaptic_weights(const Tensor<type, 1>& parameters) const
  {
    const Index inputs_number = get_inputs_number();

    const Index neurons_number = get_neurons_number();

    const Index synaptic_weights_number = get_synaptic_weights_number();

    const Index parameters_size = parameters.size();

    const Index start_synaptic_weights_number = (parameters_size - synaptic_weights_number);

    Tensor<type, 1> new_synaptic_weights = parameters.slice(Eigen::array<Eigen::Index, 1>({ start_synaptic_weights_number }), Eigen::array<Eigen::Index, 1>({ synaptic_weights_number }));

    Eigen::array<Index, 2> two_dim{ {inputs_number, neurons_number} };

    return new_synaptic_weights.reshape(two_dim);
  }


  Tensor<type, 2> MultiPerceptronLayer::get_biases(const Tensor<type, 1>& parameters) const
  {
    const Index biases_number = biases.size();

    Tensor<type, 1> new_biases(biases_number);

    new_biases = parameters.slice(Eigen::array<Eigen::Index, 1>({ 0 }), Eigen::array<Eigen::Index, 1>({ biases_number }));

    Eigen::array<Index, 2> two_dim{ {1, biases.dimension(1)} };

    return new_biases.reshape(two_dim);

  }


  /// Returns a single vector with all the layer parameters.
  /// The format is a vector of real values.
  /// The size is the number of parameters in the layer.

  Tensor<type, 1> MultiPerceptronLayer::get_parameters() const
  {
    Tensor<type, 1> parameters(synaptic_weights.size() + biases.size());

    for (Index i = 0; i < biases.size(); i++)
    {
      fill_n(parameters.data() + i, 1, biases(i));
    }

    for (Index i = 0; i < synaptic_weights.size(); i++)
    {
      fill_n(parameters.data() + biases.size() + i, 1, synaptic_weights(i));
    }

    return parameters;
  }


  /// Returns the activation function of the layer.
  /// The activation function of a layer is the activation function of all perceptrons in it.

  std::pair<MultiPerceptronLayer::ActivationFunction, MultiPerceptronLayer::ActivationFunction> MultiPerceptronLayer::get_activation_functions() const
  {
    return std::make_pair(activation_function_reg, activation_function_class);
  }


  /// Returns a string with the name of the layer activation function.
  /// This can be: Logistic, HyperbolicTangent, Threshold, SymmetricThreshold, Linear, RectifiedLinear, ScaledExponentialLinear.

  string MultiPerceptronLayer::write_activation_functions() const
  {
    std::string activationFunctions = "";

    for (const auto& activation_function : { activation_function_reg, activation_function_class }) {
      switch (activation_function)
      {
      case Logistic:
        activationFunctions += "Logistic";
        break;

      case HyperbolicTangent:
        activationFunctions += "HyperbolicTangent";
        break;

      case Threshold:
        activationFunctions += "Threshold";
        break;

      case SymmetricThreshold:
        activationFunctions += "SymmetricThreshold";
        break;

      case Linear:
        activationFunctions += "Linear";
        break;

      case RectifiedLinear:
        activationFunctions += "RectifiedLinear";
        break;

      case ScaledExponentialLinear:
        activationFunctions += "ScaledExponentialLinear";
        break;

      case SoftPlus:
        activationFunctions += "SoftPlus";
        break;

      case SoftSign:
        activationFunctions += "SoftSign";
        break;

      case HardSigmoid:
        activationFunctions += "HardSigmoid";
        break;

      case ExponentialLinear:
        activationFunctions += "ExponentialLinear";
        break;

      case Binary:
        activationFunctions += "Binary";
        break;

      case Competitive:
        activationFunctions += "Competitive";
        break;

      case Softmax:
        activationFunctions += "Softmax";
        break;
      }

      activationFunctions += " ";
    }

    if (!activationFunctions.empty()) {
      activationFunctions.pop_back();
    }

    return activationFunctions;
  }


  /// Returns true if messages from this class are to be displayed on the screen,
  /// or false if messages from this class are not to be displayed on the screen.

  const bool& MultiPerceptronLayer::get_display() const
  {
    return display;
  }


  /// Sets an empty layer, wihtout any perceptron.
  /// It also sets the rest of members to their default values.

  void MultiPerceptronLayer::set()
  {
    biases.resize(0, 0);

    synaptic_weights.resize(0, 0);

    set_default();
  }


  /// Sets new numbers of inputs and perceptrons in the layer.
  /// It also sets the rest of members to their default values.
  /// @param new_inputs_number Number of inputs.
  /// @param new_neurons_number Number of perceptron neurons.

  void MultiPerceptronLayer::set(const Index& new_inputs_number, const Index& new_neurons_number, const vector<size_t>& new_regCols, const vector<vector<size_t>>& new_catCols,
    const MultiPerceptronLayer::ActivationFunction& new_activation_function_reg, const MultiPerceptronLayer::ActivationFunction& new_activation_function_class)
  {
    biases.resize(1, new_neurons_number);

    synaptic_weights.resize(new_inputs_number, new_neurons_number);

    set_parameters_random();

    activation_function_reg = new_activation_function_reg;
    activation_function_class = new_activation_function_class;

    regCols = new_regCols;

    catCols = new_catCols;

    set_default();
  }


  /// Sets those members not related to the vector of perceptrons to their default value.
  /// <ul>
  /// <li> Display: True.
  /// <li> layer_type: Perceptron_Layer.
  /// <li> trainable: True.
  /// </ul>

  void MultiPerceptronLayer::set_default()
  {
    layer_name = "perceptron_layer";

    display = true;

    layer_type = Type::MultiPerceptron;
  }


  void MultiPerceptronLayer::set_name(const string& new_layer_name)
  {
    layer_name = new_layer_name;
  }


  /// Sets a new number of inputs in the layer.
  /// The new synaptic weights are initialized at random.
  /// @param new_inputs_number Number of layer inputs.

  void MultiPerceptronLayer::set_inputs_number(const Index& new_inputs_number)
  {
    const Index neurons_number = get_neurons_number();

    biases.resize(1, neurons_number);

    synaptic_weights.resize(new_inputs_number, neurons_number);
  }


  /// Sets a new number perceptrons in the layer.
  /// All the parameters are also initialized at random.
  /// @param new_neurons_number New number of neurons in the layer.

  void MultiPerceptronLayer::set_neurons_number(const Index& new_neurons_number)
  {
    const Index inputs_number = get_inputs_number();

    biases.resize(1, new_neurons_number);

    synaptic_weights.resize(inputs_number, new_neurons_number);
  }


  /// Sets the biases of all perceptrons in the layer from a single vector.
  /// @param new_biases New set of biases in the layer.

  void MultiPerceptronLayer::set_biases(const Tensor<type, 2>& new_biases)
  {
    biases = new_biases;
  }


  /// Sets the synaptic weights of this perceptron layer from a single matrix.
  /// The format is a matrix of real numbers.
  /// The number of rows is the number of neurons in the corresponding layer.
  /// The number of columns is the number of inputs to the corresponding layer.
  /// @param new_synaptic_weights New set of synaptic weights in that layer.

  void MultiPerceptronLayer::set_synaptic_weights(const Tensor<type, 2>& new_synaptic_weights)
  {
    synaptic_weights = new_synaptic_weights;
  }


  /// Sets the parameters of this layer.

  void MultiPerceptronLayer::set_parameters(const Tensor<type, 1>& new_parameters, const Index& index)
  {
    const Index biases_number = get_biases_number();
    const Index synaptic_weights_number = get_synaptic_weights_number();

    memcpy(biases.data(),
      new_parameters.data() + index,
      static_cast<size_t>(biases_number) * sizeof(type));

    memcpy(synaptic_weights.data(),
      new_parameters.data() + biases_number + index,
      static_cast<size_t>(synaptic_weights_number) * sizeof(type));
  }


  /// This class sets a new activation(or transfer) function in a single layer.
  /// @param new_activation_function Activation function for the layer.

  void MultiPerceptronLayer::set_activation_functions(const MultiPerceptronLayer::ActivationFunction& new_activation_function_reg, const MultiPerceptronLayer::ActivationFunction& new_activation_function_class)
  {
    activation_function_reg = new_activation_function_reg;
    activation_function_class = new_activation_function_class;
  }


  /// Sets a new activation(or transfer) function in a single layer.
  /// The second argument is a string containing the name of the function("Logistic", "HyperbolicTangent", "Threshold", etc).
  /// @param new_activation_function Activation function for that layer.

  void MultiPerceptronLayer::set_activation_functions(const string& new_activation_function_name_reg, const string& new_activation_function_name_class)
  {
    if (new_activation_function_name_reg == "Logistic")
    {
      activation_function_reg = Logistic;
    }
    else if (new_activation_function_name_reg == "HyperbolicTangent")
    {
      activation_function_reg = HyperbolicTangent;
    }
    else if (new_activation_function_name_reg == "Threshold")
    {
      activation_function_reg = Threshold;
    }
    else if (new_activation_function_name_reg == "SymmetricThreshold")
    {
      activation_function_reg = SymmetricThreshold;
    }
    else if (new_activation_function_name_reg == "Linear")
    {
      activation_function_reg = Linear;
    }
    else if (new_activation_function_name_reg == "RectifiedLinear")
    {
      activation_function_reg = RectifiedLinear;
    }
    else if (new_activation_function_name_reg == "ScaledExponentialLinear")
    {
      activation_function_reg = ScaledExponentialLinear;
    }
    else if (new_activation_function_name_reg == "SoftPlus")
    {
      activation_function_reg = SoftPlus;
    }
    else if (new_activation_function_name_reg == "SoftSign")
    {
      activation_function_reg = SoftSign;
    }
    else if (new_activation_function_name_reg == "HardSigmoid")
    {
      activation_function_reg = HardSigmoid;
    }
    else if (new_activation_function_name_reg == "ExponentialLinear")
    {
      activation_function_reg = ExponentialLinear;
    }
    else
    {
      ostringstream buffer;

      buffer << "OpenNN Exception: MultiPerceptronLayer class.\n"
        << "void set_activation_function(const string&) method.\n"
        << "Unknown activation function: " << new_activation_function_name_reg << ".\n";

      throw logic_error(buffer.str());
    }

    if (new_activation_function_name_class == "Logistic")
    {
      activation_function_class = Logistic;
    }
    else if (new_activation_function_name_class == "Binary")
    {
      activation_function_class = Binary;
    }
    else if (new_activation_function_name_class == "Competitive")
    {
      activation_function_class = Competitive;
    }
    else if (new_activation_function_name_class == "Softmax")
    {
      activation_function_class = Softmax;
    }
    else
    {
      ostringstream buffer;

      buffer << "OpenNN Exception: MultiPerceptronLayer class.\n"
        << "void set_activation_function(const string&) method.\n"
        << "Unknown activation function: " << new_activation_function_name_class << ".\n";

      throw logic_error(buffer.str());
    }
  }


  /// Sets a new display value.
  /// If it is set to true messages from this class are to be displayed on the screen;
  /// if it is set to false messages from this class are not to be displayed on the screen.
  /// @param new_display Display value.

  void MultiPerceptronLayer::set_display(const bool& new_display)
  {
    display = new_display;
  }


  /// Initializes the biases of all the perceptrons in the layer of perceptrons with a given value.
  /// @param value Biases initialization value.

  void MultiPerceptronLayer::set_biases_constant(const type& value)
  {
    biases.setConstant(value);
  }


  /// Initializes the synaptic weights of all the perceptrons in the layer of perceptrons with a given value.
  /// @param value Synaptic weights initialization value.

  void MultiPerceptronLayer::set_synaptic_weights_constant(const type& value)
  {
    synaptic_weights.setConstant(value);
  }


  /// Initializes all the biases and synaptic weights in the neural newtork with a given value.
  /// @param value Parameters initialization value.

  void MultiPerceptronLayer::set_parameters_constant(const type& value)
  {
    biases.setConstant(value);

    synaptic_weights.setConstant(value);
  }


  /// Initializes all the biases and synaptic weights in the neural newtork at random with values comprised
  /// between -1 and +1.

  void MultiPerceptronLayer::set_parameters_random()
  {
    const type minimum = -0.2;
    const type maximum = 0.2;

    for (Index i = 0; i < biases.size(); i++)
    {
      const type random = static_cast<type>(rand() / (RAND_MAX + 1.0));

      biases(i) = minimum + (maximum - minimum) * random;
    }

    for (Index i = 0; i < synaptic_weights.size(); i++)
    {
      const type random = static_cast<type>(rand() / (RAND_MAX + 1.0));

      synaptic_weights(i) = minimum + (maximum - minimum) * random;
    }
  }


  void MultiPerceptronLayer::calculate_combinations(const Tensor<type, 2>& inputs,
    const Tensor<type, 2>& biases,
    const Tensor<type, 2>& synaptic_weights,
    Tensor<type, 2>& combinations) const
  {
#ifdef OPENNN_DEBUG
    check_columns_number(inputs, get_inputs_number(), LOG);
    //    check_dimensions(biases, 1, get_neurons_number(), LOG);
    check_dimensions(synaptic_weights, get_inputs_number(), get_neurons_number(), LOG);
    check_dimensions(combinations, inputs.dimension(0), get_neurons_number(), LOG);
#endif

    const Index batch_samples_number = inputs.dimension(0);
    const Index biases_number = get_biases_number();

    for (Index i = 0; i < biases_number; i++)
    {
      fill_n(combinations.data() + i * batch_samples_number, batch_samples_number, biases(i));
    }

    combinations.device(*thread_pool_device) += inputs.contract(synaptic_weights, A_B);
  }

  Eigen::Tensor<Eigen::Index, 1> createDimensionTensor(Eigen::Index d1, Eigen::Index d2) {
    Eigen::Tensor<Eigen::Index, 1> dimensionTensor(2);
    dimensionTensor(0) = d1;
    dimensionTensor(1) = d2;

    return dimensionTensor;
  }

  void MultiPerceptronLayer::calculate_activations(const Tensor<type, 2>& combinations, Tensor<type, 2>& activations_reg, vector<Tensor<type, 2>>& activations_class) const
  {
#ifdef OPENNN_DEBUG
    check_columns_number(combinations, get_neurons_number(), LOG);
    check_dimensions(activations, combinations.dimension(0), get_neurons_number(), LOG);
#endif

    Tensor<type, 2> regCombinations(combinations.dimension(0), regCols.size());
    for (size_t i = 0; i < regCombinations.dimension(0); i++) {
      for (size_t j = 0; j < regCombinations.dimension(1); j++) {
        regCombinations(i, j) = combinations(i, regCols[j]);
      }
    }
    //Tensor<type, 2> regCombinations = combinations.slice(Eigen::array<Index, 2>({ 0, 0 }), Eigen::array<Index, 2>({ combinations.dimension(0), regCols }));

    

    switch (activation_function_reg)
    {
    case Linear: linear(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1))); break;

    case Logistic: logistic(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1))); break;

    case HyperbolicTangent: hyperbolic_tangent(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1))); break;

    case Threshold: threshold(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1))); break;

    case SymmetricThreshold: symmetric_threshold(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1))); break;

    case RectifiedLinear: rectified_linear(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1))); break;

    case ScaledExponentialLinear: scaled_exponential_linear(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1))); break;

    case SoftPlus: soft_plus(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1))); break;

    case SoftSign: soft_sign(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1))); break;

    case HardSigmoid: hard_sigmoid(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1))); break;

    case ExponentialLinear: exponential_linear(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1))); break;
    }

    for (size_t c = 0; c < activations_class.size(); c++) {
      Tensor<type, 2> classCombinations(combinations.dimension(0), catCols[c].size());
      for (size_t i = 0; i < classCombinations.dimension(0); i++) {
        for (size_t j = 0; j < classCombinations.dimension(1); j++) {
          classCombinations(i, j) = combinations(i, catCols[c][j]);
        }
      }
      //Tensor<type, 2> classCombinations = combinations.slice(Eigen::array<Index, 2>({ 0, colOffset }), Eigen::array<Index, 2>({ combinations.dimension(0), categorySize }));

      switch (activation_function_class)
      {
      case Logistic: {
        if (classCombinations.dimension(1) == 1) {
          logistic(classCombinations.data(), createDimensionTensor(classCombinations.dimension(0), classCombinations.dimension(1)), activations_class[c].data(), createDimensionTensor(activations_class[c].dimension(0), activations_class[c].dimension(1)));
        }
        else {
          softmax(classCombinations.data(), createDimensionTensor(classCombinations.dimension(0), classCombinations.dimension(1)), activations_class[c].data(), createDimensionTensor(activations_class[c].dimension(0), activations_class[c].dimension(1)));
        }
        break;
      }

      case Binary: binary(classCombinations.data(), createDimensionTensor(classCombinations.dimension(0), classCombinations.dimension(1)), activations_class[c].data(), createDimensionTensor(activations_class[c].dimension(0), activations_class[c].dimension(1))); break;

      case Competitive: competitive(classCombinations.data(), createDimensionTensor(classCombinations.dimension(0), classCombinations.dimension(1)), activations_class[c].data(), createDimensionTensor(activations_class[c].dimension(0), activations_class[c].dimension(1))); break;

      case Softmax: {
        if (classCombinations.dimension(1) == 1) {
          logistic(classCombinations.data(), createDimensionTensor(classCombinations.dimension(0), classCombinations.dimension(1)), activations_class[c].data(), createDimensionTensor(activations_class[c].dimension(0), activations_class[c].dimension(1)));
        }
        else {
          softmax(classCombinations.data(), createDimensionTensor(classCombinations.dimension(0), classCombinations.dimension(1)), activations_class[c].data(), createDimensionTensor(activations_class[c].dimension(0), activations_class[c].dimension(1)));
        }
        break;
      }
      }
    }
  }


  void MultiPerceptronLayer::calculate_activations_derivatives(const Tensor<type, 2>& combinations,
    Tensor<type, 2>& activations_reg,
    vector<Tensor<type, 2>>& activations_class,
    Tensor<type, 2>& activations_derivatives_reg,
    vector<Tensor<type, 3>>& activations_derivatives_class) const
  {
#ifdef OPENNN_DEBUG
    check_columns_number(combinations, get_neurons_number(), LOG);
    check_dimensions(activations, combinations.dimension(0), get_neurons_number(), LOG);
    check_dimensions(activations_derivatives, combinations.dimension(0), get_neurons_number(), LOG);
#endif

    Tensor<type, 2> regCombinations(combinations.dimension(0), regCols.size());
    for (size_t i = 0; i < regCombinations.dimension(0); i++) {
      for (size_t j = 0; j < regCombinations.dimension(1); j++) {
        regCombinations(i, j) = combinations(i, regCols[j]);
      }
    }
    //Tensor<type, 2> regCombinations = combinations.slice(Eigen::array<Index, 2>({ 0, 0 }), Eigen::array<Index, 2>({ combinations.dimension(0), regCols }));

    switch (activation_function_reg)
    {
    case Linear: linear_derivatives(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1)), activations_derivatives_reg.data(), createDimensionTensor(activations_derivatives_reg.dimension(0), activations_derivatives_reg.dimension(1))); break;

    case Logistic: logistic_derivatives(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1)), activations_derivatives_reg.data(), createDimensionTensor(activations_derivatives_reg.dimension(0), activations_derivatives_reg.dimension(1))); break;

    case HyperbolicTangent: hyperbolic_tangent_derivatives(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1)), activations_derivatives_reg.data(), createDimensionTensor(activations_derivatives_reg.dimension(0), activations_derivatives_reg.dimension(1))); break;

    case Threshold: threshold_derivatives(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1)), activations_derivatives_reg.data(), createDimensionTensor(activations_derivatives_reg.dimension(0), activations_derivatives_reg.dimension(1))); break;

    case SymmetricThreshold: symmetric_threshold_derivatives(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1)), activations_derivatives_reg.data(), createDimensionTensor(activations_derivatives_reg.dimension(0), activations_derivatives_reg.dimension(1))); break;

    case RectifiedLinear: rectified_linear_derivatives(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1)), activations_derivatives_reg.data(), createDimensionTensor(activations_derivatives_reg.dimension(0), activations_derivatives_reg.dimension(1))); break;

    case ScaledExponentialLinear: scaled_exponential_linear_derivatives(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1)), activations_derivatives_reg.data(), createDimensionTensor(activations_derivatives_reg.dimension(0), activations_derivatives_reg.dimension(1))); break;

    case SoftPlus: soft_plus_derivatives(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1)), activations_derivatives_reg.data(), createDimensionTensor(activations_derivatives_reg.dimension(0), activations_derivatives_reg.dimension(1))); break;

    case SoftSign: soft_sign_derivatives(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1)), activations_derivatives_reg.data(), createDimensionTensor(activations_derivatives_reg.dimension(0), activations_derivatives_reg.dimension(1))); break;

    case HardSigmoid: hard_sigmoid_derivatives(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1)), activations_derivatives_reg.data(), createDimensionTensor(activations_derivatives_reg.dimension(0), activations_derivatives_reg.dimension(1))); break;

    case ExponentialLinear: exponential_linear_derivatives(regCombinations.data(), createDimensionTensor(regCombinations.dimension(0), regCombinations.dimension(1)), activations_reg.data(), createDimensionTensor(activations_reg.dimension(0), activations_reg.dimension(1)), activations_derivatives_reg.data(), createDimensionTensor(activations_derivatives_reg.dimension(0), activations_derivatives_reg.dimension(1))); break;
    }

    for (size_t c = 0; c < activations_derivatives_class.size(); c++) {
      Tensor<type, 2> classCombinations(combinations.dimension(0), catCols[c].size());
      for (size_t i = 0; i < classCombinations.dimension(0); i++) {
        for (size_t j = 0; j < classCombinations.dimension(1); j++) {
          classCombinations(i, j) = combinations(i, catCols[c][j]);
        }
      }
      //Tensor<type, 2> classCombinations = combinations.slice(Eigen::array<Index, 2>({ 0, colOffset }), Eigen::array<Index, 2>({ combinations.dimension(0), categorySize }));

      switch (activation_function_class)
      {
      case Logistic: {
        if (classCombinations.dimension(1) == 1) {
          logistic_derivatives(classCombinations.data(), createDimensionTensor(classCombinations.dimension(0), classCombinations.dimension(1)), activations_class[c].data(), createDimensionTensor(activations_class[c].dimension(0), activations_class[c].dimension(1)), activations_derivatives_class[c].data(), createDimensionTensor(activations_derivatives_class[c].dimension(0), activations_derivatives_class[c].dimension(1)));
        }
        else {
          softmax_derivatives(classCombinations.data(), createDimensionTensor(classCombinations.dimension(0), classCombinations.dimension(1)), activations_class[c].data(), createDimensionTensor(activations_class[c].dimension(0), activations_class[c].dimension(1)), activations_derivatives_class[c].data(), createDimensionTensor(activations_derivatives_class[c].dimension(0), activations_derivatives_class[c].dimension(1)));
        }
        break;
      }

      case Softmax: {
        if (classCombinations.dimension(1) == 1) {
          logistic_derivatives(classCombinations.data(), createDimensionTensor(classCombinations.dimension(0), classCombinations.dimension(1)), activations_class[c].data(), createDimensionTensor(activations_class[c].dimension(0), activations_class[c].dimension(1)), activations_derivatives_class[c].data(), createDimensionTensor(activations_derivatives_class[c].dimension(0), activations_derivatives_class[c].dimension(1)));
        }
        else {
          softmax_derivatives(classCombinations.data(), createDimensionTensor(classCombinations.dimension(0), classCombinations.dimension(1)), activations_class[c].data(), createDimensionTensor(activations_class[c].dimension(0), activations_class[c].dimension(1)), activations_derivatives_class[c].data(), createDimensionTensor(activations_derivatives_class[c].dimension(0), activations_derivatives_class[c].dimension(1)));
        }
        break;
      }

      default: break;
      }
    }
  }


  Tensor<type, 2> MultiPerceptronLayer::calculate_outputs(const Tensor<type, 2>& inputs)
  {
#ifdef OPENNN_DEBUG
    check_columns_number(inputs, get_inputs_number(), LOG);
#endif

    const Index batch_size = inputs.dimension(0);
    const Index outputs_number = get_neurons_number();

    Tensor<type, 2> outputs(batch_size, outputs_number);

    calculate_combinations(inputs, biases, synaptic_weights, outputs);

    // Seperate outputs
    Tensor<type, 2> regOutputs(batch_size, regCols.size());
    for (size_t i = 0; i < regOutputs.dimension(0); i++) {
      for (size_t j = 0; j < regOutputs.dimension(1); j++) {
        regOutputs(i, j) = outputs(i, regCols[j]);
      }
    }
    //Tensor<type, 2> regOutputs = outputs.slice(Eigen::array<Index, 2>({ 0, 0 }), Eigen::array<Index, 2>({ batch_size, regCols }));
    vector<Tensor<type, 2>> classOutputs;
    for (size_t c = 0; c < catCols.size(); c++) {
      Tensor<type, 2> classOutput(batch_size, catCols[c].size());
      for (size_t i = 0; i < classOutput.dimension(0); i++) {
        for (size_t j = 0; j < classOutput.dimension(1); j++) {
          classOutput(i, j) = outputs(i, catCols[c][j]);
        }
      }
      classOutputs.push_back(classOutput);
      //classOutputs.push_back(outputs.slice(Eigen::array<Index, 2>({ 0, colOffset }), Eigen::array<Index, 2>({ batch_size, categorySize})));
    }

    calculate_activations(outputs, regOutputs, classOutputs);

    // Merge outputs
    for (Index row = 0; row < batch_size; row++) {
      for (Index col = 0; col < regCols.size(); col++) {
        outputs(row, regCols[col]) = regOutputs(row, col);
      }
    }

    for (size_t i = 0; i < catCols.size(); i++) {
      for (Index row = 0; row < batch_size; row++) {
        for (Index col = 0; col < catCols[i].size(); col++) {
          outputs(row, catCols[i][col]) = classOutputs[i](row, col);
        }
      }
    }

    return outputs;
  }


  void MultiPerceptronLayer::forward_propagate(const Tensor<type, 2>& inputs,
    LayerForwardPropagation* forward_propagation)
  {
#ifdef OPENNN_DEBUG
    check_columns_number(inputs, get_inputs_number(), LOG);
#endif

    MultiPerceptronLayerForwardPropagation* perceptron_layer_forward_propagation
      = static_cast<MultiPerceptronLayerForwardPropagation*>(forward_propagation);

    calculate_combinations(inputs,
      biases,
      synaptic_weights,
      perceptron_layer_forward_propagation->combinations);

    calculate_activations_derivatives(perceptron_layer_forward_propagation->combinations,
      perceptron_layer_forward_propagation->activations_reg,
      perceptron_layer_forward_propagation->activations_class,
      perceptron_layer_forward_propagation->activations_derivatives_reg,
      perceptron_layer_forward_propagation->activations_derivatives_class);
  }


  void MultiPerceptronLayer::forward_propagate(const Tensor<type, 2>& inputs,
    Tensor<type, 1> potential_parameters,
    LayerForwardPropagation* forward_propagation)
  {
#ifdef OPENNN_DEBUG
    check_columns_number(inputs, get_inputs_number(), LOG);
    check_size(potential_parameters, get_parameters_number(), LOG);
#endif

    const Index neurons_number = get_neurons_number();

    const Index inputs_number = get_inputs_number();

    const TensorMap<Tensor<type, 2>> potential_biases(potential_parameters.data(), neurons_number, 1);

    const TensorMap<Tensor<type, 2>> potential_synaptic_weights(potential_parameters.data() + neurons_number, inputs_number, neurons_number);

    MultiPerceptronLayerForwardPropagation* multi_perceptron_layer_forward_propagation
      = static_cast<MultiPerceptronLayerForwardPropagation*>(forward_propagation);

    calculate_combinations(inputs,
      potential_biases,
      potential_synaptic_weights,
      multi_perceptron_layer_forward_propagation->combinations);

    calculate_activations_derivatives(multi_perceptron_layer_forward_propagation->combinations,
      multi_perceptron_layer_forward_propagation->activations_reg,
      multi_perceptron_layer_forward_propagation->activations_class,
      multi_perceptron_layer_forward_propagation->activations_derivatives_reg,
      multi_perceptron_layer_forward_propagation->activations_derivatives_class);
  }


  void MultiPerceptronLayer::forward_propagate(type* inputs_data,
    const Tensor<Index, 1>& inputs_dimensions,
    LayerForwardPropagation* forward_propagation,
    bool& switch_train)
  {
#ifdef OPENNN_DEBUG
    if (inputs_dimensions(1) != get_inputs_number())
    {
      ostringstream buffer;
      buffer << "OpenNN Exception: PerceptronLayer class.\n"
        << "void PerceptronLayer::forward_propagate(type*, const Tensor<Index, 1>&, type*, Tensor<Index, 1>&)\n"
        << "Inputs columns number must be equal to " << get_inputs_number() << ", (" << inputs_dimensions(1) << ").\n";
      throw invalid_argument(buffer.str());
    }

#endif

    MultiPerceptronLayerForwardPropagation* multi_perceptron_layer_forward_propagation
      = static_cast<MultiPerceptronLayerForwardPropagation*>(forward_propagation);

    TensorMap<Tensor<type, 2>> inputs(inputs_data, inputs_dimensions(0), inputs_dimensions(1));

    calculate_combinations(inputs,
      biases,
      synaptic_weights,
      multi_perceptron_layer_forward_propagation->combinations);

    if (switch_train) // Perform training
    {
      calculate_activations_derivatives(multi_perceptron_layer_forward_propagation->combinations,
        multi_perceptron_layer_forward_propagation->activations_reg,
        multi_perceptron_layer_forward_propagation->activations_class,
        multi_perceptron_layer_forward_propagation->activations_derivatives_reg,
        multi_perceptron_layer_forward_propagation->activations_derivatives_class);
    }
    else // Perform deployment
    {
      calculate_activations(multi_perceptron_layer_forward_propagation->combinations,
        multi_perceptron_layer_forward_propagation->activations_reg,
        multi_perceptron_layer_forward_propagation->activations_class);
    }
  }


  void MultiPerceptronLayer::insert_squared_errors_Jacobian_lm(LayerBackPropagationLM* back_propagation,
    const Index& index,
    Tensor<type, 2>& squared_errors_Jacobian) const
  {
    MultiPerceptronLayerBackPropagationLM* multi_perceptron_layer_back_propagation_lm =
      static_cast<MultiPerceptronLayerBackPropagationLM*>(back_propagation);

    const Index batch_samples_number = multi_perceptron_layer_back_propagation_lm->squared_errors_Jacobian.dimension(0);
    const Index layer_parameters_number = get_parameters_number();

    memcpy(squared_errors_Jacobian.data() + index,
      multi_perceptron_layer_back_propagation_lm->squared_errors_Jacobian.data(),
      static_cast<size_t>(layer_parameters_number * batch_samples_number) * sizeof(type));
  }


  void MultiPerceptronLayer::calculate_error_gradient(const Tensor<type, 2>& inputs,
    LayerForwardPropagation* forward_propagation,
    LayerBackPropagation* back_propagation) const
  {
    MultiPerceptronLayerForwardPropagation* multi_perceptron_layer_forward_propagation =
      static_cast<MultiPerceptronLayerForwardPropagation*>(forward_propagation);

    MultiPerceptronLayerBackPropagation* multi_perceptron_layer_back_propagation =
      static_cast<MultiPerceptronLayerBackPropagation*>(back_propagation);

    Tensor<type, 1> biases_derivatives_reg(regCols.size());
    biases_derivatives_reg.device(*thread_pool_device) =
      (multi_perceptron_layer_back_propagation->delta_reg * multi_perceptron_layer_forward_propagation->activations_derivatives_reg).sum(Eigen::array<Index, 1>({ 0 }));
    for (size_t col = 0; col < regCols.size(); col++) {
      multi_perceptron_layer_back_propagation->biases_derivatives(regCols[col]) = biases_derivatives_reg(col);
    }

    Tensor<type, 2> synaptic_weights_derivatives_reg(multi_perceptron_layer_back_propagation->synaptic_weights_derivatives.dimension(0), regCols.size());
    synaptic_weights_derivatives_reg.device(*thread_pool_device) =
      inputs.contract(multi_perceptron_layer_back_propagation->delta_reg * multi_perceptron_layer_forward_propagation->activations_derivatives_reg, AT_B);
    for (size_t row = 0; row < multi_perceptron_layer_back_propagation->synaptic_weights_derivatives.dimension(0); row++) {
      for (size_t col = 0; col < regCols.size(); col++) {
        multi_perceptron_layer_back_propagation->synaptic_weights_derivatives(row, regCols[col]) = synaptic_weights_derivatives_reg(row, col);
      }
    }

    for (size_t i = 0; i < catCols.size(); i++) {
      const Index samples_number = inputs.dimension(0);
      const Index neurons_number = catCols[i].size();

      Tensor<type, 1> biases_derivatives_class(neurons_number);
      Tensor<type, 2> synaptic_weights_derivatives_class(multi_perceptron_layer_back_propagation->synaptic_weights_derivatives.dimension(0), neurons_number);

      if (neurons_number == 1 || activation_function_class != Softmax) // Binary gradient
      {
        TensorMap< Tensor<type, 2> > activations_derivatives(multi_perceptron_layer_forward_propagation->activations_derivatives_class[i].data(), samples_number, neurons_number);

        biases_derivatives_class.device(*thread_pool_device) =
          (multi_perceptron_layer_back_propagation->delta_class[i] * activations_derivatives).sum(Eigen::array<Index, 1>({ 0 }));

        synaptic_weights_derivatives_class.device(*thread_pool_device) =
          inputs.contract((multi_perceptron_layer_back_propagation->delta_class[i] * activations_derivatives), AT_B);
      }
      else // Multiple gradient
      {
        const Index step = neurons_number * neurons_number;

        for (Index s = 0; s < samples_number; s++)
        {
          multi_perceptron_layer_back_propagation->delta_row[i] = multi_perceptron_layer_back_propagation->delta_class[i].chip(s, 0);

          TensorMap< Tensor<type, 2> > activations_derivatives_matrix(multi_perceptron_layer_forward_propagation->activations_derivatives_class[i].data() + s * step,
            neurons_number, neurons_number);

          multi_perceptron_layer_back_propagation->error_combinations_derivatives[i].chip(s, 0) =
            multi_perceptron_layer_back_propagation->delta_row[i].contract(activations_derivatives_matrix, AT_B);
        }

        biases_derivatives_class.device(*thread_pool_device) =
          (multi_perceptron_layer_back_propagation->error_combinations_derivatives[i]).sum(Eigen::array<Index, 1>({ 0 }));

        synaptic_weights_derivatives_class.device(*thread_pool_device) =
          inputs.contract(multi_perceptron_layer_back_propagation->error_combinations_derivatives[i], AT_B);
      }

      for (size_t col = 0; col < neurons_number; col++) {
        multi_perceptron_layer_back_propagation->biases_derivatives(catCols[i][col]) = biases_derivatives_class(col);
      }

      for (size_t row = 0; row < multi_perceptron_layer_back_propagation->synaptic_weights_derivatives.dimension(0); row++) {
        for (size_t col = 0; col < neurons_number; col++) {
          multi_perceptron_layer_back_propagation->synaptic_weights_derivatives(row, catCols[i][col]) = synaptic_weights_derivatives_class(row, col);
        }
      }
    }
  }


  void MultiPerceptronLayer::insert_gradient(LayerBackPropagation* back_propagation,
    const Index& index,
    Tensor<type, 1>& gradient) const
  {
    MultiPerceptronLayerBackPropagation* multi_perceptron_layer_back_propagation =
      static_cast<MultiPerceptronLayerBackPropagation*>(back_propagation);

    const Index biases_number = get_biases_number();
    const Index synaptic_weights_number = get_synaptic_weights_number();

    memcpy(gradient.data() + index,
      multi_perceptron_layer_back_propagation->biases_derivatives.data(),
      static_cast<size_t>(biases_number) * sizeof(type));

    memcpy(gradient.data() + index + biases_number,
      multi_perceptron_layer_back_propagation->synaptic_weights_derivatives.data(),
      static_cast<size_t>(synaptic_weights_number) * sizeof(type));
  }


  /// Returns a string with the expression of the inputs-outputs relationship of the layer.
  /// @param inputs_names vector of strings with the name of the layer inputs.
  /// @param outputs_names vector of strings with the name of the layer outputs.

  string MultiPerceptronLayer::write_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
  {
#ifdef OPENNN_DEBUG
    //    check_size(inputs_names, get_inputs_number(), LOG);
    //    check_size(outputs_names, get_neurons_number(), LOG);
#endif

    ostringstream buffer;

    for (Index j = 0; j < outputs_names.size(); j++)
    {
      const Tensor<type, 1> synaptic_weights_column = synaptic_weights.chip(j, 1);

      buffer << outputs_names[j] << to_string(j) << " = " << write_activation_function_expression() << "( " << biases(0, j) << " +";

      for (Index i = 0; i < inputs_names.size() - 1; i++)
      {

        buffer << " (" << inputs_names[i] << "*" << synaptic_weights_column(i) << ") +";
      }

      buffer << " (" << inputs_names[inputs_names.size() - 1] << "*" << synaptic_weights_column[inputs_names.size() - 1] << ") );\n";
    }

    return buffer.str();
  }


  void MultiPerceptronLayer::from_XML(const tinyxml2::XMLDocument& document)
  {
    ostringstream buffer;

    // Perceptron layer

    const tinyxml2::XMLElement* perceptron_layer_element = document.FirstChildElement("MultiPerceptronLayer");

    if (!perceptron_layer_element)
    {
      buffer << "OpenNN Exception: MultiPerceptronLayer class.\n"
        << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
        << "MultiPerceptronLayer element is nullptr.\n";

      throw logic_error(buffer.str());
    }

    // Layer name

    const tinyxml2::XMLElement* layer_name_element = perceptron_layer_element->FirstChildElement("LayerName");

    if (!layer_name_element)
    {
      buffer << "OpenNN Exception: MultiPerceptronLayer class.\n"
        << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
        << "LayerName element is nullptr.\n";

      throw logic_error(buffer.str());
    }

    if (layer_name_element->GetText())
    {
      set_name(layer_name_element->GetText());
    }

    // Inputs number

    const tinyxml2::XMLElement* inputs_number_element = perceptron_layer_element->FirstChildElement("InputsNumber");

    if (!inputs_number_element)
    {
      buffer << "OpenNN Exception: MultiPerceptronLayer class.\n"
        << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
        << "InputsNumber element is nullptr.\n";

      throw logic_error(buffer.str());
    }

    if (inputs_number_element->GetText())
    {
      set_inputs_number(static_cast<Index>(stoi(inputs_number_element->GetText())));
    }

    // Neurons number

    const tinyxml2::XMLElement* neurons_number_element = perceptron_layer_element->FirstChildElement("NeuronsNumber");

    if (!neurons_number_element)
    {
      buffer << "OpenNN Exception: MultiPerceptronLayer class.\n"
        << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
        << "NeuronsNumber element is nullptr.\n";

      throw logic_error(buffer.str());
    }

    if (neurons_number_element->GetText())
    {
      set_neurons_number(static_cast<Index>(stoi(neurons_number_element->GetText())));
    }

    // Activation function

    const tinyxml2::XMLElement* activation_function_element = perceptron_layer_element->FirstChildElement("ActivationFunction");

    if (!activation_function_element)
    {
      buffer << "OpenNN Exception: MultiPerceptronLayer class.\n"
        << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
        << "ActivationFunction element is nullptr.\n";

      throw logic_error(buffer.str());
    }

    if (activation_function_element->GetText())
    {
      const string activation_function_string = activation_function_element->GetText();

      size_t ind = activation_function_string.find(' ');

      set_activation_functions(activation_function_string.substr(0, ind), activation_function_string.substr(ind + 1));
    }

    // Parameters

    const tinyxml2::XMLElement* parameters_element = perceptron_layer_element->FirstChildElement("Parameters");

    if (!parameters_element)
    {
      buffer << "OpenNN Exception: MultiPerceptronLayer class.\n"
        << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
        << "Parameters element is nullptr.\n";

      throw logic_error(buffer.str());
    }

    if (parameters_element->GetText())
    {
      const string parameters_string = parameters_element->GetText();

      set_parameters(to_type_vector(parameters_string, ' '));
    }

    // Project type neurons

    const tinyxml2::XMLElement* regCols_element = perceptron_layer_element->FirstChildElement("regCols");

    if (!regCols_element)
    {
      buffer << "OpenNN Exception: MultiPerceptronLayer class.\n"
        << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
        << "regCols element is nullptr.\n";

      throw logic_error(buffer.str());
    }

    if (regCols_element->GetText()) {
      const string regCols_string = regCols_element->GetText();

      Tensor<type, 1> regCols_tensor = to_type_vector(regCols_string, ' ');

      regCols.clear();
      for (size_t i = 0; i < regCols_tensor.size(); i++) {
        regCols.push_back(regCols_tensor(i));
      }
    }

    const tinyxml2::XMLElement* catCols_element = perceptron_layer_element->FirstChildElement("catCols");

    if (!catCols_element)
    {
      buffer << "OpenNN Exception: MultiPerceptronLayer class.\n"
        << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
        << "regCols element is nullptr.\n";

      throw logic_error(buffer.str());
    }

    if (catCols_element->GetText()) {
      const string catCols_string = catCols_element->GetText();

      Tensor<type, 1> catCols_tensor = to_type_vector(catCols_string, ' ');

      catCols.clear();
      vector<size_t> currentCatCols;
      for (size_t i = 0; i < catCols_tensor.size(); i++) {
        if (catCols_tensor(i) == -1) {
          catCols.push_back(currentCatCols);
          currentCatCols.clear();
        }
        else {
          currentCatCols.push_back(catCols_tensor(i));
        }
      }
    }
  }


  void MultiPerceptronLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
  {
    ostringstream buffer;

    // Perceptron layer

    file_stream.OpenElement("MultiPerceptronLayer");

    // Layer name
    file_stream.OpenElement("LayerName");
    buffer.str("");
    buffer << layer_name;
    file_stream.PushText(buffer.str().c_str());
    file_stream.CloseElement();

    // Inputs number
    file_stream.OpenElement("InputsNumber");

    buffer.str("");
    buffer << get_inputs_number();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Outputs number

    file_stream.OpenElement("NeuronsNumber");

    buffer.str("");
    buffer << get_neurons_number();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Activation function

    file_stream.OpenElement("ActivationFunction");

    file_stream.PushText(write_activation_functions().c_str());

    file_stream.CloseElement();

    // Parameters

    file_stream.OpenElement("Parameters");

    buffer.str("");

    const Tensor<type, 1> parameters = get_parameters();
    const Index parameters_size = parameters.size();

    for (Index i = 0; i < parameters_size; i++)
    {
      buffer << parameters(i);

      if (i != (parameters_size - 1)) buffer << " ";
    }

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Project type neurons

    file_stream.OpenElement("regCols");

    buffer.str("");

    for (Index i = 0; i < regCols.size(); i++) {
      buffer << regCols[i];

      if (i != regCols.size() - 1) {
        buffer << " ";
      }
    }

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    file_stream.OpenElement("catCols");

    buffer.str("");

    for (Index c = 0; c < catCols.size(); c++) {
      if (c != 0) {
        buffer << " ";
      }

      for (Index i = 0; i < catCols[c].size(); i++) {
        buffer << catCols[c][i];

        if (i != catCols[c].size() - 1) {
          buffer << " ";
        }
      }

      buffer << " -1";
    }

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Peceptron layer (end tag)

    file_stream.CloseElement();
  }


  string MultiPerceptronLayer::write_activation_function_expression() const
  {
    std::string activationFunctions = "";

    for (const auto& activation_function : { activation_function_reg, activation_function_class }) {
      switch (activation_function)
      {
      case Logistic:
        activationFunctions += "logistic";
        break;

      case HyperbolicTangent:
        activationFunctions += "tanh";
        break;

      case Threshold:
        activationFunctions += "threshold";
        break;

      case SymmetricThreshold:
        activationFunctions += "symmetric_threshold";
        break;

      case Linear:
        activationFunctions += "";
        break;

      case RectifiedLinear:
        activationFunctions += "ReLU";
        break;

      case ScaledExponentialLinear:
        activationFunctions += "SELU";
        break;

      case SoftPlus:
        activationFunctions += "soft_plus";
        break;

      case SoftSign:
        activationFunctions += "soft_sign";
        break;

      case HardSigmoid:
        activationFunctions += "hard_sigmoid";
        break;

      case ExponentialLinear:
        activationFunctions += "ELU";
        break;

      case Binary:
        activationFunctions += "binary";
        break;

      case Competitive:
        activationFunctions += "competitive";
        break;

      case Softmax:
        activationFunctions += "softmax";
        break;
      }
    }

    return activationFunctions;
  }


  string MultiPerceptronLayer::write_combinations_c() const
  {
    ostringstream buffer;

    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

    buffer << "\tvector<float> combinations(" << neurons_number << ");\n" << endl;

    for (Index i = 0; i < neurons_number; i++)
    {
      buffer << "\tcombinations[" << i << "] = " << biases(i);

      for (Index j = 0; j < inputs_number; j++)
      {
        buffer << " +" << synaptic_weights(j, i) << "*inputs[" << j << "]";
      }

      buffer << ";" << endl;
    }

    return buffer.str();
  }


  string MultiPerceptronLayer::write_activations_c() const
  {
    ostringstream buffer;

    const Index neurons_number = get_neurons_number();

    buffer << "\n\tvector<float> activations(" << neurons_number << ");\n" << endl;

    for (Index i = 0; i < neurons_number; i++)
    {
      buffer << "\tactivations[" << i << "] = ";

      switch (activation_function_reg)
      {

      case HyperbolicTangent:
        buffer << "tanh(combinations[" << i << "]);\n";
        break;

      case RectifiedLinear:
        buffer << "combinations[" << i << "] < 0.0 ? 0.0 : combinations[" << i << "];\n";
        break;

      case Logistic:
        buffer << "1.0/(1.0 + exp(-combinations[" << i << "]));\n";
        break;

      case Threshold:
        buffer << "combinations[" << i << "] >= 0.0 ? 1.0 : 0.0;\n";
        break;

      case SymmetricThreshold:
        buffer << "combinations[" << i << "] >= 0.0 ? 1.0 : -1.0;\n";
        break;

      case Linear:
        buffer << "combinations[" << i << "];\n";
        break;

      case ScaledExponentialLinear:
        buffer << "combinations[" << i << "] < 0.0 ? 1.0507*1.67326*(exp(combinations[" << i << "]) - 1.0) : 1.0507*combinations[" << i << "];\n";
        break;

      case SoftPlus:
        buffer << "log(1.0 + exp(combinations[" << i << "]));\n";
        break;

      case SoftSign:
        buffer << "combinations[" << i << "] < 0.0 ? combinations[" << i << "]/(1.0 - combinations[" << i << "] ) : combinations[" << i << "]/(1.0 + combinations[" << i << "] );\n";
        break;

      case ExponentialLinear:
        buffer << "combinations[" << i << "] < 0.0 ? 1.0*(exp(combinations[" << i << "]) - 1.0) : combinations[" << i << "];\n";
        break;

      case HardSigmoid:
        ///@todo
        break;

      }
    }

    return buffer.str();
  }


  string MultiPerceptronLayer::write_combinations_python() const
  {
    ostringstream buffer;

    const Index inputs_number = get_inputs_number();
    const Index neurons_number = get_neurons_number();

    buffer << "\t\tcombinations = [None] * " << neurons_number << "\n" << endl;

    for (Index i = 0; i < neurons_number; i++)
    {
      buffer << "\t\tcombinations[" << i << "] = " << biases(i);

      for (Index j = 0; j < inputs_number; j++)
      {
        buffer << " +" << synaptic_weights(j, i) << "*inputs[" << j << "]";
      }

      buffer << " " << endl;
    }

    buffer << "\t\t" << endl;

    return buffer.str();
  }


  string MultiPerceptronLayer::write_activations_python() const
  {
    ostringstream buffer;

    const Index neurons_number = get_neurons_number();

    buffer << "\t\tactivations = [None] * " << neurons_number << "\n" << endl;

    for (Index i = 0; i < neurons_number; i++)
    {
      buffer << "\t\tactivations[" << i << "] = ";

      switch (activation_function_reg)
      {

      case HyperbolicTangent:
        buffer << "np.tanh(combinations[" << i << "])\n";
        break;

      case RectifiedLinear:
        buffer << "np.maximum(0.0, combinations[" << i << "])\n";
        break;

      case Logistic:
        buffer << "1.0/(1.0 + np.exp(-combinations[" << i << "]))\n";
        break;

      case Threshold:
        buffer << "1.0 if combinations[" << i << "] >= 0.0 else 0.0\n";
        break;

      case SymmetricThreshold:
        buffer << "1.0 if combinations[" << i << "] >= 0.0 else -1.0\n";
        break;

      case Linear:
        buffer << "combinations[" << i << "]\n";
        break;

      case ScaledExponentialLinear:
        buffer << "1.0507*1.67326*(np.exp(combinations[" << i << "]) - 1.0) if combinations[" << i << "] < 0.0 else 1.0507*combinations[" << i << "]\n";
        break;

      case SoftPlus:
        buffer << "np.log(1.0 + np.exp(combinations[" << i << "]))\n";
        break;

      case SoftSign:
        buffer << "combinations[" << i << "]/(1.0 - combinations[" << i << "] ) if combinations[" << i << "] < 0.0 else combinations[" << i << "]/(1.0 + combinations[" << i << "] )\n";
        break;

      case ExponentialLinear:
        buffer << "1.0*(np.exp(combinations[" << i << "]) - 1.0) if combinations[" << i << "] < 0.0 else combinations[" << i << "]\n";
        break;

      case HardSigmoid:
        ///@todo
        break;
      }
    }

    return buffer.str();
  }


  string MultiPerceptronLayer::write_expression_c() const
  {
    ostringstream buffer;

    buffer << "vector<float> " << layer_name << "(const vector<float>& inputs)\n{" << endl;

    buffer << write_combinations_c();

    buffer << write_activations_c();

    buffer << "\n\treturn activations;\n}" << endl;

    return buffer.str();
  }


  string MultiPerceptronLayer::write_expression_python() const
  {
    ostringstream buffer;

    buffer << "\tdef " << layer_name << "(self,inputs):\n" << endl;

    buffer << write_combinations_python();

    buffer << write_activations_python();

    buffer << "\n\t\treturn activations;\n" << endl;

    return buffer.str();
  }

}