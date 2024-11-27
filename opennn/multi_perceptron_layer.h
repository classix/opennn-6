#ifndef MULTIPERCEPTRONLAYER_H
#define MULTIPERCEPTRONLAYER_H

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "config.h"
#include "layer.h"
#include "probabilistic_layer.h"
#include "perceptron_layer.h"

#include "opennn_strings.h"

namespace opennn
{

  struct MultiPerceptronLayerForwardPropagation;
  struct MultiPerceptronLayerBackPropagation;
  struct MultiPerceptronLayerBackPropagationLM;

#ifdef OPENNN_CUDA
#include "../../opennn-cuda/opennn_cuda/struct_perceptron_layer_cuda.h"
#endif


  /// This class represents a layer of perceptrons.

  /// MultiPerceptronLayer is a single-layer network with a hard-limit trabsfer function.
  /// This network is often trained with the perceptron learning rule.
  ///
  /// Layers of perceptrons will be used to construct multilayer perceptrons, such as an approximation problems .

  class MultiPerceptronLayer : public Layer

  {

  public:

    /// Enumeration of available activation functions for the perceptron neuron model.

    enum ActivationFunction {
      Threshold, SymmetricThreshold, Logistic, HyperbolicTangent, Linear, RectifiedLinear,
      ExponentialLinear, ScaledExponentialLinear, SoftPlus, SoftSign, HardSigmoid,

      Binary, Competitive, Softmax
    };

    // Constructors

    explicit MultiPerceptronLayer();

    explicit MultiPerceptronLayer(const Index&, const Index&, const vector<size_t>&, const vector<vector<size_t>>&, const ActivationFunction & = MultiPerceptronLayer::HyperbolicTangent, const ActivationFunction & = MultiPerceptronLayer::Softmax);

    // Destructor

    virtual ~MultiPerceptronLayer();

    // Get methods

    bool is_empty() const;

    Index get_inputs_number() const;
    Index get_neurons_number() const;

    // Parameters

    const Tensor<type, 2>& get_biases() const;
    const Tensor<type, 2>& get_synaptic_weights() const;

    Tensor<type, 2> get_biases(const Tensor<type, 1>&) const;
    Tensor<type, 2> get_synaptic_weights(const Tensor<type, 1>&) const;

    Index get_biases_number() const;
    Index get_synaptic_weights_number() const;
    Index get_parameters_number() const;
    Tensor<type, 1> get_parameters() const;

    // Activation functions

    std::pair<ActivationFunction, ActivationFunction> get_activation_functions() const;

    string write_activation_functions() const;

    // Display messages

    const bool& get_display() const;

    // Set methods

    void set();
    void set(const Index&, const Index&, const vector<size_t>&, const vector<vector<size_t>>&, const ActivationFunction & = MultiPerceptronLayer::HyperbolicTangent, const ActivationFunction & = MultiPerceptronLayer::Softmax);

    void set_default();
    void set_name(const string&);

    // Architecture

    void set_inputs_number(const Index&);
    void set_neurons_number(const Index&);

    // Parameters

    void set_biases(const Tensor<type, 2>&);
    void set_synaptic_weights(const Tensor<type, 2>&);

    void set_parameters(const Tensor<type, 1>&, const Index& index = 0);

    // Activation functions

    void set_activation_functions(const ActivationFunction&, const ActivationFunction&);
    void set_activation_functions(const string&, const string&);

    // Display messages

    void set_display(const bool&);

    // Parameters initialization methods
    void set_biases_constant(const type&);
    void set_synaptic_weights_constant(const type&);


    void set_parameters_constant(const type&);

    void set_parameters_random();

    // MultiPerceptronLayer layer combinations

    void calculate_combinations(const Tensor<type, 2>&,
      const Tensor<type, 2>&,
      const Tensor<type, 2>&,
      Tensor<type, 2>&) const;

    // MultiPerceptronLayer layer activations

    void calculate_activations(const Tensor<type, 2>&,
      Tensor<type, 2>&,
      vector<Tensor<type, 2>>&) const;

    void calculate_activations_derivatives(const Tensor<type, 2>&,
      Tensor<type, 2>&, vector<Tensor<type, 2>>&,
      Tensor<type, 2>&, vector<Tensor<type, 3>>&) const;

    // MultiPerceptronLayer layer outputs

    Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&);

    void forward_propagate(const Tensor<type, 2>&,
      LayerForwardPropagation*);


    void forward_propagate(const Tensor<type, 2>&,
      Tensor<type, 1>,
      LayerForwardPropagation*);

    void forward_propagate(type*, const Tensor<Index, 1>&, LayerForwardPropagation*, bool&) final;

    // Squared errors methods

    void insert_squared_errors_Jacobian_lm(LayerBackPropagationLM*,
      const Index&,
      Tensor<type, 2>&) const;

    // Gradient methods

    void calculate_error_gradient(const Tensor<type, 2>&,
      LayerForwardPropagation*,
      LayerBackPropagation*) const;

    void calculate_error_gradient(type*,
      LayerForwardPropagation*,
      LayerBackPropagation*) const final;

    void insert_gradient(LayerBackPropagation*,
      const Index&,
      Tensor<type, 1>&) const;

    // Expression methods   

    string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

    string write_activation_function_expression() const;

    string write_expression_c() const;
    string write_combinations_c() const;
    string write_activations_c() const;

    string write_combinations_python() const;
    string write_activations_python() const;
    string write_expression_python() const;

    // Serialization methods

    void from_XML(const tinyxml2::XMLDocument&);
    void write_XML(tinyxml2::XMLPrinter&) const;

    void setRegCols(const vector<size_t>& rcs) { regCols = rcs; }
    const vector<size_t>& getRegCols() const { return regCols; }
    Index getRegColCount() const { return regCols.size(); }
    void setCatCols(const vector<vector<size_t>>& css) { catCols = css; }
    const vector<vector<size_t>>& getCatCols() const { return catCols; }
    Index getCatCount() const { return catCols.size(); }
    vector<Index> getCatColCount() const {
      vector<Index> sizes;
      sizes.reserve(catCols.size());
      transform(catCols.begin(), catCols.end(), std::back_inserter(sizes),
        [](const vector<size_t>& vec) { return vec.size(); });
      return sizes;
    }

  protected:

    // MEMBERS

    /// Bias is a neuron parameter that is summed with the neuron's weighted inputs
    /// and passed through the neuron's transfer function to generate the neuron's output.

    Tensor<type, 2> biases;

    /// This matrix containing conection strengths from a layer's inputs to its neurons.

    Tensor<type, 2> synaptic_weights;

    /// Activation function variable.

    ActivationFunction activation_function_reg;
    ActivationFunction activation_function_class;

    /// Display messages to screen. 

    bool display = true;

    vector<size_t> regCols;
    vector<vector<size_t>> catCols;

#ifdef OPENNN_CUDA
#include "../../opennn-cuda/opennn_cuda/perceptron_layer_cuda.h"
#else
  };
#endif

  struct MultiPerceptronLayerForwardPropagation : LayerForwardPropagation
  {
    // Default constructor

    explicit MultiPerceptronLayerForwardPropagation() : LayerForwardPropagation()
    {
    }

    explicit MultiPerceptronLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
      : LayerForwardPropagation()
    {
      set(new_batch_samples_number, new_layer_pointer);
    }

    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    {
      layer_pointer = new_layer_pointer;

      batch_samples_number = new_batch_samples_number;

      const Index neurons_number = layer_pointer->get_neurons_number();
      const Index regCols = static_cast<MultiPerceptronLayer*>(new_layer_pointer)->getRegColCount();
      const vector<Index>& categorySizes = static_cast<MultiPerceptronLayer*>(new_layer_pointer)->getCatColCount();

      combinations.resize(batch_samples_number, neurons_number);

      activations_reg.resize(batch_samples_number, regCols);
      activations_class.clear();
      for (const auto& categorySize : categorySizes) {
        activations_class.push_back(Tensor<type, 2>(batch_samples_number, categorySize));
      }

      activations_derivatives_reg.resize(batch_samples_number, regCols);
      activations_derivatives_class.clear();
      for (const auto& categorySize : categorySizes) {
        activations_derivatives_class.push_back(Tensor<type, 3>(batch_samples_number, categorySize, categorySize));
      }
    }

    void print() const
    {
      cout << "Combinations:" << endl;
      cout << combinations << endl;

      cout << "Activations:" << endl;
      cout << activations_reg << endl;
      for (const auto& activations : activations_class) {
        cout << activations << endl;
      }

      cout << "Activations derivatives:" << endl;
      cout << activations_derivatives_reg << endl;
      for (const auto& activations_derivatives : activations_derivatives_class) {
        cout << activations_derivatives << endl;
      }
    }

    Tensor<type, 2> combinations;
    Tensor<type, 2> activations_reg;
    vector<Tensor<type, 2>> activations_class;
    Tensor<type, 2> activations_derivatives_reg;
    vector<Tensor<type, 3>> activations_derivatives_class;
  };


  struct MultiPerceptronLayerBackPropagationLM : LayerBackPropagationLM
  {
    // Default constructor

    explicit MultiPerceptronLayerBackPropagationLM() : LayerBackPropagationLM()
    {

    }


    explicit MultiPerceptronLayerBackPropagationLM(const Index& new_batch_samples_number, Layer* new_layer_pointer)
      : LayerBackPropagationLM()
    {
      set(new_batch_samples_number, new_layer_pointer);
    }


    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    {
      layer_pointer = new_layer_pointer;

      batch_samples_number = new_batch_samples_number;

      const Index neurons_number = layer_pointer->get_neurons_number();
      const Index parameters_number = layer_pointer->get_parameters_number();

      delta_reg.resize(batch_samples_number, static_cast<MultiPerceptronLayer*>(new_layer_pointer)->getRegColCount());
      delta_class.clear();
      delta_row.clear();
      for (const auto& categorySize : static_cast<MultiPerceptronLayer*>(new_layer_pointer)->getCatColCount()) {
        delta_class.push_back(Tensor<type, 2>(batch_samples_number, categorySize));
        delta_row.push_back(Tensor<type, 1>(categorySize));
      }

      squared_errors_Jacobian.resize(batch_samples_number, parameters_number);

      error_combinations_derivatives.resize(batch_samples_number, neurons_number);
    }

    void print() const
    {
      cout << "Delta:" << endl;
      cout << delta_reg << endl;

      cout << "Squared errors Jacobian: " << endl;
      cout << squared_errors_Jacobian << endl;

    }

    Tensor<type, 2> delta_reg;
    vector<Tensor<type, 2>> delta_class;

    vector<Tensor<type, 1>> delta_row;

    Tensor<type, 2> squared_errors_Jacobian;

    Tensor<type, 2> error_combinations_derivatives;
  };



  struct MultiPerceptronLayerBackPropagation : LayerBackPropagation
  {
    // Default constructor

    explicit MultiPerceptronLayerBackPropagation() : LayerBackPropagation()
    {

    }


    explicit MultiPerceptronLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
      : LayerBackPropagation()
    {
      set(new_batch_samples_number, new_layer_pointer);
    }


    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    {
      layer_pointer = new_layer_pointer;

      batch_samples_number = new_batch_samples_number;

      const Index neurons_number = layer_pointer->get_neurons_number();
      const Index inputs_number = layer_pointer->get_inputs_number();

      delta_reg.resize(batch_samples_number, static_cast<MultiPerceptronLayer*>(new_layer_pointer)->getRegColCount());
      delta_class.clear();
      delta_row.clear();
      error_combinations_derivatives.clear();
      for (const auto& categorySize : static_cast<MultiPerceptronLayer*>(new_layer_pointer)->getCatColCount()) {
        delta_class.push_back(Tensor<type, 2>(batch_samples_number, categorySize));
        delta_row.push_back(Tensor<type, 1>(categorySize));
        error_combinations_derivatives.push_back(Tensor<type, 2>(batch_samples_number, categorySize));
      }

      biases_derivatives.resize(neurons_number);

      synaptic_weights_derivatives.resize(inputs_number, neurons_number);
    }

    void print() const
    {
      cout << "Delta:" << endl;
      cout << delta_reg << endl;

      cout << "Biases derivatives:" << endl;
      cout << biases_derivatives << endl;

      cout << "Synaptic weights derivatives:" << endl;
      cout << synaptic_weights_derivatives << endl;
    }

    Tensor<type, 2> delta_reg;
    vector<Tensor<type, 2>> delta_class;

    vector<Tensor<type, 1>> delta_row;

    Tensor<type, 1> biases_derivatives;
    Tensor<type, 2> synaptic_weights_derivatives;

    vector<Tensor<type, 2>> error_combinations_derivatives;
  };



}

#endif