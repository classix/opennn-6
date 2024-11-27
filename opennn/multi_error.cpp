#include "multi_error.h"

#include "training_strategy.h"

namespace opennn
{
  /// Default constructor.
/// It creates a sum squared error term not associated to any neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

  MultiError::MultiError() : LossIndex()
  {
    set_default();
  }


  /// Neural network and data set constructor.
  /// It creates a sum squared error associated to a neural network and measured on a data set.
  /// It also initializes all the rest of class members to their default values.
  /// @param new_neural_network_pointer Pointer to a neural network object.
  /// @param new_data_set_pointer Pointer to a data set object.

  MultiError::MultiError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
    : LossIndex(new_neural_network_pointer, new_data_set_pointer)
  {
    set_default();
  }


  /// Destructor.

  MultiError::~MultiError()
  {
  }

  /// Returns the normalization coefficient.

  type MultiError::get_normalization_coefficient() const
  {
    return normalization_coefficient;
  }


  /// Returns the selection normalization coefficient.

  type MultiError::get_selection_normalization_coefficient() const
  {
    return selection_normalization_coefficient;
  }


  ///
  /// \brief set_data_set_pointer
  /// \param new_data_set_pointer

  void MultiError::set_data_set_pointer(DataSet* new_data_set_pointer)
  {
    data_set_pointer = new_data_set_pointer;

    if (neural_network_pointer->has_recurrent_layer() || neural_network_pointer->has_long_short_term_memory_layer())
    {
      set_time_series_normalization_coefficient();
    }
    else
    {
      set_normalization_coefficient();
    }
  }


  /// Sets the normalization coefficient from training samples.
  /// This method calculates the normalization coefficient of the data_set.

  void MultiError::set_normalization_coefficient()
  {
    // Data set

    const Tensor<type, 1> targets_mean = data_set_pointer->calculate_used_targets_mean();

    // Targets matrix

    const Tensor<type, 2> targets = data_set_pointer->get_target_data();

    // Normalization coefficient

    normalization_coefficient = calculate_normalization_coefficient(targets, targets_mean);
  }


  /// @todo What is targets_t1 ???

  void MultiError::set_time_series_normalization_coefficient()
  {
    //Targets matrix

    const Tensor<type, 2> targets = data_set_pointer->get_target_data();

    const Index rows = targets.dimension(0) - 1;
    const Index columns = targets.dimension(1);

    Tensor<type, 2> targets_t(rows, columns);
    Tensor<type, 2> targets_t_1(rows, columns);

    for (Index i = 0; i < columns; i++)
    {
      memcpy(targets_t_1.data() + targets_t_1.dimension(0) * i,
        targets.data() + targets.dimension(0) * i,
        static_cast<size_t>(rows) * sizeof(type));
    }

    for (Index i = 0; i < columns; i++)
    {
      memcpy(targets_t.data() + targets_t.dimension(0) * i,
        targets.data() + targets.dimension(0) * i + 1,
        static_cast<size_t>(rows) * sizeof(type));
    }

    //Normalization coefficient

    normalization_coefficient = calculate_time_series_normalization_coefficient(targets_t_1, targets_t);
  }


  type MultiError::calculate_time_series_normalization_coefficient(const Tensor<type, 2>& targets_t_1,
    const Tensor<type, 2>& targets_t) const
  {
#ifdef OPENNN_DEBUG

    check();

    const Index target_t_1_samples_number = targets_t_1.dimension(0);
    const Index target_t_1_variables_number = targets_t_1.dimension(1);
    const Index target_t_samples_number = targets_t.dimension(0);
    const Index target_t_variables_number = targets_t.dimension(1);

    if (target_t_1_samples_number != target_t_samples_number || target_t_1_variables_number != target_t_variables_number)
    {
      ostringstream buffer;

      buffer << "OpenNN Exception: NormalizedquaredError class.\n"
        << "type calculate_time_series_normalization_coefficient(const Tensor<type, 2>& targets_t_1, const Tensor<type, 2>& targets_t) function.\n"
        << " The columns number of targets(" << target_t_variables_number << ") must be equal(" << target_t_1_variables_number << ").\n"
        << " The samples number of targets(" << target_t_1_samples_number << ") must be equal(" << target_t_samples_number << ").\n";

      throw logic_error(buffer.str());
    }
#endif

    const Index target_samples_number = targets_t_1.dimension(0);
    const Index target_varaibles_number = targets_t_1.dimension(1);

    type normalization_coefficient = 0;

    for (Index i = 0; i < target_samples_number; i++)
    {
      for (Index j = 0; j < target_varaibles_number; j++)
      {
        normalization_coefficient += (targets_t_1(i, j) - targets_t(i, j)) * (targets_t_1(i, j) - targets_t(i, j));
      }
    }

    return normalization_coefficient;
  }


  /// Sets the normalization coefficient.
  /// @param new_normalization_coefficient New normalization coefficient to be set.

  void MultiError::set_normalization_coefficient(const type& new_normalization_coefficient)
  {
    normalization_coefficient = new_normalization_coefficient;
  }


  /// Sets the normalization coefficient from selection samples.
  /// This method calculates the normalization coefficient of the data_set.

  void MultiError::set_selection_normalization_coefficient()
  {
    // Data set

    const Tensor<Index, 1> selection_indices = data_set_pointer->get_selection_samples_indices();

    const Index selection_samples_number = selection_indices.size();

    if (selection_samples_number == 0) return;

    const Tensor<type, 1> selection_targets_mean = data_set_pointer->calculate_selection_targets_mean();

    const Tensor<type, 2> targets = data_set_pointer->get_selection_target_data();

    // Normalization coefficient

    selection_normalization_coefficient = calculate_normalization_coefficient(targets, selection_targets_mean);
  }


  /// Sets the normalization coefficient from selection samples.
  /// @param new_normalization_coefficient New normalization coefficient to be set.

  void MultiError::set_selection_normalization_coefficient(const type& new_selection_normalization_coefficient)
  {
    selection_normalization_coefficient = new_selection_normalization_coefficient;
  }


  /// Sets the default values.

  void MultiError::set_default()
  {
    if (has_neural_network() && has_data_set() && !data_set_pointer->is_empty())
    {
      set_normalization_coefficient();
      set_selection_normalization_coefficient();
    }
    else
    {
      normalization_coefficient = NAN;
      selection_normalization_coefficient = NAN;
    }

    minkowski_parameter = 1.5;
  }


  /// Returns the normalization coefficient to be used for the loss of the error.
  /// This is measured on the training samples of the data set.
  /// @param targets Matrix with the targets values from data_set.
  /// @param targets_mean Vector with the means of the given targets.

  type MultiError::calculate_normalization_coefficient(const Tensor<type, 2>& targets,
    const Tensor<type, 1>& targets_mean) const
  {
#ifdef OPENNN_DEBUG

    check();

    const Index means_number = targets_mean.dimension(0);
    const Index targets_number = targets.dimension(1);

    if (targets_number != means_number)
    {
      ostringstream buffer;

      buffer << "OpenNN Exception: NormalizedquaredError class.\n"
        << "type calculate_normalization_coefficient(const Tensor<type, 2>& targets, const Tensor<type, 1>& targets_mean) function.\n"
        << " The columns number of targets(" << targets_number << ") must be equal(" << means_number << ").\n";

      throw logic_error(buffer.str());
    }
#endif

    const Index size = targets.dimension(0);

    type normalization_coefficient = 0;

    for (Index i = 0; i < size; i++)
    {
      const Tensor<type, 0> norm = (targets.chip(i, 0) - targets_mean).square().sum();

      normalization_coefficient += norm(0);
    }

    if (static_cast<type>(normalization_coefficient - 0) < numeric_limits<type>::min()) normalization_coefficient = 1;

    return normalization_coefficient;
  }

  /// Returns the Minkowski exponent value used to calculate the error.

  type MultiError::get_Minkowski_parameter() const
  {
    return minkowski_parameter;
  }

  /// Sets a new Minkowski exponent value to be used in order to calculate the error.
  /// The Minkowski R-value must be comprised between 1 and 2.
  /// @param new_Minkowski_parameter Minkowski exponent value.

  void MultiError::set_Minkowski_parameter(const type& new_Minkowski_parameter)
  {
    // Control sentence

    if (new_Minkowski_parameter < static_cast<Index>(1.0) || new_Minkowski_parameter > static_cast<type>(2.0))
    {
      ostringstream buffer;

      buffer << "OpenNN Error. MinkowskiError class.\n"
        << "void set_Minkowski_parameter(const type&) method.\n"
        << "The Minkowski parameter must be comprised between 1 and 2.\n";

      throw logic_error(buffer.str());
    }

    // Set Minkowski parameter

    minkowski_parameter = new_Minkowski_parameter;
  }








  void MultiError::setErrorMethods(const ErrorMethod& new_loss_reg, const ErrorMethod& new_loss_class)
  {
    lossReg = new_loss_reg;
    lossClass = new_loss_class;
  }

  void MultiError::setErrorMethods(const string& new_loss_reg, const string& new_loss_class)
  {
    if (new_loss_reg == "SumSquaredError")
    {
      lossReg = ErrorMethod::SUM_SQUARED_ERROR;
    }
    else if (new_loss_reg == "MeanSquaredError")
    {
      lossReg = ErrorMethod::MEAN_SQUARED_ERROR;
    }
    else if (new_loss_reg == "NormalizedSquaredError")
    {
      lossReg = ErrorMethod::NORMALIZED_SQUARED_ERROR;
    }
    else if (new_loss_reg == "MinkowskiError")
    {
      lossReg = ErrorMethod::MINKOWSKI_ERROR;
    }
    else
    {
      ostringstream buffer;

      buffer << "OpenNN Exception: MultiError class.\n"
        << "void setErrorMethods(const string&) method.\n"
        << "Unknown error method: " << new_loss_reg << ".\n";

      throw logic_error(buffer.str());
    }

    if (new_loss_class == "SumSquaredError")
    {
      lossClass = ErrorMethod::SUM_SQUARED_ERROR;
    }
    else if (new_loss_class == "MeanSquaredError")
    {
      lossClass = ErrorMethod::MEAN_SQUARED_ERROR;
    }
    else if (new_loss_class == "NormalizedSquaredError")
    {
      lossClass = ErrorMethod::NORMALIZED_SQUARED_ERROR;
    }
    else if (new_loss_class == "MinkowskiError")
    {
      lossClass = ErrorMethod::MINKOWSKI_ERROR;
    }
    else if (new_loss_class == "CrossEntropyError")
    {
      lossClass = ErrorMethod::CROSS_ENTROPY_ERROR;
    }
    else
    {
      ostringstream buffer;

      buffer << "OpenNN Exception: MultiError class.\n"
        << "void setErrorMethods(const string&) method.\n"
        << "Unknown error method: " << new_loss_class << ".\n";

      throw logic_error(buffer.str());
    }
  }


  void MultiError::calculate_error(const DataSetBatch& batch,
    const NeuralNetworkForwardPropagation& forward_propagation,
    LossIndexBackPropagation& back_propagation) const
  {
    LayerBackPropagation* output_layer_back_propagation = back_propagation.neural_network.layers(neural_network_pointer->get_trainable_layers_number() - 1);

    Layer* output_layer_pointer = output_layer_back_propagation->layer_pointer;

    if (output_layer_pointer->get_type() != Layer::Type::MultiPerceptron) {
      ostringstream buffer;

      buffer << "OpenNN Exception: MultiError class.\n"
        << "MultiError can only be used with MultiPerceptronLayer.\n";

      throw logic_error(buffer.str());
    }

    MultiPerceptronLayer* multi_perceptron_layer
      = static_cast<MultiPerceptronLayer*>(output_layer_pointer);

    Tensor<type, 2> errorsReg(batch.get_batch_size(), multi_perceptron_layer->getRegColCount());
    for (size_t i = 0; i < errorsReg.dimension(0); i++) {
      for (size_t j = 0; j < errorsReg.dimension(1); j++) {
        errorsReg(i, j) = back_propagation.errors(i, multi_perceptron_layer->getRegCols()[j]);
      }
    }
    //Tensor<type, 2> errorsReg = back_propagation.errors.slice(Eigen::array<Index, 2>({ 0, 0 }), Eigen::array<Index, 2>({ batch.get_batch_size(), multi_perceptron_layer->getRegCols() }));

    type error = 0;

    if (errorsReg.dimension(1) != 0) {
      switch (lossReg) {
      case ErrorMethod::SUM_SQUARED_ERROR:
      {
        Tensor<type, 0> sum_squared_error;
        sum_squared_error.device(*thread_pool_device) = errorsReg.contract(errorsReg, SSE);
        error += sum_squared_error(0);
        break;
      }

      case ErrorMethod::MEAN_SQUARED_ERROR:
      {
        Tensor<type, 0> sum_squared_error;
        sum_squared_error.device(*thread_pool_device) = errorsReg.contract(errorsReg, SSE);
        error += sum_squared_error(0) / static_cast<type>(batch.inputs_dimensions(0));
        break;
      }

      case ErrorMethod::NORMALIZED_SQUARED_ERROR:
      {
        Tensor<type, 0> sum_squared_error;
        sum_squared_error.device(*thread_pool_device) = errorsReg.contract(errorsReg, SSE);
        error += sum_squared_error(0) / (static_cast<type>(batch.get_batch_size()) / static_cast<type>(data_set_pointer->get_samples_number()) * normalization_coefficient);
        break;
      }

      case ErrorMethod::MINKOWSKI_ERROR:
      {
        Tensor<type, 0> minkowski_error;
        minkowski_error.device(*thread_pool_device)
          = (errorsReg.abs().pow(minkowski_parameter).sum()).pow(static_cast<type>(1.0) / minkowski_parameter);
        error += minkowski_error(0) / batch.get_batch_size();
        break;
      }
      }
    }

    for (size_t c = 0; c < multi_perceptron_layer->getCatCount(); c++) {
      Tensor<type, 2> errorsClass(batch.get_batch_size(), multi_perceptron_layer->getCatColCount()[c]);
      for (size_t i = 0; i < errorsClass.dimension(0); i++) {
        for (size_t j = 0; j < errorsClass.dimension(1); j++) {
          errorsClass(i, j) = back_propagation.errors(i, multi_perceptron_layer->getCatCols()[c][j]);
        }
      }
      //Tensor<type, 2> errorsClass = back_propagation.errors.slice(Eigen::array<Index, 2>({ 0, colOffset }), Eigen::array<Index, 2>({ batch.get_batch_size(), categorySize }));

      switch (lossClass) {
      case ErrorMethod::CROSS_ENTROPY_ERROR:
      {
        const Tensor<type, 2>& outputs =
          static_cast<MultiPerceptronLayerForwardPropagation*>(forward_propagation.layers(neural_network_pointer->get_last_trainable_layer_index()))->activations_class[c];

        TensorMap<Tensor<type, 2>> targets(batch.targets_data, batch.targets_dimensions(0), batch.targets_dimensions(1));
        Tensor<type, 2> targets_class(batch.get_batch_size(), multi_perceptron_layer->getCatColCount()[c]);
        for (size_t i = 0; i < targets_class.dimension(0); i++) {
          for (size_t j = 0; j < targets_class.dimension(1); j++) {
            targets_class(i, j) = targets(i, multi_perceptron_layer->getCatCols()[c][j]);
          }
        }
        //Tensor<type, 2> targets_class = targets.slice(Eigen::array<Index, 2>({ 0, colOffset }), Eigen::array<Index, 2>({ batch.get_batch_size(), categorySize }));

        Tensor<type, 0> cross_entropy_error;

        if (multi_perceptron_layer->getCatColCount()[c] == 1) {
          Tensor<type, 2> logOutputs = outputs.log();
          for (Index i = 0; i < logOutputs.dimension(0); i++) {
            for (Index j = 0; j < logOutputs.dimension(1); j++) {
              if (targets_class(i, j) == 0) {
                logOutputs(i, j) = 0;
              }
              else {
                logOutputs(i, j) *= targets_class(i, j);
              }
            }
          }

          Tensor<type, 2> logOutputsCompliment = (1 - outputs).log();
          for (Index i = 0; i < logOutputsCompliment.dimension(0); i++) {
            for (Index j = 0; j < logOutputsCompliment.dimension(1); j++) {
              if (1 - targets_class(i, j) == 0) {
                logOutputsCompliment(i, j) = 0;
              }
              else {
                logOutputsCompliment(i, j) *= (1 - targets_class(i, j));
              }
            }
          }

          cross_entropy_error.device(*thread_pool_device) = -(logOutputs).sum() - (logOutputsCompliment).sum();
        }
        else {
          Tensor<type, 2> logOutputs = outputs.log();
          for (Index i = 0; i < logOutputs.dimension(0); i++) {
            for (Index j = 0; j < logOutputs.dimension(1); j++) {
              if (targets_class(i, j) == 0) {
                logOutputs(i, j) = 0;
              }
              else {
                logOutputs(i, j) *= targets_class(i, j);
              }
            }
          }

          cross_entropy_error.device(*thread_pool_device) = -(logOutputs).sum();
        }

        error += cross_entropy_error() / static_cast<type>(batch.inputs_dimensions(0));
        break;
      }
      case ErrorMethod::SUM_SQUARED_ERROR:
      {
        Tensor<type, 0> sum_squared_error;
        sum_squared_error.device(*thread_pool_device) = errorsClass.contract(errorsClass, SSE);
        error += sum_squared_error(0);
        break;
      }

      case ErrorMethod::MEAN_SQUARED_ERROR:
      {
        Tensor<type, 0> sum_squared_error;
        sum_squared_error.device(*thread_pool_device) = errorsClass.contract(errorsClass, SSE);
        error += sum_squared_error(0) / static_cast<type>(batch.inputs_dimensions(0));
        break;
      }

      case ErrorMethod::NORMALIZED_SQUARED_ERROR:
      {
        Tensor<type, 0> sum_squared_error;
        sum_squared_error.device(*thread_pool_device) = errorsClass.contract(errorsClass, SSE);
        error += sum_squared_error(0) / (static_cast<type>(batch.get_batch_size()) / static_cast<type>(data_set_pointer->get_samples_number()) * normalization_coefficient);
        break;
      }

      case ErrorMethod::MINKOWSKI_ERROR:
      {
        Tensor<type, 0> minkowski_error;
        minkowski_error.device(*thread_pool_device)
          = (errorsClass.abs().pow(minkowski_parameter).sum()).pow(static_cast<type>(1.0) / minkowski_parameter);
        error += minkowski_error(0) / batch.get_batch_size();
        break;
      }
      }
    }

    back_propagation.error = error;
  }


  void MultiError::calculate_output_delta(const DataSetBatch& batch,
    NeuralNetworkForwardPropagation& forward_propagation,
    LossIndexBackPropagation& back_propagation) const
  {
#ifdef OPENNN_DEBUG

    check();

#endif

    LayerBackPropagation* output_layer_back_propagation = back_propagation.neural_network.layers(neural_network_pointer->get_trainable_layers_number() - 1);

    Layer* output_layer_pointer = output_layer_back_propagation->layer_pointer;

    if (output_layer_pointer->get_type() != Layer::Type::MultiPerceptron) {
      ostringstream buffer;

      buffer << "OpenNN Exception: MultiError class.\n"
        << "MultiError can only be used with MultiPerceptronLayer.\n";

      throw logic_error(buffer.str());
    }

    MultiPerceptronLayer* multi_perceptron_layer
      = static_cast<MultiPerceptronLayer*>(output_layer_pointer);

    MultiPerceptronLayerBackPropagation* multi_perceptron_layer_back_propagation
      = static_cast<MultiPerceptronLayerBackPropagation*>(output_layer_back_propagation);

    Tensor<type, 2> errorsReg(batch.get_batch_size(), multi_perceptron_layer->getRegColCount());
    for (size_t i = 0; i < errorsReg.dimension(0); i++) {
      for (size_t j = 0; j < errorsReg.dimension(1); j++) {
        errorsReg(i, j) = back_propagation.errors(i, multi_perceptron_layer->getRegCols()[j]);
      }
    }
    //Tensor<type, 2> errorsReg = back_propagation.errors.slice(Eigen::array<Index, 2>({ 0, 0 }), Eigen::array<Index, 2>({ batch.get_batch_size(), multi_perceptron_layer->getRegCols() }));

    switch (lossReg) {
    case ErrorMethod::SUM_SQUARED_ERROR:
      multi_perceptron_layer_back_propagation->delta_reg.device(*thread_pool_device) = static_cast<type>(2.0) * errorsReg;
      break;

    case ErrorMethod::MEAN_SQUARED_ERROR:
      multi_perceptron_layer_back_propagation->delta_reg.device(*thread_pool_device) = static_cast<type>(2.0) / static_cast<type>(batch.inputs_dimensions(0)) * errorsReg;
      break;

    case ErrorMethod::NORMALIZED_SQUARED_ERROR:
      multi_perceptron_layer_back_propagation->delta_reg.device(*thread_pool_device) = static_cast<type>(2.0) / (static_cast<type>(batch.get_batch_size()) / static_cast<type>(data_set_pointer->get_samples_number()) * normalization_coefficient) * errorsReg;
      break;

    case ErrorMethod::MINKOWSKI_ERROR:
    {
      const Tensor<type, 0> p_norm_derivative =
        (errorsReg.abs().pow(minkowski_parameter).sum().pow(static_cast<type>(1.0) / minkowski_parameter)).pow(minkowski_parameter - 1);

      if (abs(p_norm_derivative()) < numeric_limits<type>::min())
      {
        multi_perceptron_layer_back_propagation->delta_reg.setZero();
      }
      else
      {
        multi_perceptron_layer_back_propagation->delta_reg.device(*thread_pool_device)
          = errorsReg * (errorsReg.abs().pow(minkowski_parameter - 2));

        multi_perceptron_layer_back_propagation->delta_reg.device(*thread_pool_device) =
          (1 / type(batch.get_batch_size())) * multi_perceptron_layer_back_propagation->delta_reg / p_norm_derivative();
      }
      break;
    }
    }

    for (size_t c = 0; c < multi_perceptron_layer->getCatCount(); c++) {
      Tensor<type, 2> errorsClass(batch.get_batch_size(), multi_perceptron_layer->getCatColCount()[c]);
      for (size_t i = 0; i < errorsClass.dimension(0); i++) {
        for (size_t j = 0; j < errorsClass.dimension(1); j++) {
          errorsClass(i, j) = back_propagation.errors(i, multi_perceptron_layer->getCatCols()[c][j]);
        }
      }
      //Tensor<type, 2> errorsClass = back_propagation.errors.slice(Eigen::array<Index, 2>({ 0, colOffset }), Eigen::array<Index, 2>({ batch.get_batch_size(), categorySize }));

      switch (lossClass) {
      case ErrorMethod::CROSS_ENTROPY_ERROR:
      {
        const Tensor<type, 2>& outputs =
          static_cast<MultiPerceptronLayerForwardPropagation*>(forward_propagation.layers(neural_network_pointer->get_last_trainable_layer_index()))->activations_class[c];

        TensorMap<Tensor<type, 2>> targets(batch.targets_data, batch.targets_dimensions(0), batch.targets_dimensions(1));
        Tensor<type, 2> targets_class(batch.get_batch_size(), multi_perceptron_layer->getCatColCount()[c]);
        for (size_t i = 0; i < targets_class.dimension(0); i++) {
          for (size_t j = 0; j < targets_class.dimension(1); j++) {
            targets_class(i, j) = targets(i, multi_perceptron_layer->getCatCols()[c][j]);
          }
        }
        //Tensor<type, 2> targets_class = targets.slice(Eigen::array<Index, 2>({ 0, colOffset }), Eigen::array<Index, 2>({ batch.get_batch_size(), categorySize }));

        Tensor<type, 2> outputQuotients(outputs.dimension(0), outputs.dimension(1));
        for (size_t i = 0; i < outputQuotients.dimension(0); i++) {
          for (size_t j = 0; j < outputQuotients.dimension(1); j++) {
            if (targets_class(i, j) == 0) {
              outputQuotients(i, j) = 0;
            }
            else {
              outputQuotients(i, j) = targets_class(i, j) / outputs(i, j);
            }
          }
        }

        if (multi_perceptron_layer->getCatColCount()[c] == 1) {
          Tensor<type, 2> outputCounterpartQuotients(outputs.dimension(0), outputs.dimension(1));
          for (size_t i = 0; i < outputCounterpartQuotients.dimension(0); i++) {
            for (size_t j = 0; j < outputCounterpartQuotients.dimension(1); j++) {
              if (1 - targets_class(i, j) == 0) {
                outputCounterpartQuotients(i, j) = 0;
              }
              else {
                outputCounterpartQuotients(i, j) = (1 - targets_class(i, j)) / (1 - outputs(i, j));
              }
            }
          }

          multi_perceptron_layer_back_propagation->delta_class[c].device(*thread_pool_device)
            = static_cast<type>(1) / static_cast<type>(batch.inputs_dimensions(0)) *
            (static_cast<type>(-1) * outputQuotients + outputCounterpartQuotients);
        }
        else {
          multi_perceptron_layer_back_propagation->delta_class[c].device(*thread_pool_device)
            = static_cast<type>(1) / static_cast<type>(batch.inputs_dimensions(0)) * -outputQuotients;
        }
        break;
      }
      case ErrorMethod::SUM_SQUARED_ERROR:
        multi_perceptron_layer_back_propagation->delta_class[c].device(*thread_pool_device) = static_cast<type>(2.0) * errorsClass;
        break;

      case ErrorMethod::MEAN_SQUARED_ERROR:
        multi_perceptron_layer_back_propagation->delta_class[c].device(*thread_pool_device) = static_cast<type>(2.0) / static_cast<type>(batch.inputs_dimensions(0)) * errorsClass;
        break;

      case ErrorMethod::NORMALIZED_SQUARED_ERROR:
        multi_perceptron_layer_back_propagation->delta_class[c].device(*thread_pool_device) = static_cast<type>(2.0) / (static_cast<type>(batch.get_batch_size()) / static_cast<type>(data_set_pointer->get_samples_number()) * normalization_coefficient) * errorsClass;
        break;

      case ErrorMethod::MINKOWSKI_ERROR:
      {
        const Tensor<type, 0> p_norm_derivative =
          (errorsClass.abs().pow(minkowski_parameter).sum().pow(static_cast<type>(1.0) / minkowski_parameter)).pow(minkowski_parameter - 1);

        if (abs(p_norm_derivative()) < numeric_limits<type>::min())
        {
          multi_perceptron_layer_back_propagation->delta_class[c].setZero();
        }
        else
        {
          multi_perceptron_layer_back_propagation->delta_class[c].device(*thread_pool_device)
            = errorsClass * (errorsClass.abs().pow(minkowski_parameter - 2));

          multi_perceptron_layer_back_propagation->delta_class[c].device(*thread_pool_device) =
            (1 / type(batch.get_batch_size())) * multi_perceptron_layer_back_propagation->delta_class[c] / p_norm_derivative();
        }
        break;
      }
      }
    }
  }


  /// Returns a string with the name of the sum squared error loss type, "SUM_SQUARED_ERROR".

  string MultiError::get_error_type() const
  {
    return "MULTI_ERROR";
  }


  /// Returns a string with the name of the sum squared error loss type in text format.

  string MultiError::get_error_type_text() const
  {
    return "Multi error";
  }


  /// Serializes the cross entropy error object into a XML document of the TinyXML library without keep the DOM tree in memory.
  /// See the OpenNN manual for more information about the format of this document

  void MultiError::write_XML(tinyxml2::XMLPrinter& file_stream) const
  {
    // Error type

    file_stream.OpenElement("MultiError");

    file_stream.CloseElement();
  }


  /// Loads a sum squared error object from a XML document.
  /// @param document TinyXML document containing the members of the object.

  void MultiError::from_XML(const tinyxml2::XMLDocument& document)
  {
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("MultiError");

    if (!root_element)
    {
      ostringstream buffer;

      buffer << "OpenNN Exception: MultiError class.\n"
        << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
        << "Multi element is nullptr.\n";

      throw logic_error(buffer.str());
    }
  }
}