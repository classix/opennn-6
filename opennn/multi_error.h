#ifndef MULTIERROR_H
#define MULTIERROR_H

// System includes

#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>
#include <string>
#include <limits>

// OpenNN includes

#include "config.h"
#include "loss_index.h"
#include "data_set.h"

namespace opennn
{
  enum ErrorMethod
  {
    SUM_SQUARED_ERROR,
    MEAN_SQUARED_ERROR,
    NORMALIZED_SQUARED_ERROR,
    MINKOWSKI_ERROR,
    //WEIGHTED_SQUARED_ERROR,
    CROSS_ENTROPY_ERROR,
  };

  /// This class represents the sum squared peformance term functional. 

  ///
  /// This is used as the error term in data modeling problems, such as function regression, 
  /// classification or time series prediction.

  class MultiError : public LossIndex
  {

  public:

    // DEFAULT CONSTRUCTOR

    explicit MultiError();

    explicit MultiError(NeuralNetwork*, DataSet*);

    virtual ~MultiError();

    void setErrorMethods(const ErrorMethod&, const ErrorMethod&);
    void setErrorMethods(const string&, const string&);

    // Get methods

    type get_normalization_coefficient() const;
    type get_selection_normalization_coefficient() const;

    type get_Minkowski_parameter() const;

    // Set methods

    void set_normalization_coefficient();
    void set_normalization_coefficient(const type&);

    void set_time_series_normalization_coefficient();

    void set_selection_normalization_coefficient();
    void set_selection_normalization_coefficient(const type&);

    void set_Minkowski_parameter(const type&);

    void set_default();

    void set_data_set_pointer(DataSet* new_data_set_pointer);

    // Normalization coefficients 

    type calculate_normalization_coefficient(const Tensor<type, 2>&, const Tensor<type, 1>&) const;

    type calculate_time_series_normalization_coefficient(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

    // Back propagation

    void calculate_error(const DataSetBatch&,
      const NeuralNetworkForwardPropagation&,
      LossIndexBackPropagation&) const;

    void calculate_output_delta(const DataSetBatch&,
      NeuralNetworkForwardPropagation&,
      LossIndexBackPropagation&) const;

    // Back propagation LM

    // Serialization methods

    string get_error_type() const;
    string get_error_type_text() const;

    void from_XML(const tinyxml2::XMLDocument&);

    void write_XML(tinyxml2::XMLPrinter&) const;

  private:

    type normalization_coefficient = NAN;

    type selection_normalization_coefficient = NAN;

    type minkowski_parameter;

    ErrorMethod lossReg = ErrorMethod::NORMALIZED_SQUARED_ERROR;
    ErrorMethod lossClass = ErrorMethod::CROSS_ENTROPY_ERROR;

    Index regCols;
    vector<Index> categorySizes;

#ifdef OPENNN_CUDA
#include "../../opennn-cuda/opennn_cuda/sum_squared_error_cuda.h"
#endif

  };

}

#endif