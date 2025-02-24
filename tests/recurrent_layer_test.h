//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   L A Y E R   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef RECURRENTLAYERTEST_H
#define RECURRENTLAYERTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"

class RecurrentLayerTest : public UnitTesting
{

public:

    explicit RecurrentLayerTest();

    virtual ~RecurrentLayerTest();

    // Constructor and destructor methods

    void test_constructor();

    void test_destructor();

    // Inputs and neurons

    void test_is_empty();

    // Parameters

    void test_calculate_activations_derivatives();

    // Forward propagate

    void test_forward_propagate();

    // Forward propagation

    void test_calculate_outputs();

    // Unit testing methods

    void run_test_case();

private:

    Index inputs_number;
    Index neurons_number;
    Index samples_number;

    RecurrentLayer recurrent_layer;
    RecurrentLayerForwardPropagation recurrent_layer_forward_propagation;

    NumericalDifferentiation numerical_differentiation;
};


#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
