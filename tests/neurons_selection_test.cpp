//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R O N S   S E L E C T I O N   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com                                           

#include "neurons_selection_test.h"


NeuronsSelectionTest::NeuronsSelectionTest() : UnitTesting()
{
}


NeuronsSelectionTest::~NeuronsSelectionTest()
{
}


void NeuronsSelectionTest::test_constructor()
{
    cout << "test_constructor\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TrainingStrategy training_strategy(&neural_network, &data_set);

    GrowingNeurons io1(&training_strategy);

    assert_true(io1.has_training_strategy(), LOG);

    GrowingNeurons io2;

    assert_true(!io2.has_training_strategy(), LOG);
}


void NeuronsSelectionTest::test_destructor()
{
    cout << "tes_destructor\n";

    GrowingNeurons* io = new GrowingNeurons;

    delete io;
}


void NeuronsSelectionTest::test_get_training_strategy_pointer()
{
    cout << "test_get_training_strategy_pointer\n";

    TrainingStrategy training_strategy;

    GrowingNeurons io(&training_strategy);

    assert_true(io.get_training_strategy_pointer() != nullptr, LOG);
}


void NeuronsSelectionTest::test_set_training_strategy_pointer()
{
    cout << "test_set_training_strategy_pointer\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TrainingStrategy training_strategy(&neural_network, &data_set);

    GrowingNeurons in;

    in.set_training_strategy_pointer(&training_strategy);

    assert_true(in.get_training_strategy_pointer() != nullptr, LOG);
}


void NeuronsSelectionTest::run_test_case()
{
    cout << "Running neurons selection algorithm test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Get methods

    test_get_training_strategy_pointer();

    // Set methods

    test_set_training_strategy_pointer();

    cout << "End of neurons selection algorithm test case.\n\n";
}

