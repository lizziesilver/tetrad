package edu.cmu.tetrad.algcomparison.algorithm;

import edu.cmu.tetrad.graph.GraphConfiguration;
import edu.cmu.tetrad.data.DataModel;
import edu.cmu.tetrad.util.Parameters;
import edu.cmu.tetrad.algcomparison.utils.HasParameters;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DataType;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.util.TetradSerializable;

import java.util.List;

/**
 * Implements an algorithm that takes multiple datasets and returns multiple graphs, learning the graphs simultaneously.
 *
 * Created by lizzie on 4/28/17.
 */
public interface MultiTaskAlgorithm extends Algorithm {

    GraphConfiguration search(List<DataModel> dataSetList, Parameters parameters);

    GraphConfiguration getComparisonGraph(GraphConfiguration graphConfiguration);

}
