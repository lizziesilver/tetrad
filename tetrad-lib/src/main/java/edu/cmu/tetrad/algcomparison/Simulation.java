package edu.cmu.tetrad.algcomparison;

import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.graph.Graph;

import java.util.Map;

/**
 * Created by jdramsey on 6/4/16.
 */
public interface Simulation {
    void simulate(Map<String, Number> parameters);

    Graph getDag();

    DataSet getData();

    String toString();

    boolean isMixed();
}