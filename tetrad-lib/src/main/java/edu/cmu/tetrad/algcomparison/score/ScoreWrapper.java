package edu.cmu.tetrad.algcomparison.score;

import edu.cmu.tetrad.algcomparison.Parameters;
import edu.cmu.tetrad.data.DataSet;
import edu.cmu.tetrad.data.DataType;
import edu.cmu.tetrad.search.Score;

import java.util.List;

/**
 * Interface that algorithms must implement.
 * @author jdramsey
 */
public interface ScoreWrapper {

    /**
     * Returns true iff x and y are independent conditional on z for the given data set.
     * @param dataSet The data set to test independence against.
     * @param parameters The paramters of the test.
     * @return True iff independence holds.
     */
    Score getScore(DataSet dataSet, Parameters parameters);

    /**
     * Returns a short of this independence test.
     * @return This description.
     */
    String getDescription();

    /**
     * Returns the data type that the search requires, whether continuous, discrete, or mixed.
     * @return This type.
     */
    DataType getDataType();

    /**
     * Returns the parameters that this search uses.
     * @return A list of String names of parameters.
     */
    List<String> getParameters();
}
