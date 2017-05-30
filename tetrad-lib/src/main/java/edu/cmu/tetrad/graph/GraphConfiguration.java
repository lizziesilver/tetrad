package edu.cmu.tetrad.graph;

import edu.cmu.tetrad.data.ContinuousVariable;
import edu.cmu.tetrad.graph.GraphUtils;
import edu.cmu.tetrad.search.SearchGraphUtils;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;


/**
 * Created by Lizzie Silver on 4/1/17.
 * This class supports Greedy Equivalence Search with Transfer (GEST).
 * A "configuration" of graphs is just a set of graphs over the same variables, as described in Niculescu-Mizil &
 * Caruana (2007): "Inductive transfer for bayesian network structure learning." AISTATS, pages 339–346.
 */
public class GraphConfiguration implements GraphGroup {

    private List<Graph> graphList;
    private List<Node> variables;
    private PrintStream out = System.out;
    private int numGraphs;

    //===========================CONSTRUCTORS=============================//

    /**
     * Creates a GraphConfiguration from an existing list of graphs
     */
    public GraphConfiguration(List<Graph> graphList) {

        this.variables = graphList.get(0).getNodes();

        // test that the variable sets are the same for all graphs in the list
        for (int i = 0; i < graphList.size(); i++){
            List<Node> variablesi = graphList.get(i).getNodes();
            if (!variables.equals(variablesi)) {
                throw new IllegalArgumentException("variable lists differ between graphs in graph list");
            }
        }

        this.graphList = graphList;

        this.numGraphs = graphList.size();

    }

    /**
     * Creates a GraphConfiguration from a list of variables. The graphs will be empty.
     * Number of graphs must be specified.
     */
    public GraphConfiguration(List<Node> variables, int numGraphs) {
        this.variables = variables;

        this.numGraphs = numGraphs;

        this.graphList = new ArrayList<>();

        for (int i = 0; i < numGraphs; i++) {
            graphList.add(new EdgeListGraphSingleConnections(variables));
        }
    }

    //==========================PUBLIC METHODS==========================//

    public Graph getGraph(int g) {
        return graphList.get(g);
    }

    public void setGraph(int i, Graph g) {
        graphList.set(i, g);
    }

    public List<Node> getVariables() {
        return variables;
    }

    public void addGraph(Graph g) {
        graphList.add(g);
        numGraphs++;
    }

    public boolean equals(Object o) {
        if (o == null) return false;
        GraphConfiguration config2 = (GraphConfiguration) o;
        if (config2.getNumGraphs() != getNumGraphs()) return false;

        for (int i = 0; i < getNumGraphs(); i++) {
            Graph graph1 = getGraph(i);
            Graph graph2 = config2.getGraph(i);

            if (!graph1.equals(graph2)) return false;
        }

        return true;
    }

    public int getNumGraphs() {
        return numGraphs;
    }

    /**
     * Calculates the number of non-shared adjacencies across the whole graphList
     */
    public int getDistance() {
        int distance = 0;

        if (numGraphs > 1) {
            for (int i = 0; i < numGraphs - 1; i++) {
                for (int j = i + 1; j < numGraphs; j++) {
                    GraphUtils.GraphComparison graphComparison = SearchGraphUtils.getGraphComparison(getGraph(i),
                            getGraph(j));
                    distance = distance + graphComparison.getAdjFn() + graphComparison.getAdjFp();
                }
            }
        }

        return distance;
    }

    /**
     * For a given pair of nodes, calculates the number of non-shared adjacencies
     * in a partial configuration
     */
    public int getPartialDistanceNodePair(Node X, Node Y, int c) {
        if (c > numGraphs | c < 1) {
            throw new IllegalArgumentException("length of a partial configuration must be between 1 and numGraphs");
        }

        int adjacent = 0;
        int nonAdjacent = 0;

        for (int i = 0; i < c; i++){
            if (getGraph(i).isAdjacentTo(X, Y)) {
                adjacent ++;
            } else {
                nonAdjacent ++;
            }
        }

        return adjacent * nonAdjacent;
    }

    /**
     * For a given pair of nodes, calculates the number of non-shared adjacencies
     * across the whole graphList
     */
    public int getDistanceNodePair(Node X, Node Y) { return getPartialDistanceNodePair(X, Y, numGraphs); }

    public String toString() {
        String s = "";

        for (int i = 0; i < numGraphs; i++) {
            s = s + "Graph " + (i+1) +": \n" + getGraph(i).toString();
        }

        s = s.trim();

        return s;
    }

    //todo write hashcode function
    //==========================PRIVATE METHODS==========================//


    // TODO: figure out if I can modify a single graph through this class

}
