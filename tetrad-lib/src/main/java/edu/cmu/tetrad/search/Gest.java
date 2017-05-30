///////////////////////////////////////////////////////////////////////////////
// For information as to what this class does, see the Javadoc, below.       //
// Copyright (C) 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,       //
// 2007, 2008, 2009, 2010, 2014, 2015 by Peter Spirtes, Richard Scheines, Joseph   //
// Ramsey, and Clark Glymour.                                                //
//                                                                           //
// This program is free software; you can redistribute it and/or modify      //
// it under the terms of the GNU General Public License as published by      //
// the Free Software Foundation; either version 2 of the License, or         //
// (at your option) any later version.                                       //
//                                                                           //
// This program is distributed in the hope that it will be useful,           //
// but WITHOUT ANY WARRANTY; without even the implied warranty of            //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             //
// GNU General Public License for more details.                              //
//                                                                           //
// You should have received a copy of the GNU General Public License         //
// along with this program; if not, write to the Free Software               //
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA //
///////////////////////////////////////////////////////////////////////////////

package edu.cmu.tetrad.search;

import edu.cmu.tetrad.data.*;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.util.ChoiceGenerator;
import edu.cmu.tetrad.util.ForkJoinPoolInstance;
import edu.cmu.tetrad.util.TaskManager;
import edu.cmu.tetrad.util.TetradLogger;

import java.io.PrintStream;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.ConcurrentHashMap;


/**
 * GesSearch is an implementation of the GES algorithm, as specified in Chickering (2002) "Optimal structure
 * identification with greedy search" Journal of Machine Learning Research. It works for both BayesNets and SEMs.
 * <p>
 * Some code optimization could be done for the scoring part of the graph for discrete models (method scoreGraphChange).
 * Some of Andrew Moore's approaches for caching sufficient statistics, for instance.
 * <p>
 * To speed things up, it has been assumed that variables X and Y with zero correlation do not correspond to edges in
 * the graph. This is a restricted form of the heuristicSpeedup assumption, something GES does not assume. This
 * the graph. This is a restricted form of the heuristicSpeedup assumption, something GES does not assume. This
 * heuristicSpeedup assumption needs to be explicitly turned on using setHeuristicSpeedup(true).
 * <p>
 * A number of other optimizations were added 5/2015. See code for details.
 *
 * @author Ricardo Silva, Summer 2003
 * @author Joseph Ramsey, Revisions 5/2015
 */
public final class Gest implements GraphGroupSearch {

    /**
     * Internal.
     */
    private enum Mode {
        allowUnfaithfulness, heuristicSpeedup, coverNoncolliders
    }

    /**
     * Specification of forbidden and required edges. TODO: get rid of this or adapt it
     */
    private IKnowledge knowledge = new Knowledge2();

    /**
     * List of variables in the data set, in order.
     */
    private List<Node> variables;

    /**
     * The true graph, if known. If this is provided, asterisks will be printed out next to false positive added edges
     * (that is, edges added that aren't adjacencies in the true graph).
     * TODO: get rid of this or adapt it
     */
    private Graph trueGraph;

    /**
     * An initial graph to start from.
     * TODO: get rid of this or adapt it
     */
    private Graph initialGraph;

    /**
     * If non-null, edges not adjacent in this graph will not be added.
     * TODO: decide whether to remove this functionality, or else check it is consistent in GEST
     */
    private Graph boundGraph = null;

    /**
     * Elapsed time of the most recent search.
     */
    private long elapsedTime;

    /**
     * A bound on cycle length.
     */
    private int cycleBound = -1;

    /**
     * The list of score objects for each task graph.
     */
    private List<Score> scoreList;

    /**
     * The logger for this class. The config needs to be set.
     */
    private TetradLogger logger = TetradLogger.getInstance();

    /**
     * True if verbose output should be printed.
     */
    private boolean verbose = false;

    /**
     * Maps an ordered pair of nodes <i, j> to the set of arrows. The list has one entry for each task graph.
     */
    private List<Map<OrderedPair<Node>, Set<Arrow>>> lookupArrowsList = null;

    /**
     * A utility map to help with orientation. The list has one entry for each task graph.
     */
    private List<Map<Node, Set<Node>>> neighborsList = null;

    /**
     * Map from variables to their column indices in the data sets.
     */
    private ConcurrentMap<Node, Integer> hashIndices;

    // The static ForkJoinPool instance.
    private ForkJoinPool pool = ForkJoinPoolInstance.getInstance().getPool();

    // A running tally of the total score for the entire graph configuration.
    private double totalScore;

    /**
     * This is the minimum score an arrow can have and not be removed from the list of arrows.
     * It is calculated using the arrow's "bump", i.e. the amount that arrow improves the
     * likelihood, and the maximum possible transfer penalty that could be applied. E.g. if the
     * max possible transfer penalty is 3, an arrow can actually reduce the likelihood by up to
     * -3 and still be considered for adding to the graph, because depending on the edges in other
     * graphs, the arrow might reduce the transfer penalty enough to compensate for how much it
     * hurts the likelihood.
     *
     * If the transfer weight is zero, the bumpMin is zero, as in FGES.
     */
    private double bumpMin;

    // A graph where X--Y means that X and Y have non-zero total effect on one another.
    // TODO figure out what this does and whether I need it.
    private List<Graph> effectEdgesGraphList;

    // The minimum number of operations to do before parallelizing.
    private final int minChunk = 100;

    // Where printed output is sent.
    private PrintStream out = System.out;

    // A initial adjacencies graph.
    // TODO figure out what this does and whether I need it. I think I can remove it? Or if I extend this method to
    // take prior knowledge, maybe I need to turn it into a list of adjacency graphs?
    private Graph adjacencies = null;

    // The graph configuration being constructed.
    private GraphConfiguration graphConfiguration;

    /**
     * The weight controlling how strong the transfer learning penalty is.
     */
    private double transferPenalty;

    /**
     * Weight the transfer penalty by log(log(sampleSize)? This allows the penalty to grow with sample size, but
     * slower than the likelihood or the sparsity penalty of BIC, preserving score consistency.
     */
    private boolean weightTransferBySample;

    // the number of graphs in the configuration
    private int numGraphs;

    // A tie-breaker when comparing arrows. The ordering doesn't matter; it just has to be transitive.
    private int arrowIndex = 0;

    // The final totalScore after search.
    private double modelScore;

    // Internal.
    private Mode mode = Mode.heuristicSpeedup;

    /**
     * True if one-edge faithfulness is assumed. Speeds the algorithm up.
     */
    private boolean faithfulnessAssumed = true;

    // Bounds the degree of the graph.
    private int maxDegree = -1;

    // True if the first step of adding an edge to an empty graph should be scored in both directions
    // for each edge with the maximum score chosen.
    private boolean symmetricFirstStep = false;

    final int maxThreads = ForkJoinPoolInstance.getInstance().getPool().getParallelism();

    //===========================CONSTRUCTORS=============================//

    /**
     * Construct a Score and pass it in here. The totalScore should return a
     * positive value in case of conditional dependence and a negative
     * values in case of conditional independence. See Chickering (2002),
     * locally consistent scoring criterion.
     *
     * @param scoreList
     */
    public Gest(List<Score> scoreList) {
        if (scoreList == null) throw new NullPointerException();
        setScore(scoreList);
        this.numGraphs = scoreList.size();
        this.graphConfiguration = new GraphConfiguration(getVariables(), numGraphs);
        this.transferPenalty = 1;
        this.weightTransferBySample = false;
        this.bumpMin = 0;
    }

    public Gest(List<Score> scoreList, double transferPenalty, boolean weightTransferBySample, boolean bumpMinTransfer) {
        if (scoreList == null) throw new NullPointerException();
        setScore(scoreList);
        this.numGraphs = scoreList.size();
        this.graphConfiguration = new GraphConfiguration(getVariables(), numGraphs);
        this.weightTransferBySample = weightTransferBySample;
        if (bumpMinTransfer) {
            this.bumpMin = - (numGraphs / 2) * ((numGraphs + 1) / 2) * transferPenalty;
        } else {
            this.bumpMin = 0;
        }

        if (weightTransferBySample) {
            double sampleTransferPenalty = 0;
            for (int i = 0; i < numGraphs; i++) {sampleTransferPenalty += Math.log(scoreList.get(i).getSampleSize() + 1);}

            sampleTransferPenalty = Math.log(sampleTransferPenalty + 1);

            this.transferPenalty = transferPenalty * sampleTransferPenalty;
        } else {
            this.transferPenalty = transferPenalty;
        }
    }


    //==========================PUBLIC METHODS==========================//

    /**
     * Set to true if it is assumed that all path pairs with one length 1 path do not cancel.
     */
    public void setFaithfulnessAssumed(boolean faithfulnessAssumed) {
        this.faithfulnessAssumed = faithfulnessAssumed;
    }

    /**
     * @return true if it is assumed that all path pairs with one length 1 path do not cancel.
     */
    public boolean isFaithfulnessAssumed() {
        return faithfulnessAssumed;
    }

    /**
     * Greedy equivalence search: Start from the empty graph, add edges till model is significant. Then start deleting
     * edges till a minimum is achieved.
     *
     * @return the resulting Pattern.
     */
    public GraphConfiguration search() {
        lookupArrowsList = new ArrayList<>();
        for (int graphIndex = 0; graphIndex < numGraphs; graphIndex++) {
            lookupArrowsList.add(new ConcurrentHashMap<OrderedPair<Node>, Set<Arrow>>());
        }

        final List<Node> nodes = new ArrayList<>(variables);
        graphConfiguration = new GraphConfiguration(nodes, numGraphs);

        // TODO see if I need initialGraph, required adjacencies or knowledge; if I do, adapt for multiple graphs
        /*if (adjacencies != null) {
            adjacencies = GraphUtils.replaceNodes(adjacencies, nodes);
        }

        if (initialGraph != null) {
            graph = new EdgeListGraphSingleConnections(initialGraph);
            graph = GraphUtils.replaceNodes(graph, nodes);
        }

        addRequiredEdges(graph;
        */

        if (faithfulnessAssumed) {
            initializeForwardEdgesFromEmptyGraph(getVariables());

            // Do forward search.
            this.mode = Mode.heuristicSpeedup;
            fest();
            best();

            this.mode = Mode.coverNoncolliders;
            initializeTwoStepEdges(getVariables());
            fest();
            best();
        } else {
            initializeForwardEdgesFromEmptyGraph(getVariables());

            // Do forward search.
            this.mode = Mode.heuristicSpeedup;
            fest();
            best();

            this.mode = Mode.allowUnfaithfulness;
            initializeForwardEdgesFromExistingGraph(getVariables());
            fest();
            best();
        }

        long start = System.currentTimeMillis();
        totalScore = 0.0;

        long endTime = System.currentTimeMillis();
        this.elapsedTime = endTime - start;
        this.logger.log("graph", "\nReturning this graph: " + graphConfiguration);

        this.logger.log("info", "Elapsed time = " + (elapsedTime) / 1000. + " s");
        this.logger.flush();

        this.modelScore = totalScore;

        return graphConfiguration;
    }

    /**
     * @return the background knowledge.
     */

    public IKnowledge getKnowledge() {
        return knowledge;
    }

    /**
     * Sets the background knowledge.
     *
     * @param knowledge the knowledge object, specifying forbidden and required edges.
     */
    public void setKnowledge(IKnowledge knowledge) {
        if (knowledge == null) throw new NullPointerException();
        this.knowledge = knowledge;
    }

    public long getElapsedTime() {
        return elapsedTime;
    }

    /**
     * If the true graph is set, askterisks will be printed in log output for the true edges.
     */
    public void setTrueGraph(Graph trueGraph) {
        this.trueGraph = trueGraph;
    }

    /**
     * @return the totalScore of the given DAG, up to a constant.
     */
    public double getScore(GraphConfiguration graphConfiguration, int graphIndex) {
        return scoreDag(graphConfiguration, graphIndex);
    }

    /**
     * @return the initial graph for the search. The search is initialized to this graph and
     * proceeds from there.
     */
    public Graph getInitialGraph() {
        return initialGraph;
    }

    /**
     * Sets the initial graph.
     */
    public void setInitialGraph(Graph initialGraph) {
        if (initialGraph != null) {
            initialGraph = GraphUtils.replaceNodes(initialGraph, variables);

            if (verbose) {
                out.println("Initial graph variables: " + initialGraph.getNodes());
                out.println("Data set variables: " + variables);
            }

            if (!new HashSet<>(initialGraph.getNodes()).equals(new HashSet<>(variables))) {
                throw new IllegalArgumentException("Variables aren't the same.");
            }
        }

        this.initialGraph = initialGraph;
    }

    /**
     * Sets whether verbose output should be produced.
     */
    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }

    /**
     * Sets the output stream that output (except for log output) should be sent to.
     * By detault System.out.
     */
    public void setOut(PrintStream out) {
        this.out = out;
    }

    /**
     * @return the output stream that output (except for log output) should be sent to.
     */
    public PrintStream getOut() {
        return out;
    }

    /**
     * @return the set of preset adjacenies for the algorithm; edges not in this adjacencies graph
     * will not be added.
     */
    public Graph getAdjacencies() {
        return adjacencies;
    }

    /**
     * Sets the set of preset adjacenies for the algorithm; edges not in this adjacencies graph
     * will not be added.
     */
    public void setAdjacencies(Graph adjacencies) {
        this.adjacencies = adjacencies;
    }

    /**
     * A bound on cycle length.
     */
    public int getCycleBound() {
        return cycleBound;
    }

    /**
     * A bound on cycle length.
     *
     * @param cycleBound The bound, >= 1, or -1 for unlimited.
     */
    public void setCycleBound(int cycleBound) {
        if (!(cycleBound == -1 || cycleBound >= 1))
            throw new IllegalArgumentException("Cycle bound needs to be -1 or >= 1: " + cycleBound);
        this.cycleBound = cycleBound;
    }

    /**
     * Creates a new processors pool with the specified number of threads.
     */
    public void setParallelism(int numProcessors) {
        this.pool = new ForkJoinPool(numProcessors);
    }

    /**
     * If non-null, edges not adjacent in this graph will not be added.
     */
    public void setBoundGraph(Graph boundGraph) {
        this.boundGraph = GraphUtils.replaceNodes(boundGraph, getVariables());
    }

    /**
     * The maximum of parents any nodes can have in output pattern.
     *
     * @return -1 for unlimited.
     */
    public int getMaxDegree() {
        return maxDegree;
    }

    /**
     * The maximum of parents any nodes can have in output pattern.
     *
     * @param maxDegree -1 for unlimited.
     */
    public void setMaxDegree(int maxDegree) {
        if (maxDegree < -1) throw new IllegalArgumentException();
        this.maxDegree = maxDegree;
    }

    public boolean isSymmetricFirstStep() {
        return symmetricFirstStep;
    }

    public void setSymmetricFirstStep(boolean symmetricFirstStep) {
        this.symmetricFirstStep = symmetricFirstStep;
    }


    //===========================PRIVATE METHODS========================//

    //Sets the discrete scoring function to use.
    private void setScore(List<Score> scoreList) {
        this.scoreList = scoreList;

        this.variables = new ArrayList<>();

        for (Node node : scoreList.get(0).getVariables()) {
            if (node.getNodeType() == NodeType.MEASURED) {
                this.variables.add(node);
            }
        }

        // test that the variable sets are the same for all scores in the list
        for (int i = 0; i < scoreList.size(); i++){
            List<Node> variablesi = scoreList.get(i).getVariables();
            if (!variables.equals(variablesi)) {
                throw new IllegalArgumentException("variable lists differ between scores in score list");
            }
        }

        buildIndexing(scoreList.get(0).getVariables());

        //todo check maxdegrees are same for all scores
        this.maxDegree = scoreList.get(0).getMaxDegree();
    }

    final int[] count = new int[1];

    public int getMinChunk(int n) {
        return Math.max(n / maxThreads, minChunk);
    }

    class NodeTaskEmptyGraph extends RecursiveTask<Boolean> {
        private final int from;
        private final int to;
        private final List<Node> nodes;
        private final Set<Node> emptySet;
        private final int graphIndex;

        public NodeTaskEmptyGraph(int from, int to, List<Node> nodes, Set<Node> emptySet, int graphIndex) {
            this.from = from;
            this.to = to;
            this.nodes = nodes;
            this.emptySet = emptySet;
            this.graphIndex = graphIndex;
        }

        @Override
        protected Boolean compute() {
            for (int i = from; i < to; i++) {
                if ((i + 1) % 1000 == 0) {
                    count[0] += 1000;
                    out.println("Initializing effect edges: " + (count[0]));
                }

                Node y = nodes.get(i);
                neighborsList.get(graphIndex).put(y, emptySet);

                for (int j = i + 1; j < nodes.size(); j++) {
                    Node x = nodes.get(j);

                    if (existsKnowledge()) {
                        if (getKnowledge().isForbidden(x.getName(), y.getName()) &&
                                getKnowledge().isForbidden(y.getName(), x.getName())) {
                            continue;
                        }

                        if (!validSetByKnowledge(y, emptySet)) {
                            continue;
                        }
                    }

                    if (adjacencies != null && !adjacencies.isAdjacentTo(x, y)) {
                        continue;
                    }

                    int child = hashIndices.get(y);
                    int parent = hashIndices.get(x);
                    double bump = scoreList.get(graphIndex).localScoreDiff(parent, child);

                    if (symmetricFirstStep) {
                        double bump2 = scoreList.get(graphIndex).localScoreDiff(child, parent);
                        bump = bump > bump2 ? bump : bump2;
                    }

                    if (boundGraph != null && !boundGraph.isAdjacentTo(x, y)) continue;

                    if (bump > bumpMin) {
                        final Edge edge = Edges.undirectedEdge(x, y);
                        effectEdgesGraphList.get(graphIndex).addEdge(edge);
                    }

                    if (bump > bumpMin) {
                        addArrow(x, y, emptySet, emptySet, bump, graphIndex);
                        addArrow(y, x, emptySet, emptySet, bump, graphIndex);
                    }
                }
            }

            return true;
        }
    }

    private void initializeForwardEdgesFromEmptyGraph(final List<Node> nodes) {
//        if (verbose) {
//            System.out.println("heuristicSpeedup = true");
//        }

        lookupArrowsList = new ArrayList<>(); //ConcurrentHashMap
        neighborsList = new ArrayList<>(); //ConcurrentHashMap
        final Set<Node> emptySet = new HashSet<>();

        long start = System.currentTimeMillis();
        this.effectEdgesGraphList = new ArrayList<>(); // EdgeListGraphSingleConnections(nodes);
        for (int graphIndex = 0; graphIndex < numGraphs; graphIndex++) {
            lookupArrowsList.add(new ConcurrentHashMap<OrderedPair<Node>, Set<Arrow>>());
            neighborsList.add(new ConcurrentHashMap<Node, Set<Node>>());
            effectEdgesGraphList.add(new EdgeListGraph(nodes));
        }

        class InitializeFromEmptyGraphTask extends RecursiveTask<Boolean> {

            public InitializeFromEmptyGraphTask() {
            }

            @Override
            protected Boolean compute() {
                Queue<NodeTaskEmptyGraph> tasks = new ArrayDeque<>();

                int numNodesPerTask = Math.max(100, nodes.size() / maxThreads);

                for (int graphIndex = 0; graphIndex < numGraphs; graphIndex++) {
                    for (int i = 0; i < nodes.size(); i += numNodesPerTask) {
                        NodeTaskEmptyGraph task = new NodeTaskEmptyGraph(i, Math.min(nodes.size(), i + numNodesPerTask),
                                nodes, emptySet, graphIndex);
                        tasks.add(task);
                        task.fork();

                        for (NodeTaskEmptyGraph _task : new ArrayList<>(tasks)) {
                            if (_task.isDone()) {
                                _task.join();
                                tasks.remove(_task);
                            }
                        }

                        while (tasks.size() > maxThreads) {
                            NodeTaskEmptyGraph _task = tasks.poll();
                            _task.join();
                        }
                    }

                    for (NodeTaskEmptyGraph task : tasks) {
                        task.join();
                    }
                }

                return true;
            }
        }

        pool.invoke(new InitializeFromEmptyGraphTask());

        long stop = System.currentTimeMillis();

        if (verbose) {
            out.println("Elapsed initializeForwardEdgesFromEmptyGraph = " + (stop - start) + " ms");
            out.println(effectEdgesGraphList);
        }
    }

    private void initializeTwoStepEdges(final List<Node> nodes) {
//        if (verbose) {
//            System.out.println("heuristicSpeedup = false");
//        }

        count[0] = 0;

        lookupArrowsList = new ArrayList<>(); //ConcurrentHashMap
        neighborsList = new ArrayList<>(); //ConcurrentHashMap
        for (int graphIndex = 0; graphIndex < numGraphs; graphIndex++) {
            lookupArrowsList.add(new ConcurrentHashMap<OrderedPair<Node>, Set<Arrow>>());
            neighborsList.add(new ConcurrentHashMap<Node, Set<Node>>());
        }

        if (this.effectEdgesGraphList == null) {
            this.effectEdgesGraphList = new ArrayList<>(); //EdgeListGraph(nodes);
            for (int graphIndex = 0; graphIndex < numGraphs; graphIndex++) {
                effectEdgesGraphList.add(new EdgeListGraph(nodes));
            }
        }

        /*if (initialGraph != null) {
            for (Edge edge : initialGraph.getEdges()) {
                if (!effectEdgesGraph.isAdjacentTo(edge.getNode1(), edge.getNode2())) {
                    effectEdgesGraph.addUndirectedEdge(edge.getNode1(), edge.getNode2());
                }
            }
        }*/

        final Set<Node> emptySet = new HashSet<>(0);

        class InitializeFromExistingGraphTask extends RecursiveTask<Boolean> {
            private int chunk;
            private int from;
            private int to;

            public InitializeFromExistingGraphTask(int chunk, int from, int to) {
                this.chunk = chunk;
                this.from = from;
                this.to = to;
            }

            @Override
            protected Boolean compute() {
                if (TaskManager.getInstance().isCanceled()) return false;


                if (to - from <= chunk) {
                    for (int graphIndex = 0; graphIndex < numGraphs; graphIndex++) {

                        for (int i = from; i < to; i++) {
                            if ((i + 1) % 1000 == 0) {
                                count[0] += 1000;
                                out.println("Initializing effect edges: " + (count[0]));
                            }

                            Node y = nodes.get(i);

                            Set<Node> g = new HashSet<>();

                            for (Node n : graphConfiguration.getGraph(graphIndex).getAdjacentNodes(y)) {
                                for (Node m : graphConfiguration.getGraph(graphIndex).getAdjacentNodes(n)) {
                                    if (m == y) continue;

                                    if (graphConfiguration.getGraph(graphIndex).isAdjacentTo(y, m)) {
                                        continue;
                                    }

                                    if (graphConfiguration.getGraph(graphIndex).isDefCollider(m, n, y)) {
                                        continue;
                                    }

                                    g.add(m);
                                }
                            }

                            for (Node x : g) {
                                if (x == y) throw new IllegalArgumentException();

                                if (existsKnowledge()) {
                                    if (getKnowledge().isForbidden(x.getName(), y.getName()) &&
                                            getKnowledge().isForbidden(y.getName(), x.getName())) {
                                        continue;
                                    }

                                    if (!validSetByKnowledge(y, emptySet)) {
                                        continue;
                                    }
                                }

                                if (adjacencies != null && !adjacencies.isAdjacentTo(x, y)) {
                                    continue;
                                }

                                if (removedEdges.contains(Edges.undirectedEdge(x, y))) {
                                    continue;
                                }

                                calculateArrowsForward(x, y, graphIndex);
                            }
                        }
                    }

                    return true;
                } else {
                    int mid = (to + from) / 2;

                    InitializeFromExistingGraphTask left = new InitializeFromExistingGraphTask(chunk, from, mid);
                    InitializeFromExistingGraphTask right = new InitializeFromExistingGraphTask(chunk, mid, to);

                    left.fork();
                    right.compute();
                    left.join();

                    return true;
                }

            }
        }

        pool.invoke(new InitializeFromExistingGraphTask(getMinChunk(nodes.size()), 0, nodes.size()));
    }

    private void initializeForwardEdgesFromExistingGraph(final List<Node> nodes) {
//        if (verbose) {
//            System.out.println("heuristicSpeedup = false");
//        }

        count[0] = 0;

        lookupArrowsList = new ArrayList<>(); //ConcurrentHashMap
        neighborsList = new ArrayList<>(); //ConcurrentHashMap
        for (int graphIndex = 0; graphIndex < numGraphs; graphIndex++) {
            lookupArrowsList.add(new ConcurrentHashMap<OrderedPair<Node>, Set<Arrow>>());
            neighborsList.add(new ConcurrentHashMap<Node, Set<Node>>());
        }

        if (this.effectEdgesGraphList == null) {
            this.effectEdgesGraphList = new ArrayList<>(); //EdgeListGraph(nodes);
            for (int graphIndex = 0; graphIndex < numGraphs; graphIndex++) {
                effectEdgesGraphList.add(new EdgeListGraph(nodes));
            }
        }


        /*if (initialGraph != null) {
            for (Edge edge : initialGraph.getEdges()) {
                if (!effectEdgesGraph.isAdjacentTo(edge.getNode1(), edge.getNode2())) {
                    effectEdgesGraph.addUndirectedEdge(edge.getNode1(), edge.getNode2());
                }
            }
        }*/

        final Set<Node> emptySet = new HashSet<>(0);

        class InitializeFromExistingGraphTask extends RecursiveTask<Boolean> {
            private int chunk;
            private int from;
            private int to;

            public InitializeFromExistingGraphTask(int chunk, int from, int to) {
                this.chunk = chunk;
                this.from = from;
                this.to = to;
            }

            @Override
            protected Boolean compute() {
                if (TaskManager.getInstance().isCanceled()) return false;

                if (to - from <= chunk) {
                    for (int graphIndex = 0; graphIndex < numGraphs; graphIndex++) {
                        for (int i = from; i < to; i++) {
                            if ((i + 1) % 1000 == 0) {
                                count[0] += 1000;
                                out.println("Initializing effect edges: " + (count[0]));
                            }

                            Node y = nodes.get(i);
                            Set<Node> D = new HashSet<>();
                            List<Node> cond = new ArrayList<>();
                            D.addAll(GraphUtils.getDconnectedVars(y, cond, graphConfiguration.getGraph(graphIndex)));
                            D.remove(y);
                            D.removeAll(effectEdgesGraphList.get(graphIndex).getAdjacentNodes(y));

                            for (Node x : D) {
                                if (existsKnowledge()) {
                                    if (getKnowledge().isForbidden(x.getName(), y.getName()) && getKnowledge().isForbidden(y.getName(), x.getName())) {
                                        continue;
                                    }

                                    if (!validSetByKnowledge(y, emptySet)) {
                                        continue;
                                    }
                                }

                                if (adjacencies != null && !adjacencies.isAdjacentTo(x, y)) {
                                    continue;
                                }

                                calculateArrowsForward(x, y, graphIndex);
                            }
                        }
                    }

                    return true;
                } else {
                    int mid = (to + from) / 2;

                    InitializeFromExistingGraphTask left = new InitializeFromExistingGraphTask(chunk, from, mid);
                    InitializeFromExistingGraphTask right = new InitializeFromExistingGraphTask(chunk, mid, to);

                    left.fork();
                    right.compute();
                    left.join();

                    return true;
                }
            }
        }

        pool.invoke(new InitializeFromExistingGraphTask(getMinChunk(nodes.size()), 0, nodes.size()));
    }

    private void fest() {
        TetradLogger.getInstance().log("info", "** FORWARD EQUIVALENCE SEARCH");

        // TODO use maxdegree
        int maxDegree = this.maxDegree == -1 ? 1000 : this.maxDegree;

        double scoreImprovement = 1;

        int numRounds = 0;

        while (scoreImprovement > 0) {
            // put BranchBounds into a sorted set, to facilitate picking the max-scoring one.
            SortedSet<BranchBound> nodePairConfigs = new ConcurrentSkipListSet<>();
            System.out.println("nodePairConfigs size: " + nodePairConfigs.size());

            // loop over pairs of nodes; calculate all the new graphconfigs and associated scores
            for (int i = 0; i < variables.size(); i++) {
                for (int j = i + 1; j < variables.size(); j++) {
                    Node x = variables.get(i);
                    Node y = variables.get(j);
                    BranchBound bb = new BranchBound(graphConfiguration, x, y, true);

                    // use bound from best nodePairConfig in set so far. This will truncate BranchBound.search() for
                    // less promising cases
                    //if (nodePairConfigs.size() > 0) {
                     //   bb.setOuterBound(nodePairConfigs.first().getBestScoreSoFar());
                    //}

                    bb.search();
                    nodePairConfigs.add(bb);
                }
            }

            System.out.println("nodePairConfigs size: " + nodePairConfigs.size());

            BranchBound bestBb = nodePairConfigs.first();
            System.out.println("arrows: " + bestBb.getBestArrowsSoFar());


            // check that the best config actually improves the score; if it doesn't, we're done with FEST.
            scoreImprovement = bestBb.getBestScoreSoFar();
            System.out.println("scoreImprovement: " + scoreImprovement);

            if (scoreImprovement <= 0) {return;}
            totalScore += scoreImprovement;

            List<Arrow> arrowList = bestBb.getBestArrowsSoFar();
            numRounds ++;
            System.out.println("Number of arrows: " + arrowList.size());
            System.out.println("Round number: " + numRounds);
            if (arrowList.size() != numGraphs) {
                throw new IllegalStateException("list of arrows from branchbound not same length as graph list");
            }
            // add all the edges in the config
            for (int graphIndex = 0; graphIndex < numGraphs; graphIndex++) {

                Arrow arrow = arrowList.get(graphIndex);

                // remember, "do nothing" is always an option, so we check that the arrow isn't null.
                // If it's not null, we insert it and then rebuild the pattern.
                if (arrow != null) {

                    Node x = arrow.getA();
                    Node y = arrow.getB();

                    Set<Node> T = arrow.getHOrT();
                    double bump = arrow.getBump();

                    boolean inserted = insert(x, y, T, bump, graphIndex);
                    if (!inserted) continue;

                    Set<Node> visited = reapplyOrientation(x, y, null, graphIndex);
                    Set<Node> toProcess = new HashSet<>();

                    for (Node node : visited) {
                        final Set<Node> neighbors1 = getNeighbors(node, graphIndex);
                        final Set<Node> storedNeighbors = this.neighborsList.get(graphIndex).get(node);

                        if (!(neighbors1.equals(storedNeighbors))) {
                            toProcess.add(node);
                        }
                    }

                    toProcess.add(x);
                    toProcess.add(y);

                    reevaluateForward(toProcess, arrow, graphIndex);
                }
            }
        }
    }

    private void best() {
        TetradLogger.getInstance().log("info", "** BACKWARD EQUIVALENCE SEARCH");

        lookupArrowsList = new ArrayList<>(); // ConcurrentHashMap
        neighborsList = new ArrayList<>(); //ConcurrentHashMap

        for (int graphIndex = 0; graphIndex < numGraphs; graphIndex++) {
            lookupArrowsList.add(new ConcurrentHashMap<OrderedPair<Node>, Set<Arrow>>());
            neighborsList.add(new ConcurrentHashMap<Node, Set<Node>>());
        }

        initializeArrowsBackward();

        double scoreImprovement = 1;

        while (scoreImprovement > 0) {
            // put branchbounds into a sorted set, to facilitate picking the max-scoring one.
            SortedSet<BranchBound> nodePairConfigs = new ConcurrentSkipListSet<>();

            // loop over pairs of nodes; calculate all the new graphconfigs and associated scores
            for (int i = 0; i < variables.size(); i++) {
                for (int j = i + 1; j < variables.size(); j++) {
                    Node x = variables.get(i);
                    Node y = variables.get(j);
                    BranchBound bb = new BranchBound(graphConfiguration, x, y, false);

                    // use bound from best nodePairConfig in set so far. This will truncate BranchBound.search() for
                    // less promising cases
                    if (nodePairConfigs.size() > 0) {
                        bb.setOuterBound(nodePairConfigs.first().getBestScoreSoFar());
                    }

                    bb.search();
                    nodePairConfigs.add(bb);
                }
            }

            BranchBound bestBb = nodePairConfigs.first();

            //System.out.println(bestBb.getBestArrowsSoFar());
            //System.out.println(bestBb.getBestScoreSoFar());

            // check that the best config actually improves the score; if it doesn't, we're done with BEST.
            scoreImprovement = bestBb.getBestScoreSoFar();
            if (scoreImprovement <= 0) {return;}
            totalScore += scoreImprovement;

            List<Arrow> arrowList = bestBb.getBestArrowsSoFar();
            if (arrowList.size() != numGraphs) {
                throw new IllegalStateException("list of arrows from branchbound not same length as graph list");
            }

            // remove all the edges in the config
            for (int graphIndex = 0; graphIndex < numGraphs; graphIndex++) {

                Arrow arrow = arrowList.get(graphIndex);

                // remember, "do nothing" is always an option, so we check that the arrow isn't null.
                // If it's not null, we remove it and then rebuild the pattern.
                if (arrow != null) {

                    Node x = arrow.getA();
                    Node y = arrow.getB();

                    Set<Node> H = arrow.getHOrT();
                    double bump = arrow.getBump();

                    boolean deleted = delete(x, y, H, bump, arrow.getNaYX(), graphIndex);
                    if (!deleted) continue;
                    clearArrow(x, y, graphIndex);

                    Set<Node> visited = reapplyOrientation(x, y, H, graphIndex);

                    Set<Node> toProcess = new HashSet<>();

                    for (Node node : visited) {
                        final Set<Node> neighbors1 = getNeighbors(node, graphIndex);
                        final Set<Node> storedNeighbors = this.neighborsList.get(graphIndex).get(node);

                        if (!(neighbors1.equals(storedNeighbors))) {
                            toProcess.add(node);
                        }
                    }

                    toProcess.add(x);
                    toProcess.add(y);
                    toProcess.addAll(getCommonAdjacents(x, y, graphIndex));

                    reevaluateBackward(toProcess, graphIndex);
                }

                meekOrientRestricted(getVariables(), getKnowledge(), graphIndex);
            }
        }
    }

    private Set<Node> getCommonAdjacents(Node x, Node y, int graphIndex) {
        Set<Node> commonChildren = new HashSet<>(graphConfiguration.getGraph(graphIndex).getAdjacentNodes(x));
        commonChildren.retainAll(graphConfiguration.getGraph(graphIndex).getAdjacentNodes(y));
        return commonChildren;
    }

    private Set<Node> reapplyOrientation(Node x, Node y, Set<Node> newArrows, int graphIndex) {
        Set<Node> toProcess = new HashSet<>();
        toProcess.add(x);
        toProcess.add(y);

        if (newArrows != null) {
            toProcess.addAll(newArrows);
        }

        return meekOrientRestricted(new ArrayList<>(toProcess), getKnowledge(), graphIndex);
    }

    // Returns true if knowledge is not empty.
    private boolean existsKnowledge() {
        return !knowledge.isEmpty();
    }


    // Initiaizes the sorted arrows lists for the backward search.
    private void initializeArrowsBackward() {
        for (int graphIndex = 0; graphIndex < numGraphs; graphIndex++) {
            for (Edge edge : graphConfiguration.getGraph(graphIndex).getEdges()) {
                Node x = edge.getNode1();
                Node y = edge.getNode2();

                if (existsKnowledge()) {
                    if (!getKnowledge().noEdgeRequired(x.getName(), y.getName())) {
                        continue;
                    }
                }

                clearArrow(x, y, graphIndex);
                clearArrow(y, x, graphIndex);

                if (edge.pointsTowards(y)) {
                    calculateArrowsBackward(x, y, graphIndex);
                } else if (edge.pointsTowards(x)) {
                    calculateArrowsBackward(y, x, graphIndex);
                } else {
                    calculateArrowsBackward(x, y, graphIndex);
                    calculateArrowsBackward(y, x, graphIndex);
                }

                this.neighborsList.get(graphIndex).put(x, getNeighbors(x, graphIndex));
                this.neighborsList.get(graphIndex).put(y, getNeighbors(y, graphIndex));
            }
        }
    }

    // Calcuates new arrows based on changes in the graph for the forward search.
    private void reevaluateForward(final Set<Node> nodes, final Arrow arrow, int graphIndex) {
        class AdjTask extends RecursiveTask<Boolean> {
            private final List<Node> nodes;
            private int from;
            private int to;
            private int chunk;
            private int graphIndex;

            public AdjTask(int chunk, List<Node> nodes, int from, int to, int graphIndex) {
                this.nodes = nodes;
                this.from = from;
                this.to = to;
                this.chunk = chunk;
                this.graphIndex = graphIndex;
            }

            @Override
            protected Boolean compute() {
                if (to - from <= chunk) {

                    for (int _w = from; _w < to; _w++) {
                        Node x = nodes.get(_w);

                        List<Node> adj;

                        if (mode == Mode.heuristicSpeedup) {
                            adj = effectEdgesGraphList.get(graphIndex).getAdjacentNodes(x);
                        } else if (mode == Mode.coverNoncolliders) {
                            Set<Node> g = new HashSet<>();

                            for (Node n : graphConfiguration.getGraph(graphIndex).getAdjacentNodes(x)) {
                                for (Node m : graphConfiguration.getGraph(graphIndex).getAdjacentNodes(n)) {
                                    if (graphConfiguration.getGraph(graphIndex).isAdjacentTo(x, m)) {
                                        continue;
                                    }

                                    if (graphConfiguration.getGraph(graphIndex).isDefCollider(m, n, x)) {
                                        continue;
                                    }

                                    g.add(m);
                                }
                            }

                            adj = new ArrayList<>(g);
                        } else if (mode == Mode.allowUnfaithfulness) {
                            HashSet<Node> D = new HashSet<>();
                            D.addAll(GraphUtils.getDconnectedVars(x, new ArrayList<Node>(), graphConfiguration.getGraph(graphIndex)));
                            D.remove(x);
                            adj = new ArrayList<>(D);
                        } else {
                            throw new IllegalStateException();
                        }

                        for (Node w : adj) {
                            if (adjacencies != null && !(adjacencies.isAdjacentTo(w, x))) {
                                continue;
                            }

                            if (w == x) continue;

                            if (!graphConfiguration.getGraph(graphIndex).isAdjacentTo(w, x)) {
                                clearArrow(w, x, graphIndex);
                                calculateArrowsForward(w, x, graphIndex);
                            }
                        }
                    }

                    return true;
                } else {
                    int mid = (to - from) / 2;

                    List<AdjTask> tasks = new ArrayList<>();

                    tasks.add(new AdjTask(chunk, nodes, from, from + mid, graphIndex));
                    tasks.add(new AdjTask(chunk, nodes, from + mid, to, graphIndex));

                    invokeAll(tasks);

                    return true;
                }
            }
        }

        final AdjTask task = new AdjTask(getMinChunk(nodes.size()), new ArrayList<>(nodes), 0, nodes.size(), graphIndex);
        pool.invoke(task);
    }

    // Calculates the new arrows for an a->b edge.
    private void calculateArrowsForward(Node a, Node b, int graphIndex) {
        if (mode == Mode.heuristicSpeedup && !effectEdgesGraphList.get(graphIndex).isAdjacentTo(a, b)) return;
        if (adjacencies != null && !adjacencies.isAdjacentTo(a, b)) return;
        this.neighborsList.get(graphIndex).put(b, getNeighbors(b, graphIndex));

        if (a == b) throw new IllegalArgumentException();

        if (existsKnowledge()) {
            if (getKnowledge().isForbidden(a.getName(), b.getName())) {
                return;
            }
        }

        Set<Node> naYX = getNaYX(a, b, graphIndex);
        if (!isClique(naYX, graphIndex)) return;

        List<Node> TNeighbors = getTNeighbors(a, b, graphIndex);

        Set<Set<Node>> previousCliques = new HashSet<>();
        previousCliques.add(new HashSet<Node>());
        Set<Set<Node>> newCliques = new HashSet<>();

        FOR:
        for (int i = 0; i <= TNeighbors.size(); i++) {
            final ChoiceGenerator gen = new ChoiceGenerator(TNeighbors.size(), i);
            int[] choice;

            while ((choice = gen.next()) != null) {
                Set<Node> T = GraphUtils.asSet(choice, TNeighbors);

                Set<Node> union = new HashSet<>(naYX);
                union.addAll(T);

                boolean foundAPreviousClique = false;

                for (Set<Node> clique : previousCliques) {
                    if (union.containsAll(clique)) {
                        foundAPreviousClique = true;
                        break;
                    }
                }

                if (!foundAPreviousClique) {
                    break FOR;
                }

                if (!isClique(union, graphIndex)) continue;
                newCliques.add(union);

                double bump = insertEval(a, b, T, naYX, hashIndices, graphIndex);

                // the transfer penalty could be as large as this.
                if (bump > bumpMin) {
                    addArrow(a, b, naYX, T, bump, graphIndex);
                }

//                if (mode == Mode.heuristicSpeedup && union.isEmpty() && score.isEffectEdge(bump) &&
//                        !effectEdgesGraph.isAdjacentTo(a, b) && graph.getParents(b).isEmpty()) {
//                    effectEdgesGraph.addUndirectedEdge(a, b);
//                }
            }

            previousCliques = newCliques;
            newCliques = new HashSet<>();
        }
    }

    private void addArrow(Node a, Node b, Set<Node> naYX, Set<Node> hOrT, double bump, int graphIndex) {
        Arrow arrow = new Arrow(bump, a, b, hOrT, naYX, arrowIndex++);
        addLookupArrow(a, b, arrow, graphIndex);
    }

    // Reevaluates arrows after removing an edge from the graph.
    private void reevaluateBackward(Set<Node> toProcess, final int graphIndex) {
        class BackwardTask extends RecursiveTask<Boolean> {
            private final Node r;
            private List<Node> adj;
            private Map<Node, Integer> hashIndices;
            private int chunk;
            private int from;
            private int to;
            private int graphIndex;

            public BackwardTask(Node r, List<Node> adj, int chunk, int from, int to,
                                Map<Node, Integer> hashIndices, int graphIndex) {
                this.adj = adj;
                this.hashIndices = hashIndices;
                this.chunk = chunk;
                this.from = from;
                this.to = to;
                this.r = r;
                this.graphIndex = graphIndex;
            }

            @Override
            protected Boolean compute() {
                if (to - from <= chunk) {
                        for (int _w = from; _w < to; _w++) {
                            final Node w = adj.get(_w);
                            Edge e = graphConfiguration.getGraph(graphIndex).getEdge(w, r);

                            if (e != null) {
                                if (e.pointsTowards(r)) {
                                    clearArrow(w, r, graphIndex);
                                    clearArrow(r, w, graphIndex);

                                    calculateArrowsBackward(w, r, graphIndex);
                                } else if (Edges.isUndirectedEdge(graphConfiguration.getGraph(graphIndex).getEdge(w, r))) {
                                    clearArrow(w, r, graphIndex);
                                    clearArrow(r, w, graphIndex);

                                    calculateArrowsBackward(w, r, graphIndex);
                                    calculateArrowsBackward(r, w, graphIndex);
                                }
                            }
                        }


                    return true;
                } else {
                    int mid = (to - from) / 2;

                    List<BackwardTask> tasks = new ArrayList<>();

                    tasks.add(new BackwardTask(r, adj, chunk, from, from + mid, hashIndices, graphIndex));
                    tasks.add(new BackwardTask(r, adj, chunk, from + mid, to, hashIndices, graphIndex));

                    invokeAll(tasks);

                    return true;
                }
            }
        }

        for (Node r : toProcess) {
            this.neighborsList.get(graphIndex).put(r, getNeighbors(r, graphIndex));
            List<Node> adjacentNodes = graphConfiguration.getGraph(graphIndex).getAdjacentNodes(r);
            pool.invoke(new BackwardTask(r, adjacentNodes, getMinChunk(adjacentNodes.size()), 0,
                    adjacentNodes.size(), hashIndices, graphIndex));
        }
    }

    // Calculates the arrows for the removal in the backward direction.
    private void calculateArrowsBackward(Node a, Node b, int graphIndex) {
        if (existsKnowledge()) {
            if (!getKnowledge().noEdgeRequired(a.getName(), b.getName())) {
                return;
            }
        }

        Set<Node> naYX = getNaYX(a, b, graphIndex);

        List<Node> _naYX = new ArrayList<>(naYX);

        final int _depth = _naYX.size();

        for (int i = 0; i <= _depth; i++) {
            final ChoiceGenerator gen = new ChoiceGenerator(_naYX.size(), i);
            int[] choice;

            while ((choice = gen.next()) != null) {
                Set<Node> diff = GraphUtils.asSet(choice, _naYX);

                Set<Node> h = new HashSet<>(_naYX);
                h.removeAll(diff);

                if (existsKnowledge()) {
                    if (!validSetByKnowledge(b, h)) {
                        continue;
                    }
                }

                double bump = deleteEval(a, b, diff, naYX, hashIndices, graphIndex);

                if (bump > 0.0) {
                    addArrow(a, b, naYX, h, bump, graphIndex);
                }
            }
        }
    }

    public double getModelScore() {
        return modelScore;
    }

    // Basic data structure for an arrow a->b considered for additiom or removal from the graph, together with
    // associated sets needed to make this determination. For both forward and backward direction, NaYX is needed.
    // For the forward direction, T neighbors are needed; for the backward direction, H neighbors are needed.
    // See Chickering (2002). The totalScore difference resulting from added in the edge (hypothetically) is recorded
    // as the "bump".
    private static class Arrow implements Comparable<Arrow> {
        private double bump;
        private Node a;
        private Node b;
        private Set<Node> hOrT;
        private Set<Node> naYX;
        private int index = 0;

        public Arrow(double bump, Node a, Node b, Set<Node> hOrT, Set<Node> naYX, int index) {
            this.bump = bump;
            this.a = a;
            this.b = b;
            this.hOrT = hOrT;
            this.naYX = naYX;
            this.index = index;
        }

        public double getBump() {
            return bump;
        }

        public Node getA() {
            return a;
        }

        public Node getB() {
            return b;
        }

        public Set<Node> getHOrT() {
            return hOrT;
        }

        public Set<Node> getNaYX() {
            return naYX;
        }

        // Sorting by bump, high to low. The problem is the SortedSet contains won't add a new element if it compares
        // to zero with an existing element, so for the cases where the comparison is to zero (i.e. have the same
        // bump, we need to determine as quickly as possible a determinate ordering (fixed) ordering for two variables.
        // The fastest way to do this is using a hash code, though it's still possible for two Arrows to have the
        // same hash code but not be equal. If we're paranoid, in this case we calculate a determinate comparison
        // not equal to zero by keeping a list. This last part is commened out by default.
        public int compareTo(Arrow arrow) {
            if (arrow == null) throw new NullPointerException();

            final int compare = Double.compare(arrow.getBump(), getBump());

            if (compare == 0) {
                return Integer.compare(getIndex(), arrow.getIndex());
            }

            return compare;
        }

        public String toString() {
            return "Arrow<" + a + "->" + b + " bump = " + bump + " t/h = " + hOrT + " naYX = " + naYX + ">";
        }

        public int getIndex() {
            return index;
        }
    }

    // for a given pair of nodes, choose the next best configuration using a branch and bound algorithm
    private class BranchBound implements Comparable<BranchBound> {
        private GraphConfiguration graphConfiguration;
        private double bestScoreSoFar;
        private List<Arrow> bestArrowsSoFar;
        private double[] bestBumps;
        private List<List<Arrow>> childArrows;
        private Node x;
        private Node y;
        private boolean fest;
        // give it a field titled "outerBound" so it can compare bestScoreSoFar with other BranchBound objects
        private double outerBound;
        private boolean[] alreadyAdjacent;
        private double existingTransferPenalty;

        BranchBound(GraphConfiguration graphConfiguration, Node x, Node y, boolean fest) {
            this.graphConfiguration = graphConfiguration;
            this.x = x;
            this.y = y;
            this.fest = fest;

            this.bestScoreSoFar = 0;
            this.bestArrowsSoFar = new ArrayList<>();
            this.bestBumps = new double[numGraphs];
            this.outerBound = 0;

            this.alreadyAdjacent = new boolean[numGraphs];
            double existingAdjacencies = 0;
            double existingNonAdjacencies = 0;
            for (int i = 0; i < numGraphs; i++) {
                alreadyAdjacent[i] = graphConfiguration.getGraph(i).isAdjacentTo(x, y);
                if (alreadyAdjacent[i]) {existingAdjacencies ++;} else {existingNonAdjacencies ++;}
            }
            if (existingAdjacencies + existingNonAdjacencies != numGraphs) {
                throw new IllegalStateException("number of existing adjacencies + nonadjacencies != numgraphs");
            }
            this.existingTransferPenalty = - transferPenalty * existingAdjacencies * existingNonAdjacencies;

            int maxDegreeBB = maxDegree == -1 ? 1000 : maxDegree;

            // Here we create the childArrows list, which gives the arrow options at each GNode.
            // In FEST, the options are to add the arrow x --> y ("to"), or x <-- y ("from"), or add no arrow.
            // If adding an arrow in a given direction, we choose the arrow with the HorT set that gives the best "bump"
            // because the transfer penalty is unaffected by HorT.
            // One direction may be unavailable because there are no arrows in that direction that satisfy the validity
            // conditions.
            // In BEST, there is only the option to remove the existing arrow, or keep it.
            this.childArrows = new ArrayList<>();
            for (int graphIndex = 0; graphIndex < numGraphs; graphIndex ++) {
                bestBumps[graphIndex] = 0;
                childArrows.add(new ArrayList<Arrow>());

                // first, check that we can actually add an edge between these nodes (in FEST) or remove one (in BEST)
                boolean conditions = false;
                if (fest) {
                    // check x and y are not adjacent and we haven't reached the max degree yet
                    conditions = !graphConfiguration.getGraph(graphIndex).isAdjacentTo(x, y) &&
                            graphConfiguration.getGraph(graphIndex).getDegree(x) < maxDegreeBB &&
                            graphConfiguration.getGraph(graphIndex).getDegree(y) < maxDegreeBB;
                } else {
                    // for backward equivalence search, check x and y ARE adjacent
                    conditions = graphConfiguration.getGraph(graphIndex).isAdjacentTo(x, y);
                }

                if (conditions) {

                    Set<Arrow> toArrows = lookupArrowsList.get(graphIndex).get(new OrderedPair<>(x, y));
                    Set<Arrow> fromArrows = lookupArrowsList.get(graphIndex).get(new OrderedPair<>(y, x));

                    // Actually this is incorrect if the existing edge is undirected.
                    /*// if we are in BEST, the lookup ArrowsList should only have entries for "to" or "from", not both
                    if (!fest & (toArrows != null & fromArrows != null)) {
                        System.out.println("Actual edge between X and Y: " + graphConfiguration.getGraph(graphIndex).getEdge(x, y));
                        System.out.println("To arrows: " + toArrows);
                        System.out.println("From arrows: " + fromArrows);
                        throw new IllegalStateException("in BEST, and both to and from arrows exist!");
                    }*/

                    // remove from consideration all arrows that violate the validity conditions
                    if (toArrows != null) {
                        Set<Arrow> toRemove = new HashSet<>();

                        boolean arrowInvalidityConditions;
                        for (Arrow arrow : toArrows) {

                            Edge edge = graphConfiguration.getGraph(graphIndex).getEdge(x, y);

                            arrowInvalidityConditions = !arrow.getNaYX().equals(getNaYX(x, y, graphIndex)) ||
                                    (fest && (!validInsert(x, y, arrow.getHOrT(), getNaYX(x, y, graphIndex), graphIndex) ||
                                            !getTNeighbors(x, y, graphIndex).containsAll(arrow.getHOrT()))) ||
                                    (!fest && (!validDelete(x, y, arrow.getHOrT(), arrow.getNaYX(), graphIndex) ||
                                            edge.pointsTowards(x)));

                            if (arrowInvalidityConditions) {toRemove.add(arrow);}
                        }
                        for (Arrow arrow : toRemove) {toArrows.remove(arrow);}

                        // make sure there's at least one arrow in the set before choosing the arrow with the max bump
                        if (toArrows.size() > 0) {
                            Arrow forwardArrow = Collections.max(toArrows);
                            childArrows.get(graphIndex).add(forwardArrow);
                            if (forwardArrow.getBump() > bestBumps[graphIndex]) {
                                bestBumps[graphIndex] = forwardArrow.getBump();
                            }
                        }
                    }

                    // repeat process for arrows in the other direction
                    if (fromArrows != null) {
                        Set<Arrow> fromRemove = new HashSet<>();

                        boolean arrowInvalidityConditions;
                        for (Arrow arrow : fromArrows) {

                            Edge edge = graphConfiguration.getGraph(graphIndex).getEdge(y, x);

                            arrowInvalidityConditions = !arrow.getNaYX().equals(getNaYX(y, x, graphIndex)) ||
                                    (fest && (!validInsert(y, x, arrow.getHOrT(), getNaYX(y, x, graphIndex), graphIndex) ||
                                            !getTNeighbors(y, x, graphIndex).containsAll(arrow.getHOrT()))) ||
                                    (!fest && (!validDelete(y, x, arrow.getHOrT(), arrow.getNaYX(), graphIndex) ||
                                            edge.pointsTowards(y)));

                            if (arrowInvalidityConditions) {fromRemove.add(arrow);}
                        }
                        for (Arrow arrow : fromRemove) {fromArrows.remove(arrow);}

                        if (fromArrows.size() > 0) {
                            Arrow backwardArrow = Collections.max(fromArrows);
                            childArrows.get(graphIndex).add(backwardArrow);
                            if (backwardArrow.getBump() > bestBumps[graphIndex]) {
                                bestBumps[graphIndex] = backwardArrow.getBump();
                            }
                        }
                    }

                }
                // always add the "do nothing" option
                Arrow none = null;
                childArrows.get(graphIndex).add(none);
            }

        }

        public double getBestScoreSoFar() {return bestScoreSoFar;}
        public List<Arrow> getBestArrowsSoFar() {return bestArrowsSoFar;}
        public void setOuterBound(double bound) {this.outerBound = bound;}
        public Node getX() {return x;}
        public Node getY() {return y;}

        public int compareTo(BranchBound branchBound) {
            if (branchBound == null) throw new NullPointerException();

            int compare = Double.compare(branchBound.getBestScoreSoFar(), getBestScoreSoFar());

            if (compare == 0) {
                compare = Integer.compare(hashIndices.get(getX()), hashIndices.get(branchBound.getX()));
            }
            if (compare == 0) {
                return Integer.compare(hashIndices.get(getY()), hashIndices.get(branchBound.getY()));
            }

            return compare;
        }

        private class GNode {
            private GNode parentNode;
            private Arrow arrow;
            private int graphIndex;
            private List<GNode> childNodes;
            private boolean explored;

            // adjacent and nonAdjacent refer to the cumulative number of GRAPHS (up to this node in the BranchBound)
            // where x and y are adjacent or not, respectively. They're used to calculate the transfer penalty.
            private int adjacent;
            private int nonAdjacent;
            private double cumulativeBump;
            private double scoreBound;

            GNode(GNode parentNode, Arrow arrow) {
                this.parentNode = parentNode;
                this.arrow = arrow;
                this.childNodes = new ArrayList<>();
                this.explored = false;

                // check to see if we are at the root node before calling any method on parentNode or arrow
                if (parentNode != null) {
                    this.graphIndex = parentNode.getGraphIndex() + 1;
                    this.cumulativeBump = parentNode.getCumulativeBump(); // bump from arrow added below

                    // calculate the number of adjacencies and non-adjacencies for the transfer penalty
                    boolean alreadyAdjacentHere = alreadyAdjacent[graphIndex];

                    if (fest) {
                        if (alreadyAdjacentHere) {
                            if (arrow == null) {
                                this.adjacent = parentNode.getAdjacent() + 1;
                                this.nonAdjacent = parentNode.getNonAdjacent();
                            } else {
                                throw new IllegalStateException("Nodes " + x + " and " + y +
                                        " are already adjacent in graph " + graphIndex +
                                        " but BranchBound (in FEST) has added an edge between them");
                            }
                        } else {
                            if (arrow == null) {
                                this.adjacent = parentNode.getAdjacent();
                                this.nonAdjacent = parentNode.getNonAdjacent() + 1;

                            } else {
                                this.adjacent = parentNode.getAdjacent() + 1;
                                this.nonAdjacent = parentNode.getNonAdjacent();
                                this.cumulativeBump += arrow.getBump();
                            }
                        }
                    } else {
                        // Note that in BEST, we are *removing* each arrow.
                        if (alreadyAdjacentHere) {
                            if (arrow == null) {
                                this.adjacent = parentNode.getAdjacent() + 1;
                                this.nonAdjacent = parentNode.getNonAdjacent();
                            } else {
                                this.adjacent = parentNode.getAdjacent();
                                this.nonAdjacent = parentNode.getNonAdjacent() + 1;
                                this.cumulativeBump += arrow.getBump();
                            }
                        } else {
                            if (arrow == null) {
                                this.adjacent = parentNode.getAdjacent();
                                this.nonAdjacent = parentNode.getNonAdjacent() + 1;
                            } else {
                                throw new IllegalStateException("Nodes " + x + " and " + y +
                                        " are not adjacent in graph " + graphIndex +
                                        " but BranchBound (in BEST) has removed an edge between them");
                            }
                        }
                    }

                    /*// this only works when starting from an empty graph!
                    if (arrow == null) {
                        this.adjacent = parentNode.getAdjacent();
                        this.nonAdjacent = parentNode.getNonAdjacent() + 1;
                        this.cumulativeBump = parentNode.getCumulativeBump();
                    } else {
                        this.adjacent = parentNode.getAdjacent() + 1;
                        this.nonAdjacent = parentNode.getNonAdjacent();
                        this.cumulativeBump = parentNode.getCumulativeBump() + arrow.getBump();
                    }*/

                } else {
                    this.graphIndex = -1;
                    this.adjacent = 0;
                    this.nonAdjacent = 0;
                    this.cumulativeBump = 0;
                }

                double bestQ = 0;
                for (int i = graphIndex + 1; i < numGraphs; i++) {bestQ = bestQ + bestBumps[i];}
                this.scoreBound = cumulativeBump + bestQ - (transferPenalty * adjacent * nonAdjacent - existingTransferPenalty);
            }

            public int getGraphIndex() {return graphIndex;}
            public Arrow getArrow() {return arrow;}
            public List<GNode> getChildNodes() { return childNodes; }
            public GNode getParentNode() {return parentNode;}
            public int getAdjacent() {return adjacent;}
            public int getNonAdjacent() {return nonAdjacent;}
            public boolean getExplored() {return explored;}
            public double getCumulativeBump() {return cumulativeBump;}
            public double getScoreBound() {return scoreBound;}
            public void pruneChildNode(GNode badChild) { childNodes.remove(badChild); }

            // We only have to instantiate the child nodes when we explore a node. Instead of creating them all at the
            // start of the search, we create and prune them as we go, so the search doesn't require as much memory.
            public void makeChildNodes() {
                if (explored) {throw new IllegalStateException("We have already created the child nodes for graph index " + (graphIndex + 1));}
                if (graphIndex + 1 < numGraphs) {
                    for (Arrow childArrow : childArrows.get(graphIndex + 1)) {
                        childNodes.add(new GNode(this, childArrow));
                    }
                }
                // Once the child nodes have been created, set the "explored" flag to true so we don't redo this
                explored = true;
            }
        }

        public void search() {
            GNode root = new GNode(null, null);
            root.makeChildNodes();
            GNode currentNode = root;

            while (root.getChildNodes().size() > 0) {
                int graphIndex = currentNode.getGraphIndex();

                // if we're at the final GNode, and the score has improved*, update bound and (reversed) list of arrows
                // *if the score has stayed the same, still update the list of arrows - important for case where "do nothing" is best
                if (graphIndex == numGraphs - 1 & currentNode.getScoreBound() >= bestScoreSoFar) {
                    bestScoreSoFar = currentNode.getScoreBound();
                    bestArrowsSoFar.clear();
                    GNode catalog = currentNode;
                    for (int i = 0; i < numGraphs; i++) {
                        bestArrowsSoFar.add(catalog.getArrow());
                        catalog = catalog.getParentNode();
                    }
                }

                // if we haven't been to this node yet, create the childNodes and set 'explored' to true
                if (!currentNode.getExplored()) {
                    currentNode.makeChildNodes();
                }

                // prune any options with scoreBounds smaller than the best option so far, or the outerBound
                List<GNode> badChildren = new ArrayList<>();
                if (currentNode.getChildNodes().size() > 0) {
                    for (GNode option : currentNode.getChildNodes()) {
                        if (option.getScoreBound() < bestScoreSoFar | option.getScoreBound() < outerBound) {
                            badChildren.add(option);
                        }
                    }
                    for (GNode badChild : badChildren) {
                        currentNode.pruneChildNode(badChild);
                    }
                }

                // THE FOLLOWING THREE OPTIONS ARE MUTUALLY EXCLUSIVE & EXHAUSTIVE
                // A: if we've already explored all options, regress to the parent node
                if (currentNode.getChildNodes().size() == 0 && currentNode != root) {
                    GNode badChild = currentNode;
                    currentNode = currentNode.getParentNode();
                    currentNode.pruneChildNode(badChild);
                }

                // B: if there's only one option, progress to it
                else if (currentNode.getChildNodes().size() == 1) {
                    currentNode = currentNode.getChildNodes().get(0);
                }

                // C: if there are multiple options, pick the best one and progress
                else if (currentNode.getChildNodes().size() > 1) {

                    GNode nextNode = currentNode.getChildNodes().get(0);

                    for (GNode option : currentNode.getChildNodes()) {
                        if (option.getScoreBound() > nextNode.getScoreBound()) {
                            nextNode = option;
                        }
                    }

                    currentNode = nextNode;
                }

            }
            Collections.reverse(bestArrowsSoFar);

        }

    }


    // Get all adj that are connected to Y by an undirected edge and not adjacent to X.
    private List<Node> getTNeighbors(Node x, Node y, int graphIndex) {
        List<Edge> yEdges = graphConfiguration.getGraph(graphIndex).getEdges(y);
        List<Node> tNeighbors = new ArrayList<>();

        for (Edge edge : yEdges) {
            if (!Edges.isUndirectedEdge(edge)) {
                continue;
            }

            Node z = edge.getDistalNode(y);

            if (graphConfiguration.getGraph(graphIndex).isAdjacentTo(z, x)) {
                continue;
            }

            tNeighbors.add(z);
        }

        return tNeighbors;
    }

    // Get all adj that are connected to Y.
    private Set<Node> getNeighbors(Node y, int graphIndex) {
        List<Edge> yEdges = graphConfiguration.getGraph(graphIndex).getEdges(y);
        Set<Node> neighbors = new HashSet<>();

        for (Edge edge : yEdges) {
            if (!Edges.isUndirectedEdge(edge)) {
                continue;
            }

            Node z = edge.getDistalNode(y);

            neighbors.add(z);
        }

        return neighbors;
    }

    // Evaluate the Insert(X, Y, T) operator (Definition 12 from Chickering, 2002).
    private double insertEval(Node x, Node y, Set<Node> t, Set<Node> naYX,
                              Map<Node, Integer> hashIndices, int graphIndex) {
        if (x == y) throw new IllegalArgumentException();
        Set<Node> set = new HashSet<>(naYX);
        set.addAll(t);
        set.addAll(graphConfiguration.getGraph(graphIndex).getParents(y));
        return scoreGraphChange(y, set, x, hashIndices, graphIndex);
    }

    // Evaluate the Delete(X, Y, T) operator (Definition 12 from Chickering, 2002).
    private double deleteEval(Node x, Node y, Set<Node> diff, Set<Node> naYX,
                              Map<Node, Integer> hashIndices, int graphIndex) {
        Set<Node> set = new HashSet<>(diff);
        set.addAll(graphConfiguration.getGraph(graphIndex).getParents(y));
        set.remove(x);
        return -scoreGraphChange(y, set, x, hashIndices, graphIndex);
    }

    // Do an actual insertion. (Definition 12 from Chickering, 2002).
    private boolean insert(Node x, Node y, Set<Node> T, double bump, int graphIndex) {
        if (graphConfiguration.getGraph(graphIndex).isAdjacentTo(x, y)) {
            return false; // The initial graph may already have put this edge in the graph.
        }

        Edge trueEdge = null;

        // todo fix this
        if (trueGraph != null) {
            Node _x = trueGraph.getNode(x.getName());
            Node _y = trueGraph.getNode(y.getName());
            trueEdge = trueGraph.getEdge(_x, _y);
        }

        if (boundGraph != null && !boundGraph.isAdjacentTo(x, y)) return false;

        graphConfiguration.getGraph(graphIndex).addDirectedEdge(x, y);

        if (verbose) {
            String label = trueGraph != null && trueEdge != null ? "*" : "";
            TetradLogger.getInstance().log("insertedEdges",
                    graphConfiguration.getGraph(graphIndex).getNumEdges() + ". INSERT " +
                            graphConfiguration.getGraph(graphIndex).getEdge(x, y) +
                            " " + T + " " + bump + " " + label);
        }

        int numEdges = graphConfiguration.getGraph(graphIndex).getNumEdges();

//        if (verbose) {
        if (numEdges % 1000 == 0) out.println("Num edges added: " + numEdges);
//        }

        if (verbose) {
            String label = trueGraph != null && trueEdge != null ? "*" : "";
            out.println(graphConfiguration.getGraph(graphIndex).getNumEdges() + ". INSERT " +
                    graphConfiguration.getGraph(graphIndex).getEdge(x, y) +
                    " " + T + " " + bump + " " + label
                    + " degree = " + GraphUtils.getDegree(graphConfiguration.getGraph(graphIndex))
                    + " indegree = " + GraphUtils.getIndegree(graphConfiguration.getGraph(graphIndex)));
        }

        for (Node _t : T) {
            graphConfiguration.getGraph(graphIndex).removeEdge(_t, y);
            if (boundGraph != null && !boundGraph.isAdjacentTo(_t, y)) continue;

            graphConfiguration.getGraph(graphIndex).addDirectedEdge(_t, y);

            if (verbose) {
                String message = "--- Directing " + graphConfiguration.getGraph(graphIndex).getEdge(_t, y);
                TetradLogger.getInstance().log("directedEdges", message);
                out.println(message);
            }
        }

        return true;
    }

    Set<Edge> removedEdges = new HashSet<>();

    // Do an actual deletion (Definition 13 from Chickering, 2002).
    private boolean delete(Node x, Node y, Set<Node> H, double bump, Set<Node> naYX, int graphIndex) {
        Edge trueEdge = null;

        if (trueGraph != null) {
            Node _x = trueGraph.getNode(x.getName());
            Node _y = trueGraph.getNode(y.getName());
            trueEdge = trueGraph.getEdge(_x, _y);
        }

        Edge oldxy = graphConfiguration.getGraph(graphIndex).getEdge(x, y);

        Set<Node> diff = new HashSet<>(naYX);
        diff.removeAll(H);

        graphConfiguration.getGraph(graphIndex).removeEdge(oldxy);
        removedEdges.add(Edges.undirectedEdge(x, y));

//        if (verbose) {
        int numEdges = graphConfiguration.getGraph(graphIndex).getNumEdges();
        if (numEdges % 1000 == 0) out.println("Num edges (backwards) = " + numEdges);
//        }

        if (verbose) {
            String label = trueGraph != null && trueEdge != null ? "*" : "";
            String message = (graphConfiguration.getGraph(graphIndex).getNumEdges()) + ". DELETE " + x + "-->" + y +
                    " H = " + H + " NaYX = " + naYX + " diff = " + diff + " (" + bump + ") " + label;
            TetradLogger.getInstance().log("deletedEdges", message);
            out.println(message);
        }

        for (Node h : H) {
            if (graphConfiguration.getGraph(graphIndex).isParentOf(h, y) || graphConfiguration.getGraph(graphIndex).isParentOf(h, x))
                continue;

            Edge oldyh = graphConfiguration.getGraph(graphIndex).getEdge(y, h);

            graphConfiguration.getGraph(graphIndex).removeEdge(oldyh);

            graphConfiguration.getGraph(graphIndex).addEdge(Edges.directedEdge(y, h));

            if (verbose) {
                TetradLogger.getInstance().log("directedEdges", "--- Directing " + oldyh + " to " +
                        graphConfiguration.getGraph(graphIndex).getEdge(y, h));
                out.println("--- Directing " + oldyh + " to " + graphConfiguration.getGraph(graphIndex).getEdge(y, h));
            }

            Edge oldxh = graphConfiguration.getGraph(graphIndex).getEdge(x, h);

            if (Edges.isUndirectedEdge(oldxh)) {
                graphConfiguration.getGraph(graphIndex).removeEdge(oldxh);

                graphConfiguration.getGraph(graphIndex).addEdge(Edges.directedEdge(x, h));

                if (verbose) {
                    TetradLogger.getInstance().log("directedEdges", "--- Directing " + oldxh + " to " +
                            graphConfiguration.getGraph(graphIndex).getEdge(x, h));
                    out.println("--- Directing " + oldxh + " to " + graphConfiguration.getGraph(graphIndex).getEdge(x, h));
                }
            }
        }

        return true;
    }

    // Test if the candidate insertion is a valid operation
    // (Theorem 15 from Chickering, 2002).
    private boolean validInsert(Node x, Node y, Set<Node> T, Set<Node> naYX, int graphIndex) {
        boolean violatesKnowledge = false;

        if (existsKnowledge()) {
            if (knowledge.isForbidden(x.getName(), y.getName())) {
                violatesKnowledge = true;
            }

            for (Node t : T) {
                if (knowledge.isForbidden(t.getName(), y.getName())) {
                    violatesKnowledge = true;
                }
            }
        }

        Set<Node> union = new HashSet<>(T);
        union.addAll(naYX);
        boolean clique = isClique(union, graphIndex);
        boolean noCycle = !existsUnblockedSemiDirectedPath(y, x, union, cycleBound, graphIndex);
        return clique && noCycle && !violatesKnowledge;
    }

    private boolean validDelete(Node x, Node y, Set<Node> H, Set<Node> naYX, int graphIndex) {
        boolean violatesKnowledge = false;

        if (existsKnowledge()) {
            for (Node h : H) {
                if (knowledge.isForbidden(x.getName(), h.getName())) {
                    violatesKnowledge = true;
                }

                if (knowledge.isForbidden(y.getName(), h.getName())) {
                    violatesKnowledge = true;
                }
            }
        }

        Set<Node> diff = new HashSet<>(naYX);
        diff.removeAll(H);
        return isClique(diff, graphIndex) && !violatesKnowledge;
    }

    // Adds edges required by knowledge.
    private void addRequiredEdges(Graph graph) {
        if (!existsKnowledge()) return;

        for (Iterator<KnowledgeEdge> it = getKnowledge().requiredEdgesIterator(); it.hasNext(); ) {
            KnowledgeEdge next = it.next();

            Node nodeA = graph.getNode(next.getFrom());
            Node nodeB = graph.getNode(next.getTo());

            if (!graph.isAncestorOf(nodeB, nodeA)) {
                graph.removeEdges(nodeA, nodeB);
                graph.addDirectedEdge(nodeA, nodeB);
                TetradLogger.getInstance().log("insertedEdges", "Adding edge by knowledge: " + graph.getEdge(nodeA, nodeB));
            }
        }
        for (Edge edge : graph.getEdges()) {
            final String A = edge.getNode1().getName();
            final String B = edge.getNode2().getName();

            if (knowledge.isForbidden(A, B)) {
                Node nodeA = edge.getNode1();
                Node nodeB = edge.getNode2();
                if (nodeA == null || nodeB == null) throw new NullPointerException();

                if (graph.isAdjacentTo(nodeA, nodeB) && !graph.isChildOf(nodeA, nodeB)) {
                    if (!graph.isAncestorOf(nodeA, nodeB)) {
                        graph.removeEdges(nodeA, nodeB);
                        graph.addDirectedEdge(nodeB, nodeA);
                        TetradLogger.getInstance().log("insertedEdges", "Adding edge by knowledge: " + graph.getEdge(nodeB, nodeA));
                    }
                }

                if (!graph.isChildOf(nodeA, nodeB) && getKnowledge().isForbidden(nodeA.getName(), nodeB.getName())) {
                    if (!graph.isAncestorOf(nodeA, nodeB)) {
                        graph.removeEdges(nodeA, nodeB);
                        graph.addDirectedEdge(nodeB, nodeA);
                        TetradLogger.getInstance().log("insertedEdges", "Adding edge by knowledge: " + graph.getEdge(nodeB, nodeA));
                    }
                }
            } else if (knowledge.isForbidden(B, A)) {
                Node nodeA = edge.getNode2();
                Node nodeB = edge.getNode1();
                if (nodeA == null || nodeB == null) throw new NullPointerException();

                if (graph.isAdjacentTo(nodeA, nodeB) && !graph.isChildOf(nodeA, nodeB)) {
                    if (!graph.isAncestorOf(nodeA, nodeB)) {
                        graph.removeEdges(nodeA, nodeB);
                        graph.addDirectedEdge(nodeB, nodeA);
                        TetradLogger.getInstance().log("insertedEdges", "Adding edge by knowledge: " + graph.getEdge(nodeB, nodeA));
                    }
                }
                if (!graph.isChildOf(nodeA, nodeB) && getKnowledge().isForbidden(nodeA.getName(), nodeB.getName())) {
                    if (!graph.isAncestorOf(nodeA, nodeB)) {
                        graph.removeEdges(nodeA, nodeB);
                        graph.addDirectedEdge(nodeB, nodeA);
                        TetradLogger.getInstance().log("insertedEdges", "Adding edge by knowledge: " + graph.getEdge(nodeB, nodeA));
                    }
                }
            }
        }
    }

    // Use background knowledge to decide if an insert or delete operation does not orient edges in a forbidden
    // direction according to prior knowledge. If some orientation is forbidden in the subset, the whole subset is
    // forbidden.
    private boolean validSetByKnowledge(Node y, Set<Node> subset) {
        for (Node node : subset) {
            if (getKnowledge().isForbidden(node.getName(), y.getName())) {
                return false;
            }
        }
        return true;
    }

    // Find all adj that are connected to Y by an undirected edge that are adjacent to X (that is, by undirected or
    // directed edge).
    private Set<Node> getNaYX(Node x, Node y, int graphIndex) {
        List<Node> adj = graphConfiguration.getGraph(graphIndex).getAdjacentNodes(y);
        Set<Node> nayx = new HashSet<>();

        for (Node z : adj) {
            if (z == x) continue;
            Edge yz = graphConfiguration.getGraph(graphIndex).getEdge(y, z);
            if (!Edges.isUndirectedEdge(yz)) continue;
            if (!graphConfiguration.getGraph(graphIndex).isAdjacentTo(z, x)) continue;
            nayx.add(z);
        }

        return nayx;
    }

    Set<Edge> cliqueEdges = new HashSet<>();

    // Returns true iif the given set forms a clique in the given graph.
    private boolean isClique(Set<Node> nodes, int graphIndex) {
        List<Node> _nodes = new ArrayList<>(nodes);
        for (int i = 0; i < _nodes.size() - 1; i++) {
            for (int j = i + 1; j < _nodes.size(); j++) {
                if (!graphConfiguration.getGraph(graphIndex).isAdjacentTo(_nodes.get(i), _nodes.get(j))) {
                    return false;
                }
            }
        }

        return true;
    }

    // Returns true if a path consisting of undirected and directed edges toward 'to' exists of
    // length at most 'bound'. Cycle checker in other words.
    private boolean existsUnblockedSemiDirectedPath(Node from, Node to, Set<Node> cond, int bound, int graphIndex) {
        Queue<Node> Q = new LinkedList<>();
        Set<Node> V = new HashSet<>();
        Q.offer(from);
        V.add(from);
        Node e = null;
        int distance = 0;

        while (!Q.isEmpty()) {
            Node t = Q.remove();
            if (t == to) {
                return true;
            }

            if (e == t) {
                e = null;
                distance++;
                if (distance > (bound == -1 ? 1000 : bound)) return false;
            }

            for (Node u : graphConfiguration.getGraph(graphIndex).getAdjacentNodes(t)) {
                Edge edge = graphConfiguration.getGraph(graphIndex).getEdge(t, u);
                Node c = traverseSemiDirected(t, edge);
                if (c == null) continue;
                if (cond.contains(c)) continue;

                if (c == to) {
                    return true;
                }

                if (!V.contains(c)) {
                    V.add(c);
                    Q.offer(c);

                    if (e == null) {
                        e = u;
                    }
                }
            }
        }

        return false;
    }

    // Used to find semidirected paths for cycle checking.
    private static Node traverseSemiDirected(Node node, Edge edge) {
        if (node == edge.getNode1()) {
            if (edge.getEndpoint1() == Endpoint.TAIL) {
                return edge.getNode2();
            }
        } else if (node == edge.getNode2()) {
            if (edge.getEndpoint2() == Endpoint.TAIL) {
                return edge.getNode1();
            }
        }
        return null;
    }

    // Runs Meek rules on just the changed adj.
    private Set<Node> reorientNode(List<Node> nodes, int graphIndex) {
        addRequiredEdges(graphConfiguration.getGraph(graphIndex));
        return meekOrientRestricted(nodes, getKnowledge(), graphIndex);
    }

    // Runs Meek rules on just the changed adj.
    private Set<Node> meekOrientRestricted(List<Node> nodes, IKnowledge knowledge, int graphIndex) {
        MeekRules rules = new MeekRules();
        rules.setKnowledge(knowledge);
        rules.setUndirectUnforcedEdges(true);
        rules.orientImplied(graphConfiguration.getGraph(graphIndex), nodes);
        return rules.getVisited();
    }

    // Maps adj to their indices for quick lookup.
    private void buildIndexing(List<Node> nodes) {
        this.hashIndices = new ConcurrentHashMap<>();

        int i = -1;

        for (Node n : nodes) {
            this.hashIndices.put(n, ++i);
        }
    }

    // Removes information associated with an edge x->y.
    private synchronized void clearArrow(Node x, Node y, int graphIndex) {
        final OrderedPair<Node> pair = new OrderedPair<>(x, y);
        final Set<Arrow> lookupArrows = this.lookupArrowsList.get(graphIndex).get(pair);

        this.lookupArrowsList.get(graphIndex).remove(pair);
    }

    // Adds the given arrow for the adjacency i->j. These all are for i->j but may have
    // different T or H or NaYX sets, and so different bumps.
    private void addLookupArrow(Node i, Node j, Arrow arrow, int graphIndex) {
        OrderedPair<Node> pair = new OrderedPair<>(i, j);
        Set<Arrow> arrows = lookupArrowsList.get(graphIndex).get(pair);

        if (arrows == null) {
            arrows = new ConcurrentSkipListSet<>();
            lookupArrowsList.get(graphIndex).put(pair, arrows);
        }

        arrows.add(arrow);
    }

    //===========================SCORING METHODS===================//

    /**
     * Scores the given DAG, up to a constant.
     */
    public double scoreDag(GraphConfiguration graphConfiguration, int graphIndex) {
        buildIndexing(graphConfiguration.getGraph(graphIndex).getNodes());

        double _score = 0.0;

        for (Node y : graphConfiguration.getGraph(graphIndex).getNodes()) {
            Set<Node> parents = new HashSet<>(graphConfiguration.getGraph(graphIndex).getParents(y));
            int parentIndices[] = new int[parents.size()];
            Iterator<Node> pi = parents.iterator();
            int count = 0;

            while (pi.hasNext()) {
                Node nextParent = pi.next();
                parentIndices[count++] = hashIndices.get(nextParent);
            }

            int yIndex = hashIndices.get(y);
            _score += scoreList.get(graphIndex).localScore(yIndex, parentIndices);
        }

        return _score;
    }

    private double scoreGraphChange(Node y, Set<Node> parents,
                                    Node x, Map<Node, Integer> hashIndices, int graphIndex) {
        int yIndex = hashIndices.get(y);

        if (x == y) throw new IllegalArgumentException();
        if (parents.contains(y)) throw new IllegalArgumentException();

        int[] parentIndices = new int[parents.size()];

        int count = 0;
        for (Node parent : parents) {
            parentIndices[count++] = hashIndices.get(parent);
        }

        return scoreList.get(graphIndex).localScoreDiff(hashIndices.get(x), yIndex, parentIndices);
    }

    private List<Node> getVariables() {
        return variables;
    }

}