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

package edu.cmu.tetrad.cmd;

import edu.cmu.tetrad.algcomparison.statistic.*;
import edu.cmu.tetrad.bayes.BayesIm;
import edu.cmu.tetrad.bayes.BayesPm;
import edu.cmu.tetrad.bayes.MlBayesEstimator;
import edu.cmu.tetrad.data.*;
import edu.cmu.tetrad.graph.*;
import edu.cmu.tetrad.search.*;
import edu.cmu.tetrad.sem.LargeScaleSimulation;
import edu.cmu.tetrad.util.RandomUtil;
import edu.cmu.tetrad.util.TetradLogger;
import edu.cmu.tetrad.util.TextTable;

import java.io.*;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Set;

/**
 * Runs several algorithm from Tetrad. Documentation is available
 * in the wiki of the Tetrad project on GitHub. This will be replaced by
 * the package tetrad-cli.
 *
 * @author Joseph Ramsey
 */
public final class TetradCmd {
    private String algorithmName;
    private String dataFileName;
    private String[] dataFileNameArray;
    private String knowledgeFileName;
    private String dataTypeName;
    private String graphXmlFilename;
    private String graphTxtFilename;
    private String initialGraphTxtFilename;
    private String trueGraphTxtFilename;
    private String comparisonGraphTxtFilename;
    private int depth = -1;
    private double significance = 0.05;
    private DataSet data;
    private List<DataSet> dataSetList;
    private ICovarianceMatrix covarianceMatrix;
    private String outputStreamPath;
    private PrintStream out = System.out;
    private String seed;
    private String numNodes = "5";
    private String numEdges = "5";
    private IKnowledge knowledge = new Knowledge2();
    private boolean whitespace = false;
    private boolean verbose = false;
    private double samplePrior = 1.0;
    private double structurePrior = 1.0;
    private double penaltyDiscount = 1.0;
    private TestType testType = TestType.TETRAD_DELTA;
    private Graph initialGraph;
    private Graph trueInputGraph;
    private Graph comparisonGraph;
    private boolean rfciUsed = false;
    private boolean nodsep = false;
    private boolean useCovariance = true;
    private boolean silent = false;
    private boolean useConditionalCorrelation = false;

    // Gest simulation parameters:
    private int[] numNodesArray = {30};
    private double[] numEdgesFactorArray = {1};
    private int[] kArray = {3};
    private double[] graphDistanceFactorArray = {0.1};
    private int[] sampleSizeArray = {100};
    private double[] transferPenaltyArray = {3};
    private double[] penaltyDiscountArray = {4};
    private int numRuns = 10;
    private boolean[] weightTransferBySampleArray = {false}; // this doesn't seem to affect performance much
    private boolean[] bumpMinTransferArray = {true};
    private boolean[] faithfulnessAssumedArray = {true, false};

    public TetradCmd(String[] argv) {
        readArguments(new StringArrayTokenizer(argv));

        setOutputStream();
//        loadDataSelect();
        runAlgorithm();

        if (out != System.out) {
            out.close();
        }
    }

    private void setOutputStream() {
        if (outputStreamPath == null) {
            return;
        }

        File file = new File(outputStreamPath);

        try {
            out = new PrintStream(new FileOutputStream(file));
        } catch (FileNotFoundException e) {
            throw new IllegalStateException(
                    "Could not create a logfile at location " +
                            file.getAbsolutePath()
            );
        }
    }

    private void readArguments(StringArrayTokenizer tokenizer) {
        while (tokenizer.hasToken()) {
            String token = tokenizer.nextToken();

            if ("-data".equalsIgnoreCase(token)) {
                String argument = tokenizer.nextToken();

                if (argument.startsWith("-")) {
                    throw new IllegalArgumentException(
                            "'-data' tag must be followed " +
                                    "by an argument indicating the path to the data " +
                                    "file."
                    );
                }

                dataFileName = argument;
                useCovariance = false;
            } else if ("-dataList".equalsIgnoreCase(token)) {
                String s = tokenizer.nextToken();

                if (s.startsWith("-")) {
                    throw new IllegalArgumentException(
                            "'-dataList' tag must be followed " +
                                    "by an argument indicating the paths to the data " +
                                    "files (comma-separated, no extra whitespace)."
                    );
                }
                String[] argument = s.split(",");

                dataFileNameArray = argument;
                useCovariance = false;
            } else if ("-covariance".equalsIgnoreCase(token)) {
                String argument = tokenizer.nextToken();

                if (argument.startsWith("-")) {
                    throw new IllegalArgumentException(
                            "'-data' tag must be followed " +
                                    "by an argument indicating the path to the data " +
                                    "file."
                    );
                }

                dataFileName = argument;
                useCovariance = true;
                dataTypeName = "continuous";
            } else if ("-datatype".equalsIgnoreCase(token)) {
                String argument = tokenizer.nextToken();

                if (argument.startsWith("-")) {
                    throw new IllegalArgumentException(
                            "'-datatype' tag must be followed " +
                                    "by either 'discrete' or 'continuous'."
                    );
                }

                dataTypeName = argument;
            } else if ("-algorithm".equalsIgnoreCase(token)) {
                String argument = tokenizer.nextToken();

                if (argument.startsWith("-")) {
                    throw new IllegalArgumentException(
                            "'-algorithm' tag must be followed " +
                                    "by an algorithm name."
                    );
                }

                algorithmName = argument;
            } else if ("-depth".equalsIgnoreCase(token)) {
                try {
                    String argument = tokenizer.nextToken();

                    if (argument == null) {
                        throw new NumberFormatException();
                    }

                    this.depth = Integer.parseInt(argument);

                    if (this.depth < -1) {
                        throw new IllegalArgumentException(
                                "'depth' must be followed " +
                                        "by an integer >= -1 (-1 means unlimited)."
                        );
                    }
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(
                            "'depth' must be followed " +
                                    "by an integer >= -1 (-1 means unlimited)."
                    );
                }
            } else if ("-significance".equalsIgnoreCase(token)) {
                try {
                    String argument = tokenizer.nextToken();

                    if (argument.startsWith("-")) {
                        throw new NumberFormatException();
                    }

                    this.significance = Double.parseDouble(argument);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(
                            "'-significance' must be " +
                                    "followed by a number in the range [0.0, 1.0]."
                    );
                }
            } else if ("-verbose".equalsIgnoreCase(token)) {
                this.verbose = true;
            } else if ("-outfile".equalsIgnoreCase(token)) {
                String argument = tokenizer.nextToken();

                if (argument.startsWith("-")) {
                    throw new IllegalArgumentException(
                            "'-outfile' tag must be " +
                                    "followed  by an argument indicating the path to the " +
                                    "data file."
                    );
                }

                outputStreamPath = argument;
            } else if ("-seed".equalsIgnoreCase(token)) {
                String argument = tokenizer.nextToken();

                if (argument.startsWith("-")) {
                    throw new IllegalArgumentException(
                            "-seed must be followed by an integer (long) value."
                    );
                }

                seed = argument;
            } else if ("-numNodes".equals(token)) {
                String argument = tokenizer.nextToken();

                if (argument.startsWith("-")) {
                    throw new IllegalArgumentException(
                            "-numNodes must be followed by an integer >= 3.");
                }

                numNodes = argument;
            } else if ("-numEdges".equals(token)) {
                String argument = tokenizer.nextToken();

                if (argument.startsWith("-")) {
                    throw new IllegalArgumentException(
                            "-numEdges must be followed by an integer >= 0.");
                }

                numEdges = argument;
            } else if ("-knowledge".equals(token)) {
                String argument = tokenizer.nextToken();

                if (argument.startsWith("-")) {
                    throw new IllegalArgumentException(
                            "'-knowledge' tag must be followed " +
                                    "by an argument indicating the path to the knowledge " +
                                    "file."
                    );
                }

                knowledgeFileName = argument;
            } else if ("-testtype".equals(token)) {
                String argument = tokenizer.nextToken();

                if (argument.startsWith("-")) {
                    throw new IllegalArgumentException(
                            "'-testType' tag must be followed by 'delta' or 'wishart'");
                }

                switch (argument) {
                    case "delta":
                        testType = TestType.TETRAD_DELTA;
                        break;
                    case "wishart":
                        testType = TestType.TETRAD_WISHART;
                        break;
                    default:
                        throw new IllegalArgumentException("Expecting 'delta' or 'wishart'.");
                }
            } else if ("-graphxml".equals(token)) {
                String argument = tokenizer.nextToken();

                if (argument.startsWith("-")) {
                    throw new IllegalArgumentException(
                            "'-graphxml' tag must be followed " +
                                    "by an argument indicating the path to the file where the graph xml output " +
                                    "is to be written."
                    );
                }

                graphXmlFilename = argument;
            } else if ("-graphtxt".equals(token)) {
                String argument = tokenizer.nextToken();

                if (argument.startsWith("-")) {
                    throw new IllegalArgumentException(
                            "'-graphtxt' tag must be followed " +
                                    "by an argument indicating the path to the file where the graph txt output " +
                                    "is to be written."
                    );
                }

                graphTxtFilename = argument;
            } else if ("-initialgraphtxt".equals(token)) {
                String argument = tokenizer.nextToken();

                if (argument.startsWith("-")) {
                    throw new IllegalArgumentException(
                            "'-initialgraphtxt' tag must be followed " +
                                    "by an argument indicating the path to the file where the graph txt output " +
                                    "is to be written."
                    );
                }

                initialGraphTxtFilename = argument;
            } else if ("-truegraphtxt".equals(token)) {
                String argument = tokenizer.nextToken();

                if (argument.startsWith("-")) {
                    throw new IllegalArgumentException(
                            "'-truegraphtxt' tag must be followed " +
                                    "by an argument indicating the path to the file where the graph txt output " +
                                    "is to be written."
                    );
                }

                trueGraphTxtFilename = argument;
            } else if ("-comparisongraphtxt".equals(token)) {
                String argument = tokenizer.nextToken();

                if (argument.startsWith("-")) {
                    throw new IllegalArgumentException(
                            "'-comparisongraphtxt' tag must be followed " +
                                    "by an argument indicating the path to the file where the graph txt output " +
                                    "is to be written."
                    );
                }

                comparisonGraphTxtFilename = argument;
            } else if ("-whitespace".equals(token)) {
                whitespace = true;
            } else if ("-sampleprior".equals(token)) {
                try {
                    String argument = tokenizer.nextToken();

                    if (argument.startsWith("-")) {
                        throw new IllegalArgumentException(
                                "'-sampleprior' tag must be followed " +
                                        "by an argument indicating the BDEU structure prior."
                        );
                    }

                    samplePrior = Double.parseDouble(argument);

                    if (samplePrior < 0) {
                        throw new IllegalArgumentException("Sample prior must be >= 0.");
                    }
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("Not a number.");
                }
            } else if ("-structureprior".equals(token)) {
                try {
                    String argument = tokenizer.nextToken();

                    if (argument.startsWith("-")) {
                        throw new IllegalArgumentException(
                                "'-structureprior' tag must be followed " +
                                        "by an argument indicating the BDEU sample prior."
                        );
                    }

                    structurePrior = Double.parseDouble(argument);

                    if (structurePrior < 0) {
                        throw new IllegalArgumentException("Structure prior must be >= 0.");
                    }
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("Not a number.");
                }
            } else if ("-penaltydiscount".equals(token)) {
                try {
                    String argument = tokenizer.nextToken();

                    if (argument.startsWith("-")) {
                        throw new IllegalArgumentException(
                                "'-penaltydiscount' tag must be followed " +
                                        "by an argument indicating penalty discount."
                        );
                    }

                    penaltyDiscount = Double.parseDouble(argument);

                    if (penaltyDiscount <= 0) {
                        throw new IllegalArgumentException("Penalty discount must be > 0.");
                    }
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("Not a number.");
                }
            } else if ("-rfci".equalsIgnoreCase(token)) {
                this.rfciUsed = true;
            } else if ("-nodsep".equalsIgnoreCase(token)) {
                this.nodsep = true;            }
            else if ("-silent".equalsIgnoreCase(token)) {
                this.silent = true;
            } else if ("-condcorr".equalsIgnoreCase(token)) {
                this.useConditionalCorrelation = true;
            } else if ("-numNodesArray".equalsIgnoreCase(token)) {
                try {
                    String argument = tokenizer.nextToken();
                    this.numNodesArray = stringToIntArray(argument);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("numNodesArray: Not a number.");
                }
            } else if ("-numEdgesFactorArray".equalsIgnoreCase(token)) {
                try {
                    String argument = tokenizer.nextToken();
                    this.numEdgesFactorArray = stringToDoubleArray(argument);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("numEdgesFactorArray: Not a number.");
                }
            } else if ("-kArray".equalsIgnoreCase(token)) {
                try {
                    String argument = tokenizer.nextToken();
                    this.kArray = stringToIntArray(argument);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("kArray: Not a number.");
                }
            } else if ("-graphDistanceFactorArray".equalsIgnoreCase(token)) {
                try {
                    String argument = tokenizer.nextToken();
                    this.graphDistanceFactorArray = stringToDoubleArray(argument);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("graphDistanceFactorArray: Not a number.");
                }
            } else if ("-sampleSizeArray".equalsIgnoreCase(token)) {
                try {
                    String argument = tokenizer.nextToken();
                    this.sampleSizeArray = stringToIntArray(argument);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("sampleSizeArray: Not a number.");
                }
            } else if ("-transferPenaltyArray".equalsIgnoreCase(token)) {
                try {
                    String argument = tokenizer.nextToken();
                    this.transferPenaltyArray = stringToDoubleArray(argument);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("transferPenaltyArray: Not a number.");
                }
            } else if ("-penaltyDiscountArray".equalsIgnoreCase(token)) {
                try {
                    String argument = tokenizer.nextToken();
                    this.penaltyDiscountArray = stringToDoubleArray(argument);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("penaltyDiscountArray: Not a number.");
                }
            } else if ("-weightTransferBySampleArray".equalsIgnoreCase(token)) {
                try {
                    String argument = tokenizer.nextToken();
                    this.weightTransferBySampleArray = stringToBooleanArray(argument);
                } catch (Exception e) {
                    throw new IllegalArgumentException("weightTransferBySampleArray: Not a boolean.");
                }
            } else if ("-bumpMinTransferArray".equalsIgnoreCase(token)) {
                try {
                    String argument = tokenizer.nextToken();
                    this.bumpMinTransferArray = stringToBooleanArray(argument);
                } catch (Exception e) {
                    throw new IllegalArgumentException("bumpMinTransferArray: Not a boolean.");
                }
            } else if ("-faithfulnessAssumedArray".equalsIgnoreCase(token)) {
                try {
                    String argument = tokenizer.nextToken();
                    this.faithfulnessAssumedArray = stringToBooleanArray(argument);
                } catch (Exception e) {
                    throw new IllegalArgumentException("faithfulnessAssumedArray: Not a boolean.");
                }
            } else if ("-numRuns".equalsIgnoreCase(token)) {
                try {
                    String argument = tokenizer.nextToken();
                    this.numRuns = Integer.parseInt(argument);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("numRuns: Not a number.");
                }
            } else {
                throw new IllegalArgumentException(
                        "Unexpected argument: " + token);
            }

        }
    }

    private void loadDataList() {
        if (dataFileNameArray == null) {
            throw new IllegalStateException("No lis of data files was specified.");
        }

        this.dataSetList = new ArrayList<>();

        for (int i = 0; i < dataFileNameArray.length; i++) {
            this.dataFileName = dataFileNameArray[i];
            loadData();
            dataSetList.add(data);
        }

    }


    private void compareGraphs() {

        if (trueGraphTxtFilename == null) {
            throw new IllegalStateException("No true graph was specified.");
        } else {
            trueInputGraph = GraphUtils.loadGraphTxt(new File(trueGraphTxtFilename));
        }
        if (comparisonGraphTxtFilename == null) {
            throw new IllegalStateException("No comparison graph was specified.");
        } else {
            comparisonGraph = GraphUtils.loadGraphTxt(new File(comparisonGraphTxtFilename));
        }
        if (outputStreamPath == null) {
            throw new IllegalStateException("No output file was specified.");
        } else {
            setOutputStream();
        }

        trueInputGraph = GraphUtils.replaceNodes(trueInputGraph, comparisonGraph.getNodes());

        AdjacencyPrecision ap = new AdjacencyPrecision();
        AdjacencyRecall ar = new AdjacencyRecall();
        ArrowheadPrecision arp = new ArrowheadPrecision();
        ArrowheadRecall arr = new ArrowheadRecall();
        MathewsCorrAdj mca = new MathewsCorrAdj();
        MathewsCorrArrow mcar = new MathewsCorrArrow();
        F1Adj f1a = new F1Adj();
        F1Arrow f1ar = new F1Arrow();
        SHD shd = new SHD();

        // write header to file
        out.println("trueNumEdges \ttrueNumNodes \tcomparisonNumEdges \tcomparisonNumNodes " +
                "\tadjacencyPrecision \tadjacencyRecall \tarrowheadPrecision \tarrowheadRecall \tmathewsCorrAdj " +
                "\tmathewsCorrArrow \tf1Adj \tf1Arrow \tshd");

        double adjacencyPrecisionF = ap.getValue(trueInputGraph, comparisonGraph);
        double adjacencyRecallF = ar.getValue(trueInputGraph, comparisonGraph);
        double arrowheadPrecisionF = arp.getValue(trueInputGraph, comparisonGraph);
        double arrowheadRecallF = arr.getValue(trueInputGraph, comparisonGraph);
        double mathewsCorrAdjF = mca.getValue(trueInputGraph, comparisonGraph);
        double mathewsCorrArrowF = mcar.getValue(trueInputGraph, comparisonGraph);
        double f1AdjF = f1a.getValue(trueInputGraph, comparisonGraph);
        double f1ArrowF = f1ar.getValue(trueInputGraph, comparisonGraph);
        double shd1F = shd.getValue(trueInputGraph, comparisonGraph);

        // write comparison output to file
        out.println(trueInputGraph.getNumEdges() + "\t" +
                trueInputGraph.getNumNodes() + "\t" +
                comparisonGraph.getNumEdges() + "\t" +
                comparisonGraph.getNumNodes() + "\t" +
                adjacencyPrecisionF + "\t" +
                adjacencyRecallF + "\t" +
                arrowheadPrecisionF + "\t" +
                arrowheadRecallF + "\t" +
                mathewsCorrAdjF + "\t" +
                mathewsCorrArrowF + "\t" +
                f1AdjF + "\t" +
                f1ArrowF + "\t" +
                shd1F + "\t");
    }

    private void loadData() {
        if (dataFileName == null) {
            throw new IllegalStateException("No data file was specified.");
        }

        if (dataTypeName == null) {
            throw new IllegalStateException(
                    "No data type (continuous/discrete) " + "was specified.");
        }

        outPrint("Loading data from " + dataFileName + ".");

//        if ("continuous".equalsIgnoreCase(dataTypeName)) {
//            outPrint("Data type = continuous.");
//        } else if ("discrete".equalsIgnoreCase(dataTypeName)) {
//            outPrint("Data type = discrete.");
//        } else {
//            throw new IllegalStateException(
//                    "Data type was expected to be either " +
//                            "'continuous' or 'discrete'."
//            );
//        }

        File file = new File(dataFileName);

        try {
            try {
//                    List<Node> knownVariables = null;
//                    RectangularDataSet data = DataLoaders.loadDiscreteData(file,
//                            DelimiterType.WHITESPACE_OR_COMMA, "//",
//                            knownVariables);

                DataReader reader = new DataReader();

                //NOTE: Lizzie set this to zero so she could run GEST. It should be Integer.MAX_VALUE
                reader.setMaxIntegralDiscrete(0);

                if (whitespace) {
                    reader.setDelimiter(DelimiterType.WHITESPACE);
                } else {
                    reader.setDelimiter(DelimiterType.TAB);
                }

                if (useCovariance) {
                    ICovarianceMatrix cov = reader.parseCovariance(file);
                    this.covarianceMatrix = cov;
                } else {
                    DataSet data = reader.parseTabular(file);
                    outPrint("# variables = " + data.getNumColumns() +
                            ", # cases = " + data.getNumRows());
                    this.data = data;
                }

//                systemPrint(data);

                if (initialGraphTxtFilename != null) {
                    initialGraph = GraphUtils.loadGraphTxt(new File(initialGraphTxtFilename));
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        } catch (RuntimeException e) {
            throw new IllegalStateException(
                    "Could not load file at " + file.getAbsolutePath());
        }
    }

    private void loadKnowledge() {
        if (knowledgeFileName == null) {
            throw new IllegalStateException("No knowledge file was specified.");
        }

        try {
            File knowledgeFile = new File(knowledgeFileName);

            CharArrayWriter writer = new CharArrayWriter();

            FileReader fr = new FileReader(knowledgeFile);
            int i;

            while ((i = fr.read()) != -1) {
                writer.append((char) i);
            }

            DataReader reader = new DataReader();
            char[] chars = writer.toCharArray();

            String x = new String(chars);
            systemPrint(x);

            this.knowledge = reader.parseKnowledge(chars);
        } catch (Exception e) {
            throw new RuntimeException("Couldn't read knowledge.");
        }
    }

    private void systemPrint(String x) {
        if (!silent) {
            System.out.println(x);
        }
    }

    private void outPrint(String x) {
        if (!silent) {
            out.println(x);
        }
    }

    private void runAlgorithm() {

        if (dataFileName != null) {
            loadData();
        }

        if (dataFileNameArray != null) {
            loadDataList();
        }

        if (knowledgeFileName != null) {
            loadKnowledge();
        }

        if ("pc".equalsIgnoreCase(algorithmName)) {
            runPc();
        } else if ("pc.stable".equalsIgnoreCase(algorithmName)) {
            runPcStable();
        } else if ("cpc".equalsIgnoreCase(algorithmName)) {
            runCpc();
        } else if ("fci".equalsIgnoreCase(algorithmName)) {
            runFci();
        } else if ("cfci".equalsIgnoreCase(algorithmName)) {
            runCfci();
        } else if ("ccd".equalsIgnoreCase(algorithmName)) {
            runCcd();
        } else if ("fges".equalsIgnoreCase(algorithmName)) {
            runFges();
        } else if ("bayes_est".equalsIgnoreCase(algorithmName)) {
            runBayesEst();
        } else if ("fofc".equalsIgnoreCase(algorithmName)) {
            runFofc();
        } else if ("randomDag".equalsIgnoreCase(algorithmName)) {
            printRandomDag();
        } else if ("testGest".equalsIgnoreCase(algorithmName)) {
            runGestTest();
        } else if ("gest".equalsIgnoreCase(algorithmName)) {
            runGest();
        } else if ("images".equalsIgnoreCase(algorithmName)) {
            runImages();
        } else if ("compareGraphs".equalsIgnoreCase(algorithmName)) {
            compareGraphs();
        } else {
            TetradLogger.getInstance().reset();
            TetradLogger.getInstance().removeOutputStream(System.out);
            throw new IllegalStateException("No algorithm was specified.");
        }

//        TetradLogger.getInstance().setForceLog(false);
        TetradLogger.getInstance().removeOutputStream(System.out);

    }

    private void printRandomDag() {
        if (seed != null) {
            long _seed;

            try {
                _seed = Long.parseLong(seed);
            } catch (NumberFormatException e) {
                throw new RuntimeException("Seed must be an integer (actually, long) value.");
            }

            RandomUtil.getInstance().setSeed(_seed);
        }

        int _numNodes;

        try {
            _numNodes = Integer.parseInt(numNodes);
        } catch (NumberFormatException e) {
            throw new RuntimeException("numNodes must be an integer.");
        }

        int _numEdges;

        try {
            _numEdges = Integer.parseInt(numEdges);
        } catch (NumberFormatException e) {
            throw new RuntimeException("numEdges must be an integer.");
        }

        List<Node> nodes = new ArrayList<>();

        for (int i = 0; i < _numNodes; i++) {
            nodes.add(new ContinuousVariable("X" + (i + 1)));
        }

        Dag dag;

        do {
            dag = new Dag(GraphUtils.randomGraph(nodes, 0, _numEdges, 30,
                    15, 15, false));
        } while (dag.getNumEdges() < _numEdges);

        String xml = GraphUtils.graphToXml(dag);
        systemPrint(xml);
    }

    private void runPc() {
        if (this.data == null && this.covarianceMatrix == null) {
            throw new IllegalStateException("Data did not load correctly.");
        }

        if (verbose) {
            systemPrint("PC");
            systemPrint(getKnowledge().toString());
            systemPrint(getVariables().toString());

            TetradLogger.getInstance().addOutputStream(System.out);

            TetradLogger.getInstance().setEventsToLog("info", "independencies", "knowledgeOrientations",
                    "impliedOrientations", "graph");
//            TetradLogger.getInstance().setForceLog(true);

            TetradLogger.getInstance().log("info", "Testing it.");
        }

        Pc pc = new Pc(getIndependenceTest());
        pc.setDepth(getDepth());
        pc.setKnowledge(getKnowledge());
        pc.setVerbose(verbose);

        // Convert back to Graph..
        Graph resultGraph = pc.search();

        // PrintUtil outputStreamPath problem and graphs.
        outPrint("\nResult graph:");
        outPrint(resultGraph.toString());

        writeGraph(resultGraph);
    }

    private void runPcStable() {
        if (this.data == null && this.covarianceMatrix == null) {
            throw new IllegalStateException("Data did not load correctly.");
        }

        if (verbose) {
            systemPrint("PC-Stable");
            systemPrint(getKnowledge().toString());
            systemPrint(getVariables().toString());

            TetradLogger.getInstance().addOutputStream(System.out);

            TetradLogger.getInstance().setEventsToLog("info", "independencies", "knowledgeOrientations",
                    "impliedOrientations", "graph");
//            TetradLogger.getInstance().setForceLog(true);

            TetradLogger.getInstance().log("info", "Testing it.");
        }

        PcStable pc = new PcStable(getIndependenceTest());
        pc.setDepth(getDepth());
        pc.setKnowledge(getKnowledge());
        pc.setVerbose(verbose);

        // Convert back to Graph..
        Graph resultGraph = pc.search();

        // PrintUtil outputStreamPath problem and graphs.
        outPrint("\nResult graph:");
        outPrint(resultGraph.toString());

        writeGraph(resultGraph);
    }

    private void runFges() {
        if (this.data == null && this.covarianceMatrix == null) {
            throw new IllegalStateException("Data did not load correctly.");
        }

        if (verbose) {
            systemPrint("FGES");
            systemPrint(getKnowledge().toString());
            systemPrint(getVariables().toString());

            TetradLogger.getInstance().addOutputStream(System.out);

            TetradLogger.getInstance().setEventsToLog("info", "independencies", "knowledgeOrientations",
                    "impliedOrientations", "graph");
//            TetradLogger.getInstance().setForceLog(true);

            TetradLogger.getInstance().log("info", "Testing it.");
        }

        Fges fges;

        if (useCovariance) {
            SemBicScore fgesScore = new SemBicScore(covarianceMatrix);
            fgesScore.setPenaltyDiscount(penaltyDiscount);
            fges = new Fges(fgesScore);

        } else {
            if (data.isDiscrete()) {
                BDeuScore score = new BDeuScore(data);
                score.setSamplePrior(samplePrior);
                score.setStructurePrior(structurePrior);

                fges = new Fges(score);
            } else if (data.isContinuous()) {
                SemBicScore score = new SemBicScore(new CovarianceMatrixOnTheFly(data));
                score.setPenaltyDiscount(penaltyDiscount);
                fges = new Fges(score);
            } else {
                throw new IllegalArgumentException();
            }
        }

        if (initialGraph != null) {
            fges.setInitialGraph(initialGraph);
        }

        fges.setKnowledge(getKnowledge());

        // Convert back to Graph..
        Graph resultGraph = fges.search();

        // PrintUtil outputStreamPath problem and graphs.
        outPrint("\nResult graph:");
        outPrint(resultGraph.toString());

        writeGraph(resultGraph);
    }

    private void runCpc() {
        if (this.data == null && this.covarianceMatrix == null) {
            throw new IllegalStateException("Data did not load correctly.");
        }

        if (verbose) {
            systemPrint("CPC");
            systemPrint(getKnowledge().toString());
            systemPrint(getVariables().toString());

            TetradLogger.getInstance().addOutputStream(System.out);

            TetradLogger.getInstance().setEventsToLog("info", "independencies", "knowledgeOrientations",
                    "impliedOrientations", "graph");
//            TetradLogger.getInstance().setForceLog(true);

            TetradLogger.getInstance().log("info", "Testing it.");
        }

        Cpc pc = new Cpc(getIndependenceTest());
        pc.setDepth(getDepth());
        pc.setKnowledge(getKnowledge());
        pc.setVerbose(verbose);

        // Convert back to Graph..
        Graph resultGraph = pc.search();

        // PrintUtil outputStreamPath problem and graphs.
        outPrint("\nResult graph:");
        outPrint(resultGraph.toString());

        writeGraph(resultGraph);
    }

    private void runFci() {
        if (this.data == null && this.covarianceMatrix == null) {
            throw new IllegalStateException("Data did not load correctly.");
        }

        if (verbose) {
            systemPrint("FCI");
            systemPrint(getKnowledge().toString());
            systemPrint(getVariables().toString());

            TetradLogger.getInstance().addOutputStream(System.out);

            TetradLogger.getInstance().setEventsToLog("info", "independencies", "knowledgeOrientations",
                    "impliedOrientations", "graph");
//            TetradLogger.getInstance().setForceLog(true);

            TetradLogger.getInstance().log("info", "Testing it.");
        }

        if (rfciUsed) {
            Rfci fci = new Rfci(getIndependenceTest());
            fci.setDepth(getDepth());
            fci.setKnowledge(getKnowledge());
            fci.setVerbose(verbose);

            // Convert back to Graph..
            Graph resultGraph = fci.search();

            // PrintUtil outputStreamPath problem and graphs.
            outPrint("\nResult graph:");
            outPrint(resultGraph.toString());

            writeGraph(resultGraph);
        } else {
            Fci fci = new Fci(getIndependenceTest());
            fci.setDepth(getDepth());
            fci.setKnowledge(getKnowledge());
            fci.setPossibleDsepSearchDone(!nodsep);
            fci.setVerbose(verbose);

            // Convert back to Graph..
            Graph resultGraph = fci.search();

            // PrintUtil outputStreamPath problem and graphs.
            outPrint("\nResult graph:");
            outPrint(resultGraph.toString());

            writeGraph(resultGraph);

        }
    }

    private void runCfci() {
        if (this.data == null && this.covarianceMatrix == null) {
            throw new IllegalStateException("Data did not load correctly.");
        }

        if (verbose) {
            systemPrint("CFCI");
            systemPrint(getKnowledge().toString());
            systemPrint(getVariables().toString());

            TetradLogger.getInstance().addOutputStream(System.out);

            TetradLogger.getInstance().setEventsToLog("info", "independencies", "colliderOrientations",
                    "impliedOrientations", "graph");
//            TetradLogger.getInstance().setForceLog(true);

            TetradLogger.getInstance().log("info", "Testing it.");
        }

        Cfci fci = new Cfci(getIndependenceTest());
        fci.setDepth(getDepth());
        fci.setKnowledge(getKnowledge());
        fci.setDepth(depth);
        fci.setVerbose(verbose);

        // Convert back to Graph..
        Graph resultGraph = fci.search();

        // PrintUtil outputStreamPath problem and graphs.
        outPrint("\nResult graph:");
        outPrint(resultGraph.toString());

        writeGraph(resultGraph);
    }

    private void runCcd() {
        if (this.data == null && this.covarianceMatrix == null) {
            throw new IllegalStateException("Data did not load correctly.");
        }

        if (verbose) {
            systemPrint("CCD");
            systemPrint(getKnowledge().toString());
            systemPrint(getVariables().toString());

            TetradLogger.getInstance().addOutputStream(System.out);

            TetradLogger.getInstance().setEventsToLog("info", "independencies", "knowledgeOrientations",
                    "impliedOrientations", "graph");
//            TetradLogger.getInstance().setForceLog(true);

            TetradLogger.getInstance().log("info", "Testing it.");
        }

        Ccd ccd = new Ccd(getIndependenceTest());
        ccd.setDepth(getDepth());

        // Convert back to Graph..
        Graph resultGraph = ccd.search();

        // PrintUtil outputStreamPath problem and graphs.
        outPrint("\nResult graph:");
        outPrint(resultGraph.toString());

        writeGraph(resultGraph);
    }

    private void runBayesEst() {
        if (this.data == null && this.covarianceMatrix != null) {
            throw new IllegalStateException("Continuous tabular data required.");
        }

        if (this.data == null) {
            throw new IllegalStateException("Data did not load correctly.");
        }

        if (!this.data.isDiscrete()) {
            outPrint("Please supply discrete data.");
        }

        IndependenceTest independence = new IndTestChiSquare(data, significance);

        Cpc cpc = new Cpc(independence);
        cpc.setVerbose(verbose);
        Graph pattern = cpc.search();

        outPrint("Found this pattern: " + pattern);

        Dag dag = new Dag(SearchGraphUtils.dagFromPattern(pattern));

        outPrint("Chose this DAG: " + dag);

        BayesPm pm = new BayesPm(dag);

        MlBayesEstimator est = new MlBayesEstimator();
        BayesIm im = est.estimate(pm, data);

        outPrint("Estimated IM: " + im);

    }

    private void runFofc() {
        FindOneFactorClusters fofc;

        if (this.data != null) {
            fofc = new FindOneFactorClusters(this.data,
                    this.testType, FindOneFactorClusters.Algorithm.GAP, significance);
            if (!this.data.isContinuous()) {
                outPrint("Please supply continuous data.");
            }
        } else if (this.covarianceMatrix != null) {
            fofc = new FindOneFactorClusters(this.covarianceMatrix,
                    this.testType, FindOneFactorClusters.Algorithm.GAP, significance);
        } else {
            throw new IllegalStateException("Data did not load correctly.");
        }

        fofc.search();
        List<List<Node>> clusters = fofc.getClusters();

        systemPrint("Clusters:");

        for (int i = 0; i < clusters.size(); i++) {
            systemPrint((i + 1) + ": " + clusters.get(i));
        }
    }

    private void writeGraph(Graph resultGraph) {
        if (graphXmlFilename != null) {
            try {
                String xml = GraphUtils.graphToXml(resultGraph);

                File file = new File(graphXmlFilename);

                PrintWriter out = new PrintWriter(file);

                out.print(xml);
                out.close();
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }

        if (graphTxtFilename != null) {
            try {
                File file = new File(graphTxtFilename);

                PrintWriter out = new PrintWriter(file);

                out.print(resultGraph);
                out.close();
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
    }

    private void writeGraphs(GraphConfiguration resultGraphs) {
        for (int i = 0; i < resultGraphs.getNumGraphs(); i++) {
            if (graphXmlFilename != null) {
                try {
                    String xml = GraphUtils.graphToXml(resultGraphs.getGraph(i));

                    File file = new File(graphXmlFilename + "_" + i + ".xml");

                    PrintWriter out = new PrintWriter(file);

                    out.print(xml);
                    out.close();
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }
            }

            if (graphTxtFilename != null) {
                try {
                    File file = new File(graphTxtFilename + "_" + i + ".txt");

                    PrintWriter out = new PrintWriter(file);

                    out.print(resultGraphs.getGraph(i));
                    out.close();
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private IndependenceTest getIndependenceTest() {
        IndependenceTest independence;

        if (useCovariance) {
            independence = new IndTestFisherZ(covarianceMatrix, significance);
        } else {
            if (this.data.isDiscrete()) {
                independence = new IndTestChiSquare(data, significance);
            } else if (this.data.isContinuous()) {
                if (useConditionalCorrelation) {

                    independence = new IndTestConditionalCorrelation(data, significance);
                    System.err.println("Using Conditional Correlation");

                } else {

                    independence = new IndTestFisherZ(data, significance);
                }


            } else {
                throw new IllegalStateException(
                        "Data must be either continuous or " + "discrete.");
            }
        }
        return independence;
    }

//    private Level convertToLevel(String level) {
//        if ("severe".equalsIgnoreCase(level)) {
//            return Level.SEVERE;
//        } else if ("warning".equalsIgnoreCase(level)) {
//            return Level.WARNING;
//        } else if ("info".equalsIgnoreCase(level)) {
//            return Level.INFO;
//        } else if ("config".equalsIgnoreCase(level)) {
//            return Level.CONFIG;
//        } else if ("fine".equalsIgnoreCase(level)) {
//            return Level.FINE;
//        } else if ("finer".equalsIgnoreCase(level)) {
//            return Level.FINER;
//        } else if ("finest".equalsIgnoreCase(level)) {
//            return Level.FINEST;
//        }
//
//        throw new IllegalArgumentException("Level must be one of 'Severe', " +
//                "'Warning', 'Info', 'Config', 'Fine', 'Finer', 'Finest'.");
//    }

    public void runImages() {
        if (this.dataSetList == null) {
            throw new IllegalStateException("Data did not load correctly.");
        }

        if (verbose) {
            systemPrint("IMaGES");
            //systemPrint(getKnowledge().toString());
            systemPrint(getVariables().toString());

            //TetradLogger.getInstance().addOutputStream(System.out);

            //TetradLogger.getInstance().setEventsToLog("info", "independencies", "knowledgeOrientations",
            //"impliedOrientations", "graph");
//            TetradLogger.getInstance().setForceLog(true);

            //TetradLogger.getInstance().log("info", "Testing it.");
        }

        DataModelList dataModelList = new DataModelList();

        for (int i = 0; i < dataSetList.size(); i++) {
            dataModelList.add(i, data);
        }

        SemBicScoreImages imagesScore = new SemBicScoreImages(dataModelList);
        imagesScore.setPenaltyDiscount(penaltyDiscount);

        if (faithfulnessAssumedArray.length > 1) {
            System.out.println("Warning: faithfulnessAssumedArray has length > 1, only first element will be used");
        }

        Fges fges = new Fges(imagesScore);
        fges.setFaithfulnessAssumed(faithfulnessAssumedArray[0]);
        Graph imagesResult = fges.search();

        outPrint("\nResult graph:");
        outPrint(imagesResult.toString());

        writeGraph(imagesResult);
    }

    public void runGest() {

        if (this.dataSetList == null) {
            throw new IllegalStateException("Data did not load correctly.");
        }

        if (verbose) {
            systemPrint("GEST");
            //systemPrint(getKnowledge().toString());
            systemPrint(getVariables().toString());

            //TetradLogger.getInstance().addOutputStream(System.out);

            //TetradLogger.getInstance().setEventsToLog("info", "independencies", "knowledgeOrientations",
            //        "impliedOrientations", "graph");
//            TetradLogger.getInstance().setForceLog(true);

            //TetradLogger.getInstance().log("info", "Testing it.");
        }

        Gest gest;

        List<Node> variables = dataSetList.get(0).getVariables();

        List<Score> scoreList = new ArrayList<>();

        for (int i = 0; i < dataSetList.size(); i++) {
            SemBicScore score = new SemBicScore(new CovarianceMatrixOnTheFly(dataSetList.get(i)));
            score.setVariables(variables);
            score.setPenaltyDiscount(penaltyDiscount);
            scoreList.add(score);
        }

        if (transferPenaltyArray.length > 1) {
            System.out.println("Warning: transferPenaltyArray has length > 1, only first element will be used");
        }
        if (weightTransferBySampleArray.length > 1) {
            System.out.println("Warning: weightTransferBySampleArray has length > 1, only first element will be used");
        }
        if (bumpMinTransferArray.length > 1) {
            System.out.println("Warning: bumpMinTransferArray has length > 1, only first element will be used");
        }

        gest = new Gest(scoreList, transferPenaltyArray[0], weightTransferBySampleArray[0], bumpMinTransferArray[0]);


        //gest.setKnowledge(getKnowledge());

        // Convert back to Graphs..
        GraphConfiguration resultGraphs = gest.search();

        // PrintUtil outputStreamPath problem and graphs.
        outPrint("\nResult graphs:");
        outPrint(resultGraphs.toString());

        writeGraphs(resultGraphs);

    }

    public void runGestTest() {
        /*int[] numNodesArray = {30, 100}; // try 10, 30, 100; and if time elapsed is not too long, try 300, 1000
        double[] numEdgesFactorArray = {1, 3, 10}; // try numNodes * {1, 1.5, 2}
        int[] kArray = {2,3,4,5}; // try 2-5; if time elapsed is not too long, try 7-10
        double[] graphDistanceFactorArray = {0.05, 0.1, 0.2}; // try 0,4,8,12, and then some percent of numEdges * 4
        int[] sampleSizeArray = {30, 60, 100}; // try numNodes * 0.5, numNodes, numNodes * 2 (with the floor being 100)
        */
        int maxDegree = 10;
        int maxIndegree = 10;
        int maxOutdegree = 10;
        //double[] transferPenaltyArray = {0, 3, 10, 30}; // try 0, 1, 3, 10
        //boolean[] weightTransferBySampleArray = {false}; // this doesn't seem to affect performance much
        //boolean[] bumpMinTransferArray = {true, false}; // try true, false
        //boolean[] faithfulnessAssumedArray = {true, false};
        // double[] penaltyDiscountArray = {4}; // try 1, 2, 4, 10
        // int idNumber = 0;
        //int numRuns = 10;

        //String filePath = "/Users/lizzie/Dissertation_code/Evaluation/";

        // calculate and save performance statistics
        Date date = new Date();
        AdjacencyPrecision ap = new AdjacencyPrecision();
        AdjacencyRecall ar = new AdjacencyRecall();
        ArrowheadPrecision arp = new ArrowheadPrecision();
        ArrowheadRecall arr = new ArrowheadRecall();
        MathewsCorrAdj mca = new MathewsCorrAdj();
        MathewsCorrArrow mcar = new MathewsCorrArrow();
        F1Adj f1a = new F1Adj();
        F1Arrow f1ar = new F1Arrow();
        SHD shd = new SHD();

        //File file = new File(outputStreamPath);

        out.println("Date \talgorithmName \tk \tgraphDistance \tnumNodes \tnumEdges " +
                "\tsampleSize \tmaxDegree \tmaxIndegree \tmaxOutdegree \ttransferPenalty " +
                "\tweightTransferBySample \tbumpMinTransfer \tpenaltyDiscount \tfaithfulnessAssumed" +
                "\tadjacencyPrecision \tadjacencyRecall \tarrowheadPrecision \tarrowheadRecall \tmathewsCorrAdj " +
                "\tmathewsCorrArrow \tf1Adj \tf1Arrow \tshd \telapsedTime " +
                "\trunID \tgraphID \tgraphOriginID \tsampleID");

        /*try {
            boolean fileExists = file.exists();
            this.out = new PrintStream(new FileOutputStream(file, true));
            if (!fileExists) {

            } else {
                System.out.println("file exists");
            }
        } catch (Exception e) {
            throw new IllegalStateException(
                    "Could not create a logfile at location " +
                            file.getAbsolutePath()
            );
        }*/

        int sampleID = 0;
        int graphID = 0;
        int graphOriginID = 0;

        for (int r = 0; r < numRuns; r++) {
            System.out.println("Run number: " + r);
            for (int numNodes : numNodesArray) {
                System.out.println("numNodes: " + numNodes);
                for (double numEdgesFactor : numEdgesFactorArray) {
                    int numEdges = (int) Math.round(numEdgesFactor * numNodes);
                    System.out.println("numEdges: " + numEdges);

                    Graph originalDag = GraphUtils.randomDag(numNodes, 0, numEdges, maxDegree, maxIndegree, maxOutdegree, false);
                    graphOriginID++;

                    ArrayList<Graph> newDagList = new ArrayList<>();
                    newDagList.add(originalDag);

                    for (int k : kArray) {
                        System.out.println("total k: " + k);

                        for (double graphDistanceFactor : graphDistanceFactorArray) {
                            int graphDistance = (int) Math.round(graphDistanceFactor * numEdges) * 4;
                            System.out.println("graphDistance: " + graphDistance);

                            for (int j = 0; j < k - 1; j++) {
                                Graph newDag = new EdgeListGraph(originalDag);
                                final List<Node> nodes2 = newDag.getNodes();
                                int SHD = SearchGraphUtils.structuralHammingDistance(newDag, originalDag);

                                while (SHD < graphDistance) {

                                    // add a random forward edge:
                                    int c1 = RandomUtil.getInstance().nextInt(nodes2.size());
                                    int c2 = RandomUtil.getInstance().nextInt(nodes2.size());

                                    if (c1 == c2) {
                                        continue;
                                    }

                                    if (c1 > c2) {
                                        int temp = c1;
                                        c1 = c2;
                                        c2 = temp;
                                    }

                                    Node n1 = nodes2.get(c1);
                                    Node n2 = nodes2.get(c2);

                                    if (newDag.isAdjacentTo(n1, n2)) {
                                        continue;
                                    }

                                    final int indegree = newDag.getIndegree(n2);
                                    final int outdegree = newDag.getOutdegree(n1);

                                    if (((indegree >= maxIndegree) ||
                                            (outdegree >= maxOutdegree)) ||
                                            ((newDag.getIndegree(n1) + newDag.getOutdegree(n1) + 1 > maxDegree) ||
                                                    (newDag.getIndegree(n2) + newDag.getOutdegree(n2) + 1 > maxDegree))) {
                                        continue;
                                    }

                                    if (!newDag.isAdjacentTo(n1, n2)) {
                                        newDag.addDirectedEdge(n1, n2);
                                    }

                                    // remove a random edge:
                                    Set<Edge> edges2 = newDag.getEdges();
                                    Edge e = randomEdge(edges2);
                                    newDag.removeEdge(e);

                                    // update SHD:
                                    SHD = SearchGraphUtils.structuralHammingDistance(newDag, originalDag);
                                }

                                graphID++;
                                newDagList.add(newDag);
                            }

                            //GraphConfiguration config = new GraphConfiguration(newDagList);

                            for (int sampleSize : sampleSizeArray) {
                                System.out.println("sampleSize: " + sampleSize);

                                for (double penaltyDiscount : penaltyDiscountArray) {
                                    List<Score> scoreList = new ArrayList<>();
                                    DataModelList dataModelList = new DataModelList();
                                    for (int i = 0; i < k; i++) {
                                        LargeScaleSimulation semSimulator = new LargeScaleSimulation(newDagList.get(i));
                                        DataSet data = semSimulator.simulateDataFisher(sampleSize);
                                        SemBicScore score = new SemBicScore(new CovarianceMatrixOnTheFly(data));
                                        score.setPenaltyDiscount(penaltyDiscount);
                                        scoreList.add(score);
                                        dataModelList.add(i, data);
                                    }
                                    sampleID++;

                                    // create score for IMaGES
                                    final SemBicScoreImages imagesScore = new SemBicScoreImages(dataModelList);
                                    imagesScore.setPenaltyDiscount(penaltyDiscount);

                                    for (boolean faithfulnessAssumed : faithfulnessAssumedArray) {
                                        System.out.println("faithfulnessAssumed: " + faithfulnessAssumed);

                                        List<Graph> fgesList = new ArrayList<>();
                                        List<Graph> trueList = new ArrayList<>();

                                        long fgesStart = System.currentTimeMillis();
                                        for (int i = 0; i < k; i++) {
                                            Fges fges = new Fges(scoreList.get(i));
                                            fges.setFaithfulnessAssumed(faithfulnessAssumed);
                                            Graph fgesOutput = fges.search();
                                            fgesList.add(fgesOutput);

                                            Graph truePattern = SearchGraphUtils.patternForDag(newDagList.get(i));
                                            truePattern = GraphUtils.replaceNodes(truePattern, fgesOutput.getNodes());
                                            trueList.add(truePattern);
                                        }

                                        long fgesStop = System.currentTimeMillis();
                                        long fgesElapsedTime = fgesStop - fgesStart;

                                        // run IMaGES
                                        long imagesStart = System.currentTimeMillis();
                                        Fges imagesSearch = new edu.cmu.tetrad.search.Fges(imagesScore);
                                        imagesSearch.setFaithfulnessAssumed(faithfulnessAssumed);
                                        Graph imagesResult = imagesSearch.search();
                                        long imagesStop = System.currentTimeMillis();
                                        long imagesElapsedTime = imagesStop - imagesStart;

                                        GraphConfiguration fgesResults = new GraphConfiguration(fgesList);
                                        GraphConfiguration trueResults = new GraphConfiguration(trueList);

                                        for (int i = 0; i < k; i++) {
                                            double adjacencyPrecisionF = ap.getValue(trueResults.getGraph(i), fgesResults.getGraph(i));
                                            double adjacencyRecallF = ar.getValue(trueResults.getGraph(i), fgesResults.getGraph(i));
                                            double arrowheadPrecisionF = arp.getValue(trueResults.getGraph(i), fgesResults.getGraph(i));
                                            double arrowheadRecallF = arr.getValue(trueResults.getGraph(i), fgesResults.getGraph(i));
                                            double mathewsCorrAdjF = mca.getValue(trueResults.getGraph(i), fgesResults.getGraph(i));
                                            double mathewsCorrArrowF = mcar.getValue(trueResults.getGraph(i), fgesResults.getGraph(i));
                                            double f1AdjF = f1a.getValue(trueResults.getGraph(i), fgesResults.getGraph(i));
                                            double f1ArrowF = f1ar.getValue(trueResults.getGraph(i), fgesResults.getGraph(i));
                                            double shd1F = shd.getValue(trueResults.getGraph(i), fgesResults.getGraph(i));

                                            // write FGES output to file
                                            out.println(date +
                                                    // algorithm and data generation parameters
                                                    "\t FGES \t" +
                                                    k + "\t" +
                                                    graphDistance + "\t" +
                                                    numNodes + "\t" +
                                                    numEdges + "\t" +
                                                    sampleSize + "\t" +
                                                    maxDegree + "\t" +
                                                    maxIndegree + "\t" +
                                                    maxOutdegree + "\t" +
                                                    "NA" + "\t" +
                                                    "NA" + "\t" +
                                                    "NA" + "\t" +
                                                    penaltyDiscount + "\t" +
                                                    faithfulnessAssumed + "\t" +

                                                    // performance
                                                    adjacencyPrecisionF + "\t" +
                                                    adjacencyRecallF + "\t" +
                                                    arrowheadPrecisionF + "\t" +
                                                    arrowheadRecallF + "\t" +
                                                    mathewsCorrAdjF + "\t" +
                                                    mathewsCorrArrowF + "\t" +
                                                    f1AdjF + "\t" +
                                                    f1ArrowF + "\t" +
                                                    shd1F + "\t" +
                                                    fgesElapsedTime + "\t" +

                                                    // indices
                                                    r  + "\t" +
                                                    graphID + "\t" +
                                                    graphOriginID + "\t" +
                                                    sampleID);

                                            imagesResult = GraphUtils.replaceNodes(imagesResult, trueResults.getGraph(i).getNodes());

                                            double adjacencyPrecisionI = ap.getValue(trueResults.getGraph(i), imagesResult);
                                            double adjacencyRecallI = ar.getValue(trueResults.getGraph(i), imagesResult);
                                            double arrowheadPrecisionI = arp.getValue(trueResults.getGraph(i), imagesResult);
                                            double arrowheadRecallI = arr.getValue(trueResults.getGraph(i), imagesResult);
                                            double mathewsCorrAdjI = mca.getValue(trueResults.getGraph(i), imagesResult);
                                            double mathewsCorrArrowI = mcar.getValue(trueResults.getGraph(i), imagesResult);
                                            double f1AdjI = f1a.getValue(trueResults.getGraph(i), imagesResult);
                                            double f1ArrowI = f1ar.getValue(trueResults.getGraph(i), imagesResult);
                                            double shd1I = shd.getValue(trueResults.getGraph(i), imagesResult);

                                            // write IMaGES output to file
                                            out.println(date +
                                                    // algorithm and data generation parameters
                                                    "\t IMaGES \t" +
                                                    k + "\t" +
                                                    graphDistance + "\t" +
                                                    numNodes + "\t" +
                                                    numEdges + "\t" +
                                                    sampleSize + "\t" +
                                                    maxDegree + "\t" +
                                                    maxIndegree + "\t" +
                                                    maxOutdegree + "\t" +
                                                    "NA" + "\t" +
                                                    "NA" + "\t" +
                                                    "NA" + "\t" +
                                                    penaltyDiscount + "\t" +
                                                    faithfulnessAssumed + "\t" +

                                                    // performance
                                                    adjacencyPrecisionI + "\t" +
                                                    adjacencyRecallI + "\t" +
                                                    arrowheadPrecisionI + "\t" +
                                                    arrowheadRecallI + "\t" +
                                                    mathewsCorrAdjI + "\t" +
                                                    mathewsCorrArrowI + "\t" +
                                                    f1AdjI + "\t" +
                                                    f1ArrowI + "\t" +
                                                    shd1I + "\t" +
                                                    imagesElapsedTime + "\t" +

                                                    // indices
                                                    r  + "\t" +
                                                    graphID + "\t" +
                                                    graphOriginID + "\t" +
                                                    sampleID
                                            );
                                        }

                                        for (double transferPenalty : transferPenaltyArray) {
                                            System.out.println("transferPenalty: " + transferPenalty);

                                            for (boolean weightTransferBySample : weightTransferBySampleArray) {
                                                System.out.println("weightTransferBySample: " + weightTransferBySample);

                                                for (boolean bumpMinTransfer : bumpMinTransferArray) {
                                                    System.out.println("bumpMinTransfer: " + bumpMinTransfer);

                                                    Gest gest = new Gest(scoreList, transferPenalty, weightTransferBySample, bumpMinTransfer);
                                                    gest.setFaithfulnessAssumed(faithfulnessAssumed);
                                                    gest.setVerbose(true);

                                                    long gestStart = System.currentTimeMillis();

                                                    GraphConfiguration gestResults = gest.search();

                                                    long gestStop = System.currentTimeMillis();

                                                    System.out.println("Elapsed " + (gestStop - gestStart) + " ms");
                                                    long gestElapsedTime = gestStop - gestStart;

                                                    for (int i = 0; i < k; i++) {
                                                        gestResults.setGraph(i, GraphUtils.replaceNodes(gestResults.getGraph(i), fgesResults.getGraph(i).getNodes()));
                                                    }

                                                    for (int i = 0; i < k; i++) {
                                                        double adjacencyPrecision = ap.getValue(trueResults.getGraph(i), gestResults.getGraph(i));
                                                        double adjacencyRecall = ar.getValue(trueResults.getGraph(i), gestResults.getGraph(i));
                                                        double arrowheadPrecision = arp.getValue(trueResults.getGraph(i), gestResults.getGraph(i));
                                                        double arrowheadRecall = arr.getValue(trueResults.getGraph(i), gestResults.getGraph(i));
                                                        double mathewsCorrAdj = mca.getValue(trueResults.getGraph(i), gestResults.getGraph(i));
                                                        double mathewsCorrArrow = mcar.getValue(trueResults.getGraph(i), gestResults.getGraph(i));
                                                        double f1Adj = f1a.getValue(trueResults.getGraph(i), gestResults.getGraph(i));
                                                        double f1Arrow = f1ar.getValue(trueResults.getGraph(i), gestResults.getGraph(i));
                                                        double shd1 = shd.getValue(trueResults.getGraph(i), gestResults.getGraph(i));

                                                        // write GEST output to file
                                                        out.println(date +

                                                                // algorithm and data generation parameters
                                                                "\t GEST \t" +
                                                                k + "\t" +
                                                                graphDistance + "\t" +
                                                                numNodes + "\t" +
                                                                numEdges + "\t" +
                                                                sampleSize + "\t" +
                                                                maxDegree + "\t" +
                                                                maxIndegree + "\t" +
                                                                maxOutdegree + "\t" +
                                                                transferPenalty + "\t" +
                                                                weightTransferBySample + "\t" +
                                                                bumpMinTransfer + "\t" +
                                                                penaltyDiscount + "\t" +
                                                                faithfulnessAssumed + "\t" +

                                                                // performance
                                                                adjacencyPrecision + "\t" +
                                                                adjacencyRecall + "\t" +
                                                                arrowheadPrecision + "\t" +
                                                                arrowheadRecall + "\t" +
                                                                mathewsCorrAdj + "\t" +
                                                                mathewsCorrArrow + "\t" +
                                                                f1Adj + "\t" +
                                                                f1Arrow + "\t" +
                                                                shd1 + "\t" +
                                                                gestElapsedTime + "\t" +

                                                                // indices
                                                                r  + "\t" +
                                                                graphID + "\t" +
                                                                graphOriginID + "\t" +
                                                                sampleID
                                                        );

                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    public static int[][] sumMultiMisclassifications(GraphConfiguration estGraphs, GraphConfiguration trueGraphs) {
        int[][] counts = GraphUtils.edgeMisclassificationCounts(estGraphs.getGraph(0), trueGraphs.getGraph(0), false);
        int row = counts.length;
        int col = counts[0].length;
        for (int i = 1; i < trueGraphs.getNumGraphs(); i++) {
            int[][] addCounts = GraphUtils.edgeMisclassificationCounts(estGraphs.getGraph(i), trueGraphs.getGraph(i), false);
            for (int r = 0; r < row; r++) {
                for (int c = 0; c < col; c++) {
                    counts[r][c] += addCounts[r][c];
                }
            }
        }
        return counts;
    }

    public static int[] stringToIntArray(String s) {
        // string must be comma-separated with no whitespace, e.g. "1,2,3,4,5"
        String[] array = s.split(",");
        int[] ints = new int[array.length];
        for(int i=0; i<array.length; i++) {
            try {
                ints[i] = Integer.parseInt(array[i]);
            } catch (NumberFormatException nfe) {
                //Not an integer
            }
        }
        return ints;
    }

    public static double[] stringToDoubleArray(String s) {
        // string must be comma-separated with no whitespace, e.g. "1,2,3,4,5"
        String[] array = s.split(",");
        double[] doubles = new double[array.length];
        for(int i=0; i<array.length; i++) {
            try {
                doubles[i] =  Double.parseDouble(array[i]);
            } catch (NumberFormatException nfe) {
                //Not an integer
            }
        }
        return doubles;
    }

    public static boolean[] stringToBooleanArray(String s) {
        String[] array = s.split(",");
        boolean[] bools = new boolean[array.length];
        for(int i=0; i<array.length; i++) {
            try {
                bools[i] = Boolean.parseBoolean(array[i]);
            } catch (Exception nfe) {
                //Not an integer
            }
        }
        return bools;
    }

    public static String countsToMisclassifications(int[][] counts) {
        StringBuilder builder = new StringBuilder();

        TextTable table2 = new TextTable(9, 7);

        table2.setToken(1, 0, "---");
        table2.setToken(2, 0, "o-o");
        table2.setToken(3, 0, "o->");
        table2.setToken(4, 0, "<-o");
        table2.setToken(5, 0, "-->");
        table2.setToken(6, 0, "<--");
        table2.setToken(7, 0, "<->");
        table2.setToken(8, 0, "null");
        table2.setToken(0, 1, "---");
        table2.setToken(0, 2, "o-o");
        table2.setToken(0, 3, "o->");
        table2.setToken(0, 4, "-->");
        table2.setToken(0, 5, "<->");
        table2.setToken(0, 6, "null");

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 6; j++) {
                if (i == 7 && j == 5) table2.setToken(i + 1, j + 1, "*");
                else table2.setToken(i + 1, j + 1, "" + counts[i][j]);
            }
        }

        builder.append("\n").append(table2.toString());
        return builder.toString();
    }

    public static Edge randomEdge(Set<Edge> edgeSet) {
        int num = RandomUtil.getInstance().nextInt(edgeSet.size());
        for (Edge e : edgeSet) {
            if (--num < 0) {
                return e;
            }
        }
        throw new AssertionError();
    }




    public static void main(final String[] argv) {
        new TetradCmd(argv);
    }

    private int getDepth() {
        return depth;
    }

    private IKnowledge getKnowledge() {
        return knowledge;
    }

    private List<Node> getVariables() {
        if (data != null) {
            return data.getVariables();
        } else if (covarianceMatrix != null) {
            return covarianceMatrix.getVariables();
        } else if (dataSetList != null) {
            return dataSetList.get(0).getVariables();
        }

        throw new IllegalArgumentException("Data nor covariance specified.");
    }

    /**
     * Allows an array of strings to be treated as a tokenizer.
     */
    private static class StringArrayTokenizer {
        String[] tokens;
        int i = -1;

        public StringArrayTokenizer(String[] tokens) {
            this.tokens = tokens;
        }

        public boolean hasToken() {
            return i < tokens.length - 1;
        }

        public String nextToken() {
            return tokens[++i];
        }
    }
}





