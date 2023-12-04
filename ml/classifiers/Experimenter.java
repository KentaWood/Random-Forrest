package ml.classifiers;

import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;

public class Experimenter {
    public static void main(String[] args) {

        DataSet wineDataset = new DataSet("/Users/nobuko/Desktop/CS-158 Projects/assignment-5-alex-frank-1/data/wines.train", DataSet.TEXTFILE);

        // // 10 cross validated dataset with randomization
        // CrossValidationSet data = new CrossValidationSet(wineDataset, 10, true);

        // for(int j = 0 ; j < 16; j++){

        //     System.out.println("Depth Limit: " + j);

        //     for(int i = 0; i < 10; i++){
        //         DataSetSplit dataSetSplit = data.getValidationSet(i);
        //         DataSet train = dataSetSplit.getTrain();
        //         DataSet test = dataSetSplit.getTest();

        //         DecisionTreeClassifier tree = new DecisionTreeClassifier();

        //         tree.setDepthLimit(j);
        //         tree.train(train);

        //         double acc = 0.0;

                
        //         for (Example ex : test.getData()) {
        //             acc += (tree.classify(ex) == ex.getLabel()) ? 1 : 0;
        //         }

        //         System.out.println( acc / test.getData().size());

        //     }
        // }

        DataSetSplit dataSplit = wineDataset.split(.9);

        DataSet test = dataSplit.getTest();
        DataSet train = dataSplit.getTrain();

        RandomForrestClassifier model = new RandomForrestClassifier(train, 10, true);

        model.setNumNodes(3);

        model.buildForrest();



        double acc = 0.0;

        for (Example ex : test.getData()) {
                    acc += (model.classify(ex) == ex.getLabel()) ? 1 : 0;
        //             // System.out.println(model.classify(ex));
        //             // System.out.println(ex.getLabel());
            }

        System.out.println( acc / test.getData().size());

        // System.out.println(model.getForrest());

        // System.out.println(model.confidence(test.getData().get(0)));

        // System.out.println(test.getData().get(0).getLabel());



        // for(int i = 0; i < 10; i++){
        //     Example ex = test.getData().get(i);

        //     System.out.println(model.confidence(ex));
        //     System.out.println(ex.getLabel());
        // }
    }
    
    
}
