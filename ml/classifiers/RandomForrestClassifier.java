package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import javax.print.attribute.HashAttributeSet;

import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;

public class RandomForrestClassifier implements Classifier{

    //number of trees we have in forrest
    private int numTrees;
    //nubmer of nodes per tree in forrest
    private int numNodes;
    //the entire random forrest model
    public ArrayList<DecisionTreeClassifier> forrest;
    //the dataset we are gonna use to train the model on 
    private ArrayList<DataSet> dataSet;
    
    public RandomForrestClassifier(DataSet data, int numTrees, boolean bagging){
        this.numNodes = 3;
        this.numTrees = numTrees;
        this.forrest = new ArrayList<>();
        this.dataSet = new ArrayList<>();

        int size = data.getData().size();

        if(bagging){
            for(int i = 0; i < numTrees; i++){

            int range = size;

            DataSet randData = new DataSet(data.getFeatureMap());

            for(int j = 0; j < size; j++){

                int randNum = (int)(Math.random() * range);
                
                randData.addData(data.getData().get(randNum));
            }
            

            this.dataSet.add(randData);
            }
        }

        else{
            
        }
        

        
            
    }

    public void setNumNodes(int num){
        this.numNodes = num;
    }

    public void setNumTrees(int num){
        this.numTrees = num;
    }

    public ArrayList<DecisionTreeClassifier> getForrest(){
        return this.forrest;
    }

    public void buildForrest(){

        for(DataSet data : this.dataSet){
            this.train(data);
        }

    }


    //builds one tree for the forrest
    public void train(DataSet data) {
        DecisionTreeClassifier tree = new DecisionTreeClassifier();
        tree.setDepthLimit(this.numNodes);
        tree.train(data);
        forrest.add(tree);
    }

    @Override
    public double classify(Example example) {

        HashMap<Double, Integer> votes = new HashMap<>();
        
        
        for(DecisionTreeClassifier tree : forrest){
            Double label = tree.classify(example);

            votes.compute(label, (k, v) -> v == null ? 1 : v + 1);

        }
        System.out.println(votes);
        
        return Collections.max(votes.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    @Override
    public double confidence(Example example) {
        HashMap<Double, Integer> votes = new HashMap<>();

        for(DecisionTreeClassifier tree : forrest){
            Double label = tree.classify(example);

            votes.compute(label, (k, v) -> v == null ? 1 : v + 1);

        }
        
        return votes.get(classify(example)) / numTrees;
    }
    
}
