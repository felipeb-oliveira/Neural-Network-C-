#include <cstdlib>
#include <iostream>
#include <vector>
#include "neuralnet.cc"

vector< vector<float> > createDataSet(char datapath[]){
    int numInstances = 0;
    int numVariables = 0;
    char dataIteration = 'a';

    FILE* data = fopen(datapath, "r");

    if(data != NULL){

        printf("\nReading data content... ");

        while(dataIteration != 'E'){

            fscanf(data, "%c", &dataIteration);

            if(dataIteration != 'E'){
                numVariables = 0;

                numInstances++;
            }

            while(dataIteration != '\n' && dataIteration != 'E'){
                if(dataIteration == ' '){
                    numVariables++;
                }
                fscanf(data, "%c", &dataIteration);

            }


        }

        printf("Done!\n");
    } else printf("\nCannot read data!\n");


    rewind(data);

    printf("Storing data content... ");


    vector< vector<float> > dataset;
    dataset =  vector< vector<float> >(numInstances, vector<float>(numVariables));

    for(int i = 0; i < numInstances; i++){
        for(int j = 0; j < numVariables; j++){
            fscanf(data, "%f", &dataset[i][j]);
            fscanf(data, "%c", &dataIteration);
        }
    }

    fclose(data);

    printf("Done!\n");
    printf("Dataset size: %d instances with %d attributes each.\n", numInstances, numVariables);

    return dataset;




}

using namespace std;

int main(){

    vector< vector<float> > dataset = createDataSet("semeion.data");


    int numOutputs = 10;
    int numInputs = dataset[0].size() - numOutputs;
    printf("Num inputs: %d", numInputs);

    vector<float> testInput(numInputs);
    vector<float> testOutput(numOutputs);

    for(int i = 0; i < numInputs; i++){
        testInput[i] = dataset[dataset.size()-1][i];
    }

    int c = 0;

    for(int i = numInputs; i < dataset[0].size(); i++){
        printf("\nDesired Output %d: %f", c, dataset[dataset.size()-1][i]);
        c++;
    }

    NeuralNetwork* nn = new NeuralNetwork(numInputs, 1, numInputs * 2, 10);

    nn->train(dataset);

    nn->feedForward(testInput);
    testOutput = nn->getOuput();

    for(int i = 0; i < testOutput.size(); i++){
        printf("\nOutput %d: %f", i, testOutput[i]);
    }


    return 0;
}
