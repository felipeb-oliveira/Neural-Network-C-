#include <iostream>
#include <math.h>
#include <vector>
#include <map>
#include <ctime>
#include <cstdlib>

#define LEARNING_RATE 1
#define MAX_ERROR 0.00001

using namespace std;

class Neuron {

private:
    float output;

public:
    float feed(float input){
        this->output = 1/(1 + exp(-input));
        return this->output;
    }

    float getOutput(){
        return this->output;
    }
};



class Layer {

private:
    bool isLastLayer;
    int numInputs;
    int numNeurons;
    vector <float> outputs;
    vector <float> inputs;
    vector <float> rawOutputs;
    vector <float> error;
    vector <float> derOutput;
    vector <float> errorTimesDerOutput;
    vector <Neuron*> neurons;
    vector<vector<float>> weights;
    vector<vector<float>> deltaWeights;

public:
    Layer(int numNeurons, int numInputs, bool isLastLayer = false){
        this->isLastLayer = isLastLayer;

        for(int i = 0; i < numNeurons; i++){
            this->neurons.push_back(new Neuron());
        }


        this->numNeurons = numNeurons;
        this->numInputs = numInputs;


        error = vector <float>(numNeurons);
        derOutput = vector <float>(numNeurons);
        errorTimesDerOutput = vector <float>(numNeurons);
        rawOutputs = vector<float>(numNeurons);
        outputs = vector<float>(numNeurons);
        printf("\nnumNeurons: %d, numInputs: %d", numNeurons, numInputs);
        weights = vector< vector<float> > (numNeurons, vector<float>(numInputs));
        deltaWeights = vector< vector<float> > (numNeurons, vector<float>(numInputs));

        generateRandomWeights();

    }

    vector<float> getError(){
        return error;
    }

    vector<float> getDerOutput(){
        return derOutput;
    }

    vector<float> getErrorTimesDerOutput(){
        return errorTimesDerOutput;
    }

    vector<float> getOutputs(){
        return outputs;
    }

    vector<vector<float>> getWeights(){
        return weights;
    }


    vector<float> feedForward(vector<float> inputs){
        this->inputs = inputs;
        for(int i = 0; i < this->numNeurons; i++){
            rawOutputs[i] = 0;
            for(int j = 0; j < numInputs; j++){
                float a = inputs[j];
                //printf("%f\n", inputs[j]);
                //printf("%f\n", weights[i][j]);
                printf("%d %d\n", i, j);
                rawOutputs[i] += weights[i][j] * inputs[j];
            }

            outputs[i] = neurons[i]->feed(rawOutputs[i]);
        }

        return outputs;
    }

    void generateRandomWeights(){

        for(int i = 0; i < this->numNeurons; i++){
            for(int j = 0; j < this->numInputs; j++){
                weights[i][j] = (rand()%100)/50.0f - 1.0f;
                //printf("\nPeso %d -> %d: %f",j, i, weights[i][j]);
            }
        }
    }

    void generateUnitaryWeights(){

        for(int i = 0; i < this->numNeurons; i++){
            for(int j = 0; j < this->numInputs; j++){
                weights[i][j] = 1.0f;
                //printf("\nPeso %d -> %d: %f",j, i, weights[i][j]);
            }
        }
    }

    void backPropagationLastLayer(vector<float> expectedResults){
        for(int i = 0; i < getOutputs().size(); i++){
            error[i] = getOutputs()[i] - expectedResults[i];

            derOutput[i] = 1/(1 + exp(-rawOutputs[i])) * (1 - (1/(1 + exp(-rawOutputs[i]))));
            errorTimesDerOutput[i] = error[i] * derOutput[i];

            for(int j = 0; j < numInputs; j++){
                deltaWeights[i][j] = error[i] * derOutput[i] * inputs[j];
            }
        }
    }

    void backPropagationHiddenLayer(vector< vector<float> > forwardWeights, vector<float> forwardErrorTimesDerOutput){
        for(int i = 0; i < getOutputs().size(); i++){
            errorTimesDerOutput[i] = 0;

            for(int j = 0; j < forwardErrorTimesDerOutput.size(); j++){
                errorTimesDerOutput[i] += forwardWeights[j][i] * forwardErrorTimesDerOutput[j];
            }

            errorTimesDerOutput[i] *= 1/(1 + exp(-rawOutputs[i])) * (1 - (1/(1 + exp(-rawOutputs[i]))));

            for(int j = 0; j < numInputs; j++){
                deltaWeights[i][j] = errorTimesDerOutput[i] * inputs[j];
            }
        }
    }

    void updateWeights(){
        for(int i = 0; i < outputs.size(); i++){
            for(int j = 0; j < inputs.size(); j++){
                weights[i][j] -= deltaWeights[i][j] * LEARNING_RATE;
            }
        }
    }

};





class NeuralNetwork{

private:
    int numInputs;
    int numHiddenLayers;
    int numOutputs;
    int neuronsPerHiddenLayer;

    vector<Layer*> layers;

public:
    NeuralNetwork(int numInputs, int numHiddenLayers, int neuronsPerHiddenLayer, int numOutputs){
        srand(time(NULL));

        this->numInputs = numInputs;
        this->numHiddenLayers = numHiddenLayers;
        this->numOutputs = numOutputs;
        this->neuronsPerHiddenLayer = neuronsPerHiddenLayer;

        layers.push_back(new Layer(neuronsPerHiddenLayer, numInputs, true));

        for(int i = 1; i < numHiddenLayers; i++){
            layers.push_back(new Layer(numHiddenLayers, neuronsPerHiddenLayer));
        }

        layers.push_back(new Layer(numOutputs, neuronsPerHiddenLayer));



    }

    vector<float> feedForward(vector<float> inputs){
        layers[0]->feedForward(inputs);

        for(int i = 1; i <= this->numHiddenLayers; i++){
            layers[i]->feedForward(layers[i-1]->getOutputs());
        }

        return layers.back()->getOutputs();
    }

    void backPropagation(vector<float> expectedResults){

        layers.back()->backPropagationLastLayer(expectedResults);

        for(int i = layers.size() - 2; i >= 0; i--){
            layers[i]->backPropagationHiddenLayer(layers[i+1]->getWeights(), layers[i+1]->getErrorTimesDerOutput());
        }

        updateLayersWeights();
    }

    void updateLayersWeights(){
        for(int i = 0; i < layers.size(); i++){
            layers[i]->updateWeights();
        }
    }

    float getTotalError(){
        float totalError = 0;

        for(int i = 0; i < layers.back()->getOutputs().size(); i++){
            totalError += pow(layers.back()->getError()[i], 2);
        }

        return totalError/2;
    }

    vector<float> getOuput(){
        return layers.back()->getOutputs();
    }

    void print(){
        printf("\n\n");

        for(int i = 0; i < layers.size(); i++){
            for(int j = 0; j < layers[i]->getOutputs().size(); j++){
                printf("%f ", layers[i]->getOutputs()[j]);
            }
            printf("\n");
        }
    }

    void train(vector< vector<float> > dataset){
        vector<float> inputs(numInputs);
        vector<float> desiredOutputs(numOutputs);

        float dataError = 1;
        int numInstances = dataset.size();


        printf("\nStarting Training...\n");

        while(dataError > MAX_ERROR){
            dataError = 0;

            for(int i = 0; i < numInstances; i++){
                printf("load inputs...\n");
                for(int j = 0; j < numInputs; j++){
                    inputs[j] = dataset[i][j];
                }
                printf("load desired outputs...\n");
                for(int j = numInputs; j < dataset[0].size(); j++){
                    desiredOutputs[j] = dataset[i][j];
                }
                printf("feed forward...\n");
                feedForward(inputs);
                printf("backpropagation...\n");
                backPropagation(desiredOutputs);
                printf("get error...\n");

                dataError += getTotalError();
            }

            printf("\nData Error: %f", dataError);

        }

        printf("\nTraining finished!");

    }







};
