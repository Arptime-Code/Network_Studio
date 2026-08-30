class NeuralNetwork {
    constructor(valueLayerArray) {
        this.valueLayers = this.initializeValueLayers(valueLayerArray);
        this.biasLayers = this.initializeValueLayers(valueLayerArray);
        this.weightLayers = this.initializeWeightLayers(valueLayerArray);
    }

    initializeValueLayers(valueLayerNumbersArray) {
        let valueLayerArray = [];

        for(let i = 0; i < valueLayerNumbersArray.length; i++) {
            valueLayerArray.push(new Array(valueLayerNumbersArray[i]).fill(0));
        }
        //console.log(valueLayerArray);
        return valueLayerArray;
    }

    initializeWeightLayers(weightLayerNumbersArray) {
        let weightLayerArray = [];

        for(let i = 0; i < (weightLayerNumbersArray.length - 1); i++) {
            let weightLayerSubArray = [];

            for(let j = 0; j < weightLayerNumbersArray[i + 1]; j++) {
                let weights = [];

                for(let k = 0; k < weightLayerNumbersArray[i]; k++) {
                    // Random values between -0.5 and 0.5
                    weights.push((Math.random() - 0.5));
                }
                weightLayerSubArray.push(weights);
            }
            weightLayerArray.push(weightLayerSubArray);
        }
        // console.log(weightLayerArray);
        return weightLayerArray;
    }

    gelu(input) {
        const c = Math.sqrt(8 / Math.PI);

        return (input / (1 + Math.exp(-c * input)));
    }

    derivedGelu(input) {
        const c = Math.sqrt(8 / Math.PI);
        const s = (1 / (1 + Math.exp(-c * input)));

        return (s + c * input * s * (1 - s));
    }

    async feedForward(tmp = { "layer" : 0, "input" : 0, "output" : 0}) {
        //Reset current working node for output to zero
        if(tmp["output"] == 0) this.valueLayers[tmp["layer"] + 1][tmp["input"]] = 0;


        //Sum up all weights with their nodes
        this.valueLayers[tmp["layer"] + 1][tmp["input"]] += (this.valueLayers[tmp["layer"]][tmp["output"]] * this.weightLayers[tmp["layer"]][tmp["input"]][tmp["output"]]);



        //Loop through all outputs
        if(tmp["output"] < this.valueLayers[tmp["layer"]].length - 1) return await this.recursive(this.feedForward, (
            tmp["output"] += 1, 
            tmp));

        //Add the bias and activation function
        this.valueLayers[tmp["layer"] + 1][tmp["input"]] += this.biasLayers[tmp["layer"] + 1][tmp["input"]];
        this.valueLayers[tmp["layer"] + 1][tmp["input"]] = this.gelu(this.valueLayers[tmp["layer"] + 1][tmp["input"]]);


        //Loop through all inputs and get them
        if(tmp["input"] < this.valueLayers[tmp["layer"] + 1].length - 1) return await this.recursive(this.feedForward, (
            tmp["input"] += 1, 
            tmp["output"] = 0, 
            tmp));


        //Loop through all the layers
        if(tmp["layer"] < this.weightLayers.length - 1) return await this.recursive(this.feedForward, (
            tmp["layer"] += 1, 
            tmp["input"] = 0, 
            tmp["output"] = 0, 
            tmp));
    }

    outputError(y_hat, y) {
        return y_hat - y;
    }

    computeZ(layer, neuronIndex) {
        let z = this.biasLayers[layer][neuronIndex];
        for (let k = 0; k < this.valueLayers[layer - 1].length; k++) {
            z += (this.valueLayers[layer - 1][k] * this.weightLayers[layer - 1][neuronIndex][k]);
        }
        return z;
    }

    backPropagationSum(innerFunction, layer, input) {
        let sum = 0;
        let currentLayerNeuronCount = this.valueLayers[layer].length;

        for (let k = 0; k < currentLayerNeuronCount; k++) {
            let output = k;
            let z = this.computeZ(layer, output);
            
            let R_prime = this.derivedGelu(z);
            // CHANGED: Output layer derivative is 1 (linear activation)
            if (layer === this.valueLayers.length - 1) {
                R_prime = 1;
            }

            let innerVal = innerFunction.call(this, output);
            // CHANGED: Read from oldWeights snapshot so updates don't distort recursive calls
            let weight = this.oldWeights[layer - 1][output][input];

            sum += innerVal * R_prime * weight;
        }

        return sum;
    }

    backPropagation(output, learningRate) {
            let maxLayers = this.weightLayers.length;
            let outputLayerIndex = this.valueLayers.length - 1;

            // CHANGED: 1-line native clone snapshot so backPropagationSum reads original weights
            this.oldWeights = structuredClone(this.weightLayers);

            let currentInnerFunc = function(neuronIndex) {
                let y_hat = this.valueLayers[outputLayerIndex][neuronIndex];
                let y = output[neuronIndex];
                return this.outputError(y_hat, y);
            };

            for (let i = maxLayers - 1; i >= 0; i--) {
                let targetValueLayer = i + 1;

                for (let j = 0; j < this.valueLayers[targetValueLayer].length; j++) {
                    let z = this.computeZ(targetValueLayer, j);
                    
                    let R_prime = this.derivedGelu(z);
                    // CHANGED: Output layer derivative is 1 (linear activation)
                    if (targetValueLayer === outputLayerIndex) {
                        R_prime = 1;
                    }

                    let innerVal = currentInnerFunc.call(this, j);

                    let delta = innerVal * R_prime;

                    //Update with learning rate (first biases)
                    this.biasLayers[targetValueLayer][j] -= learningRate * delta;

                    //Update weights
                    for (let k = 0; k < this.valueLayers[i].length; k++) {
                        let x_i = this.valueLayers[i][k];
                        this.weightLayers[i][j][k] -= learningRate * delta * x_i;
                    }
                }

                //Nest the new function to go one level deeper (one layer one towards the start of the network)
                let prevInnerFunc = currentInnerFunc;
                let layerToSum = targetValueLayer;

                currentInnerFunc = function(inputIndex) {
                    return this.backPropagationSum(prevInnerFunc, layerToSum, inputIndex);
                };
            }
    }

    async trainNetwork(trainingData, learningRate, epochs, lossLimit, network, tmp = { "epoch" : 0, "datapoint" : 0, "totalEpochLoss" : 0}) {
        network.valueLayers[0] = trainingData[tmp["datapoint"]][0];
        await network.feedForward();

        tmp["totalEpochLoss"] += this.computeLoss(trainingData[tmp["datapoint"]][1]);

        network.backPropagation(trainingData[tmp["datapoint"]][1], learningRate);


        //Run this for all datapoints
        if(tmp["datapoint"] < trainingData.length - 1) return await this.recursive(this.trainNetwork, trainingData, learningRate, epochs, lossLimit, network, (
            tmp["datapoint"] += 1, 
            tmp));


        //Every x(10000) epochs output the loss
        if(tmp["epoch"] % 10000 == 0) console.log(tmp["totalEpochLoss"]);

        //Stop at loss limit
        if(tmp["totalEpochLoss"] < lossLimit) return console.log("Epochs: " + tmp["epoch"]);

        //Run this for the number of epochs i want
        if(tmp["epoch"] < epochs) return await this.recursive(this.trainNetwork, trainingData, learningRate, epochs, lossLimit, network, (
            tmp["epoch"] += 1, 
            tmp["datapoint"] = 0, 
            tmp["totalEpochLoss"] = 0, 
            tmp));
    }


    computeLoss(output) {
        let outputLayerIndex = this.valueLayers.length - 1;
        let loss = 0;
        for (let j = 0; j < this.valueLayers[outputLayerIndex].length; j++) {
            let error = this.valueLayers[outputLayerIndex][j] - output[j];
            loss += 0.5 * (error * error);
        }
        return loss;
    }

    async recursive(fn, ...args) {
        await null;
        return fn.apply(this, args);
    }
}






























function generateBalancedTrainingData(count) {
    let dataset = [];

    for (let i = 0; i < count; i++) {
        let inputs = [];
        let points = [];

        // Force 50% of data to have two points very close together (< 0.15)
        let forceClose = (i % 2 === 0);

        for (let p = 0; p < 4; p++) {
            let x, y;
            if (forceClose && p === 1) {
                // Place point 1 right next to point 0
                x = points[0].x + (Math.random() * 0.2 - 0.1);
                y = points[0].y + (Math.random() * 0.2 - 0.1);
            } else {
                x = Math.random() * 2 - 1;
                y = Math.random() * 2 - 1;
            }
            inputs.push(x);
            inputs.push(y);
            points.push({ x: x, y: y });
        }

        let minDistance = 999999;
        for (let a = 0; a < points.length; a++) {
            for (let b = a + 1; b < points.length; b++) {
                let dx = points[a].x - points[b].x;
                let dy = points[a].y - points[b].y;
                let dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < minDistance) minDistance = dist;
            }
        }

        dataset.push([inputs, [minDistance]]);
    }

    return dataset;
}










// Generate 500 data points
let trainingData = generateBalancedTrainingData(150);





// const trainingData = [[[0,0],[0,0]]];





const network = new NeuralNetwork([8, 5, 2, 1]);


await network.trainNetwork(trainingData, 0.01, 10000, 0.005, network);



// Test on one instance
network.valueLayers[0] = trainingData[0][0];
await network.feedForward();

console.log("Predicted Min Pair Distance:", network.valueLayers[network.valueLayers.length - 1][0]);
console.log("Actual Min Pair Distance:", trainingData[0][1][0]);


// 3 Hardcoded Unseen Test Points [x4, y4] and their exact actual min distances
const testCases = [
    { input: [0.2, 0.3, -0.6, 0.5, 0.5, -0.4,  0.15, -0.25], actual: 0.381 },
    { input: [0.2, 0.3, -0.6, 0.5, 0.5, -0.4, -0.55,  0.45], actual: 0.071 },
    { input: [0.2, 0.3, -0.6, 0.5, 0.5, -0.4,  0.25,  0.35], actual: 0.071 }
];

for (const [i, test] of testCases.entries()) {
    network.valueLayers[0] = test.input;
    await network.feedForward();
    let predicted = network.valueLayers[network.valueLayers.length - 1][0];
    console.log(`Test ${i + 1} | Predicted: ${predicted.toFixed(3)} | Actual: ${test.actual}`);
}