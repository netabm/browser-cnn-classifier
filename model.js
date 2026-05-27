// === DOM Elements Setup ===
const canvas = document.getElementById('paintCanvas');
const ctx = canvas.getContext('2d');
const resetBtn = document.getElementById('resetBtn');

// === Canvas State Variables ===
let isDrawing = false;

// === Initialize Canvas Properties ===
function initCanvas() {
    // Fill background with white (crucial for CNN pixel extraction)
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Set drawing brush properties
    ctx.lineWidth = 15; // Thick lines make it easier for the CNN to extract features
    ctx.lineCap = "round";
    ctx.strokeStyle = "black";
}

// Call initialization immediately on page load
initCanvas();

// === Mouse Event Listeners for Drawing ===
canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener('mousemove', (e) => {
    if (isDrawing) {
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    }
});

canvas.addEventListener('mouseup', () => {
    isDrawing = false;
});

canvas.addEventListener('mouseleave', () => {
    isDrawing = false; // Stops drawing if the mouse leaves the canvas area
});

// === Button Event Listeners ===
resetBtn.addEventListener('click', () => {
    initCanvas();
    const statusDiv = document.getElementById('trainingStatus');
    if (statusDiv) {
        statusDiv.innerText = "Canvas reset. Waiting for input...";
    }
});

// === Data Extraction and Preprocessing ===

// Set the target size for the CNN input (28x28 is standard for simple shape detection)
const MODEL_INPUT_SIZE = 28;

function getPixelData() {
    // 1. Create a temporary hidden canvas to resize the drawing
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = MODEL_INPUT_SIZE;
    tempCanvas.height = MODEL_INPUT_SIZE;
    const tempCtx = tempCanvas.getContext('2d');

    // 2. Draw the large original canvas onto the small temporary one (shrinks the image)
    tempCtx.drawImage(canvas, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

    // 3. Extract the raw pixel data (RGBA format)
    const imageData = tempCtx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    const data = imageData.data;

    // 4. Convert to Grayscale and Normalize (0 to 1)
    let pixelArray = [];
    
    // The data array contains 4 values per pixel: Red, Green, Blue, Alpha
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i+1];
        const b = data[i+2];
        
        // Calculate the average grayscale value (0 is black, 255 is white)
        let grayscaleValue = (r + g + b) / 3;
        
        // Normalize: Since background is white and drawing is black, 
        // we invert it so drawn lines are 1.0 and background is 0.0
        let normalizedValue = 1.0 - (grayscaleValue / 255.0);
        
        pixelArray.push(normalizedValue);
    }

    return pixelArray; // Returns a 1D array of 784 numbers (28 * 28)
}

// === Math and Matrix Helpers ===

// Helper function to convert 1D array to a 2D matrix (e.g., 784 -> 28x28)
function toMatrix(array, size) {
    let matrix = [];
    for (let i = 0; i < size; i++) {
        matrix.push(array.slice(i * size, (i + 1) * size));
    }
    return matrix;
}

// === Loss Function & Gradients ===

/**
 * Computes Categorical Cross-Entropy Loss and the initial gradient.
 * @param {Array<number>} predictions - Output from Softmax (probabilities)
 * @param {number} targetIndex - The correct class index (0, 1, or 2)
 * @returns {Object} { loss, gradient }
 */
function computeLossAndGradient(predictions, targetIndex) {
    // Cross-Entropy Loss: -log(probability of the correct class)
    // Adding a tiny number (1e-9) to prevent log(0) which results in Infinity
    const loss = -Math.log(predictions[targetIndex] + 1e-9);
    
    // Gradient of Softmax + Cross-Entropy is remarkably simple: p_i - y_i
    let gradient = predictions.slice(); // Copy predictions array
    gradient[targetIndex] -= 1.0;       // Subtract 1 from the true class
    
    return { loss, gradient };
}

// === Neural Network Layers ===

class Conv2D {
    /**
     * @param {number} numFilters - Number of feature maps to output
     * @param {number} filterSize - Size of the NxN filter (e.g., 3 for 3x3)
     */
    constructor(numFilters, filterSize) {
        this.numFilters = numFilters;
        this.filterSize = filterSize;
        this.filters = [];
        
        // Initialize filters with small random values
        for (let i = 0; i < numFilters; i++) {
            let filter = [];
            for (let r = 0; r < filterSize; r++) {
                let row = [];
                for (let c = 0; c < filterSize; c++) {
                    // Random values between -0.5 and 0.5
                    row.push(Math.random() - 0.5);
                }
                filter.push(row);
            }
            this.filters.push(filter);
        }
        
        // Initialize biases to 0 for each filter
        this.biases = new Array(numFilters).fill(0);
    }

    /**
     * Performs the forward pass of the convolution layer.
     * @param {Array<Array<number>>} input - 2D input matrix (e.g., 28x28)
     * @returns {Array<Array<Array<number>>>} - 3D output tensor (numFilters x outSize x outSize)
     */
    forward(input) {
        this.lastInput = input; // Save for backpropagation later
        
        const inputSize = input.length;
        // Calculate output dimension assuming stride = 1 and padding = 0 (Valid Convolution)
        const outputSize = inputSize - this.filterSize + 1; 
        
        let output = []; // This will hold all feature maps
        
        // Apply each filter
        for (let f = 0; f < this.numFilters; f++) {
            let featureMap = [];
            
            // Slide the filter over the input
            for (let r = 0; r < outputSize; r++) {
                let row = [];
                for (let c = 0; c < outputSize; c++) {
                    let sum = 0;
                    
                    // Element-wise multiplication between filter and current input region
                    for (let fr = 0; fr < this.filterSize; fr++) {
                        for (let fc = 0; fc < this.filterSize; fc++) {
                            sum += input[r + fr][c + fc] * this.filters[f][fr][fc];
                        }
                    }
                    
                    // Add the bias for this filter
                    sum += this.biases[f];
                    row.push(sum);
                }
                featureMap.push(row);
            }
            output.push(featureMap);
        }
        
        return output;
    }

    /**
     * Performs the backward pass for the convolution layer and updates filters.
     * @param {Array<Array<Array<number>>>} dOutput - 3D gradient from the next layer
     * @param {number} learningRate - The learning rate for weight updates
     * @returns {Array<Array<number>>} - 2D gradient for the input image
     */
    backward(dOutput, learningRate) {
        const inputSize = this.lastInput.length;
        const outputSize = dOutput[0].length;
        
        // Initialize dInput (gradient for previous layer) with zeros
        let dInput = [];
        for (let r = 0; r < inputSize; r++) {
            dInput.push(new Array(inputSize).fill(0));
        }

        // Initialize gradients for filters
        let dFilters = [];
        for (let f = 0; f < this.numFilters; f++) {
            let filterGrad = [];
            for (let r = 0; r < this.filterSize; r++) {
                filterGrad.push(new Array(this.filterSize).fill(0));
            }
            dFilters.push(filterGrad);
        }
        
        let dBiases = new Array(this.numFilters).fill(0);

        // Calculate gradients by sliding the window again
        for (let f = 0; f < this.numFilters; f++) {
            for (let r = 0; r < outputSize; r++) {
                for (let c = 0; c < outputSize; c++) {
                    const gradient = dOutput[f][r][c];
                    dBiases[f] += gradient; // Accumulate bias gradient

                    // Accumulate filter and input gradients
                    for (let fr = 0; fr < this.filterSize; fr++) {
                        for (let fc = 0; fc < this.filterSize; fc++) {
                            dFilters[f][fr][fc] += this.lastInput[r + fr][c + fc] * gradient;
                            dInput[r + fr][c + fc] += this.filters[f][fr][fc] * gradient;
                        }
                    }
                }
            }
        }

        // Update weights (Filters) and biases using Gradient Descent
        for (let f = 0; f < this.numFilters; f++) {
            for (let r = 0; r < this.filterSize; r++) {
                for (let c = 0; c < this.filterSize; c++) {
                    this.filters[f][r][c] -= learningRate * dFilters[f][r][c];
                }
            }
            this.biases[f] -= learningRate * dBiases[f];
        }

        return dInput;
    }
}

class ReLU {
    constructor() {
        // ReLU has no trainable parameters (weights/biases)
    }

    /**
     * Performs the forward pass of the ReLU activation layer.
     * @param {Array<Array<Array<number>>>} input - 3D input tensor
     * @returns {Array<Array<Array<number>>>} - 3D output tensor after max(0, x)
     */
    forward(input) {
        // Save input for backpropagation (derivative is 1 for x > 0, else 0)
        this.lastInput = input; 
        
        let output = [];
        
        // Iterate over filters (depth)
        for (let f = 0; f < input.length; f++) {
            let featureMap = [];
            
            // Iterate over rows
            for (let r = 0; r < input[f].length; r++) {
                let row = [];
                
                // Iterate over columns
                for (let c = 0; c < input[f][r].length; c++) {
                    // Apply ReLU: max(0, x)
                    row.push(Math.max(0, input[f][r][c]));
                }
                featureMap.push(row);
            }
            output.push(featureMap);
        }
        
        return output;
    }

    /**
     * Performs the backward pass of the ReLU layer.
     * @param {Array<Array<Array<number>>>} dOutput - 3D gradient from the next layer
     * @returns {Array<Array<Array<number>>>} - 3D gradient for the previous layer
     */
    backward(dOutput) {
        let dInput = [];
        
        for (let f = 0; f < dOutput.length; f++) {
            let featureMap = [];
            for (let r = 0; r < dOutput[f].length; r++) {
                let row = [];
                for (let c = 0; c < dOutput[f][r].length; c++) {
                    // Derivative is 1 if original input > 0, else 0
                    if (this.lastInput[f][r][c] > 0) {
                        row.push(dOutput[f][r][c]);
                    } else {
                        row.push(0);
                    }
                }
                featureMap.push(row);
            }
            dInput.push(featureMap);
        }
        
        return dInput;
    }
}

class MaxPool2D {
    /**
     * @param {number} poolSize - Size of the pooling window (usually 2)
     * @param {number} stride - Step size for the window (usually equals poolSize)
     */
    constructor(poolSize = 2, stride = 2) {
        this.poolSize = poolSize;
        this.stride = stride;
    }

    /**
     * Performs the forward pass of the max pooling layer.
     * @param {Array<Array<Array<number>>>} input - 3D input tensor
     * @returns {Array<Array<Array<number>>>} - Downsampled 3D output tensor
     */
    forward(input) {
        this.lastInput = input; // Save for backpropagation
        
        const numFilters = input.length;
        const inputSize = input[0].length; // Assuming square input (e.g., 26x26)
        
        // Calculate the output dimensions based on pool size and stride
        const outputSize = Math.floor((inputSize - this.poolSize) / this.stride) + 1;
        
        let output = [];
        
        // Iterate over each filter map independently
        for (let f = 0; f < numFilters; f++) {
            let featureMap = [];
            
            // Slide the pooling window vertically
            for (let r = 0; r < outputSize; r++) {
                let row = [];
                
                // Slide the pooling window horizontally
                for (let c = 0; c < outputSize; c++) {
                    let maxVal = -Infinity;
                    
                    // Extract values within the current pool window and find the max
                    for (let pr = 0; pr < this.poolSize; pr++) {
                        for (let pc = 0; pc < this.poolSize; pc++) {
                            const val = input[f][r * this.stride + pr][c * this.stride + pc];
                            if (val > maxVal) {
                                maxVal = val;
                            }
                        }
                    }
                    
                    row.push(maxVal);
                }
                featureMap.push(row);
            }
            output.push(featureMap);
        }
        
        return output;
    }

    /**
     * Performs the backward pass, routing gradients to the original max elements.
     * @param {Array<Array<Array<number>>>} dOutput - 3D gradient from the next layer
     * @returns {Array<Array<Array<number>>>} - 3D gradient for the previous layer
     */
    backward(dOutput) {
        const numFilters = this.lastInput.length;
        const inputSize = this.lastInput[0].length;
        
        // Initialize dInput with zeros using the original input dimensions
        let dInput = [];
        for (let f = 0; f < numFilters; f++) {
            let featureMap = [];
            for (let r = 0; r < inputSize; r++) {
                featureMap.push(new Array(inputSize).fill(0));
            }
            dInput.push(featureMap);
        }
        
        const outputSize = dOutput[0].length;
        
        for (let f = 0; f < numFilters; f++) {
            for (let r = 0; r < outputSize; r++) {
                for (let c = 0; c < outputSize; c++) {
                    // 1. Find the exact coordinate of the maximum value in the original forward window
                    let maxVal = -Infinity;
                    let maxR = -1;
                    let maxC = -1;
                    
                    for (let pr = 0; pr < this.poolSize; pr++) {
                        for (let pc = 0; pc < this.poolSize; pc++) {
                            const currR = r * this.stride + pr;
                            const currC = c * this.stride + pc;
                            const val = this.lastInput[f][currR][currC];
                            
                            if (val > maxVal) {
                                maxVal = val;
                                maxR = currR;
                                maxC = currC;
                            }
                        }
                    }
                    
                    // 2. Route the gradient ONLY to that maximum coordinate
                    dInput[f][maxR][maxC] = dOutput[f][r][c];
                }
            }
        }
        
        return dInput;
    }
}

class Flatten {
    constructor() {}

    /**
     * Flattens a 3D tensor into a 1D array.
     * @param {Array<Array<Array<number>>>} input - 3D input tensor
     * @returns {Array<number>} - 1D output array
     */
    forward(input) {
        // Save the original shape for backpropagation later
        this.lastInputShape = [input.length, input[0].length, input[0][0].length]; 
        
        let output = [];
        
        // Iterate through depth, rows, and cols to push everything into a flat array
        for (let f = 0; f < input.length; f++) {
            for (let r = 0; r < input[f].length; r++) {
                for (let c = 0; c < input[f][r].length; c++) {
                    output.push(input[f][r][c]);
                }
            }
        }
        
        return output;
    }

    /**
     * Performs the backward pass by reshaping the 1D gradient back to 3D.
     * @param {Array<number>} dOutput - 1D gradient from the next layer
     * @returns {Array<Array<Array<number>>>} - 3D gradient for the previous layer
     */
    backward(dOutput) {
        let dInput = [];
        let index = 0;
        
        const [depth, rows, cols] = this.lastInputShape;
        
        // Reconstruct the 3D tensor from the flat array
        for (let f = 0; f < depth; f++) {
            let featureMap = [];
            for (let r = 0; r < rows; r++) {
                let row = [];
                for (let c = 0; c < cols; c++) {
                    row.push(dOutput[index]);
                    index++;
                }
                featureMap.push(row);
            }
            dInput.push(featureMap);
        }
        
        return dInput;
    }
}

class Dense {
    /**
     * @param {number} inputSize - Number of inputs (size of the flattened array)
     * @param {number} outputSize - Number of output neurons (e.g., 3 for triangle, square, circle)
     */
    constructor(inputSize, outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        
        // Initialize weights with small random values
        this.weights = [];
        for (let i = 0; i < outputSize; i++) {
            let row = [];
            for (let j = 0; j < inputSize; j++) {
                // Initializing with values between -0.05 and 0.05
                row.push(Math.random() * 0.1 - 0.05); 
            }
            this.weights.push(row);
        }
        
        // Initialize biases to zero
        this.biases = new Array(outputSize).fill(0);
    }

    /**
     * Performs the forward pass: output = (weights * input) + biases
     * @param {Array<number>} input - 1D input array
     * @returns {Array<number>} - 1D output array
     */
    forward(input) {
        this.lastInput = input; // Save for backprop
        
        let output = [];
        
        // Calculate the dot product for each output neuron
        for (let i = 0; i < this.outputSize; i++) {
            let sum = this.biases[i];
            for (let j = 0; j < this.inputSize; j++) {
                sum += this.weights[i][j] * input[j];
            }
            output.push(sum);
        }
        
        return output;
    }

    /**
     * Performs the backward pass, updating weights and returning the gradient for the previous layer.
     * @param {Array<number>} dOutput - Gradient from the next layer
     * @param {number} learningRate - The learning rate for weight updates
     * @returns {Array<number>} - Gradient to pass to the previous layer
     */
    backward(dOutput, learningRate) {
        // Initialize the gradient for the previous layer with zeros
        let dInput = new Array(this.inputSize).fill(0);
        
        // Update weights and calculate dInput
        for (let i = 0; i < this.outputSize; i++) {
            for (let j = 0; j < this.inputSize; j++) {
                // Gradient with respect to the input (to pass backwards)
                dInput[j] += this.weights[i][j] * dOutput[i];
                
                // Gradient with respect to the weight
                const dWeight = dOutput[i] * this.lastInput[j];
                
                // Update the weight (Gradient Descent)
                this.weights[i][j] -= learningRate * dWeight;
            }
            
            // Update the bias
            this.biases[i] -= learningRate * dOutput[i];
        }
        
        return dInput;
    }
}

class Softmax {
    constructor() {}

    /**
     * Performs the Softmax activation.
     * @param {Array<number>} input - 1D input array of raw logits
     * @returns {Array<number>} - 1D array of probabilities summing to 1
     */
    forward(input) {
        this.lastInput = input; // Save for backprop
        
        // Find the maximum value for numerical stability
        const maxInput = Math.max(...input);
        
        let expSum = 0;
        let exps = [];
        
        // Calculate e^(x - max) for each element
        for (let i = 0; i < input.length; i++) {
            const e = Math.exp(input[i] - maxInput);
            exps.push(e);
            expSum += e;
        }
        
        // Normalize to get probabilities
        let output = [];
        for (let i = 0; i < exps.length; i++) {
            output.push(exps[i] / expSum);
        }
        
        return output;
    }
}

// === LocalStorage Model Saving & Loading ===

function saveModel() {
    // Collect the learned weights and biases from the learning layers
    const modelData = {
        convFilters: convLayer.filters,
        convBiases: convLayer.biases,
        denseWeights: denseLayer.weights,
        denseBiases: denseLayer.biases
    };
    
    // Convert the JavaScript object to a JSON string and save to LocalStorage
    localStorage.setItem('cnn_pretrained_weights', JSON.stringify(modelData));
    console.log("Model weights successfully saved to LocalStorage.");
}

function loadModel() {
    // Try to retrieve the saved JSON string
    const savedData = localStorage.getItem('cnn_pretrained_weights');
    
    if (savedData) {
        // Parse the JSON string back to a JavaScript object
        const modelData = JSON.parse(savedData);
        
        // Inject the saved weights back into our global layer instances
        convLayer.filters = modelData.convFilters;
        convLayer.biases = modelData.convBiases;
        denseLayer.weights = modelData.denseWeights;
        denseLayer.biases = modelData.denseBiases;
        
        console.log("Pre-trained weights loaded from LocalStorage!");
        
        // Update the UI to let the user/checker know
        const statusDiv = document.getElementById('trainingStatus');
        if (statusDiv) {
            statusDiv.innerText = "Pre-trained model loaded from LocalStorage. Ready to predict!";
            statusDiv.style.color = "green";
            statusDiv.style.fontWeight = "bold";
        }
    }
}

// === Global Model Initialization ===
// We create the layers globally so they retain their memory (weights) between clicks
const convLayer = new Conv2D(8, 3);
const reluLayer = new ReLU();
const poolLayer = new MaxPool2D(2, 2);
const flattenLayer = new Flatten();
const denseLayer = new Dense(1352, 3); 
const softmaxLayer = new Softmax();
loadModel();

// Helper to run a full forward pass
function modelForward(inputMatrix) {
    const out1 = convLayer.forward(inputMatrix);
    const out2 = reluLayer.forward(out1);
    const out3 = poolLayer.forward(out2);
    const out4 = flattenLayer.forward(out3);
    const out5 = denseLayer.forward(out4);
    const out6 = softmaxLayer.forward(out5);
    return out6; // Returns probabilities
}

// Helper to run a full backward pass
function modelBackward(lossGradient, lr) {
    const d1 = denseLayer.backward(lossGradient, lr);
    const d2 = flattenLayer.backward(d1);
    const d3 = poolLayer.backward(d2);
    const d4 = reluLayer.backward(d3);
    convLayer.backward(d4, lr); // Updates filters!
}

// === Train Button (Training Loop) ===
const trainBtn = document.getElementById('trainBtn');
trainBtn.addEventListener('click', () => {
    // Get target label from user
    let label = prompt("What did you draw? Enter 0 for Circle, 1 for Square, 2 for Triangle:");
    if (label === null || label === "") return;
    const targetIndex = parseInt(label);
    
    // Get Hyperparameters from UI
    const lr = parseFloat(document.getElementById('lrInput').value);
    const epochs = parseInt(document.getElementById('epochsInput').value);
    
    // Get image data
    const pixels = getPixelData(); 
    const inputMatrix = toMatrix(pixels, MODEL_INPUT_SIZE); 
    
    let finalLoss = 0;
    
    // Training Loop!
    for (let e = 1; e <= epochs; e++) {
        // 1. Forward Pass
        const predictions = modelForward(inputMatrix);
        
        // 2. Compute Loss and Gradient
        const { loss, gradient } = computeLossAndGradient(predictions, targetIndex);
        finalLoss = loss;
        
        // 3. Backward Pass (Update Weights)
        modelBackward(gradient, lr);
        
        // (Optional: You could use setTimeout here to animate the UI update per epoch)
    }
    
    // Update Dashboard
    document.getElementById('trainingStatus').innerText = "Training completed!";
    document.getElementById('currentEpoch').innerText = epochs;
    document.getElementById('currentLoss').innerText = finalLoss.toFixed(4);
    document.getElementById('currentAccuracy').innerText = (Math.exp(-finalLoss) * 100).toFixed(2) + "%";
    
    saveModel();

    alert(`Trained for ${epochs} epochs. Loss is now: ${finalLoss.toFixed(4)}`);
});

// === Predict Button ===
const testBtn = document.getElementById('testBtn');
testBtn.addEventListener('click', () => {
    const pixels = getPixelData(); 
    const inputMatrix = toMatrix(pixels, MODEL_INPUT_SIZE); 
    
    // Use the trained global model
    const predictions = modelForward(inputMatrix);
    
    const classes = ["Circle", "Square", "Triangle"];
    let maxIndex = 0;
    let maxProb = predictions[0];
    
    for (let i = 1; i < predictions.length; i++) {
        if (predictions[i] > maxProb) {
            maxProb = predictions[i];
            maxIndex = i;
        }
    }
    
    const predictionOutput = document.getElementById('predictionOutput');
    if (predictionOutput) {
        predictionOutput.innerHTML = `
            <strong>Predicted: ${classes[maxIndex]}</strong><br>
            Circle: ${(predictions[0] * 100).toFixed(2)}%<br>
            Square: ${(predictions[1] * 100).toFixed(2)}%<br>
            Triangle: ${(predictions[2] * 100).toFixed(2)}%
        `;
    }
});