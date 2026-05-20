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

// Test the Full Feature Extraction Pipeline
const testBtn = document.getElementById('testBtn');
testBtn.addEventListener('click', () => {
    // 1. Get raw pixels and convert to matrix (Input: 28x28)
    const pixels = getPixelData(); 
    const inputMatrix = toMatrix(pixels, MODEL_INPUT_SIZE); 
    
    // 2. Initialize layers
    const convLayer = new Conv2D(8, 3);
    const reluLayer = new ReLU();
    const poolLayer = new MaxPool2D(2, 2);
    
    // 3. Perform forward passes sequentially
    const convOutput = convLayer.forward(inputMatrix);       // Output: 8 x 26 x 26
    const reluOutput = reluLayer.forward(convOutput);        // Output: 8 x 26 x 26
    const poolOutput = poolLayer.forward(reluOutput);        // Output: 8 x 13 x 13
    
    // 4. Log the results
    console.log("MaxPool Output Dimensions:", poolOutput.length, "x", poolOutput[0].length, "x", poolOutput[0][0].length);
    console.log("Sample of final downsampled feature map:", poolOutput[0][0].slice(0, 5));
    
    alert("Feature Extraction Pipeline Complete! Check the console.");
});

// === Math and Matrix Helpers ===

// Helper function to convert 1D array to a 2D matrix (e.g., 784 -> 28x28)
function toMatrix(array, size) {
    let matrix = [];
    for (let i = 0; i < size; i++) {
        matrix.push(array.slice(i * size, (i + 1) * size));
    }
    return matrix;
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
}