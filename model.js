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

const PRETRAINED_WEIGHTS = {"convFilters":[[[0.03513972219084499,1.208785191082763,-0.27178344241616426],[0.5499888864065534,1.0154454025740678,-0.7809329208043182],[0.1059293348208799,0.8506978928350816,-0.9575571336188633]],[[-0.31829453483002523,1.0699730805842433,1.070634155131129],[-0.6467818829753805,0.39557327602480696,0.4136265328484108],[-0.5443764663002583,-0.1062312038832481,0.08165072082614595]],[[-0.2385657969461997,-0.15560387306772577,-0.40623481609790557],[0.17706893522876527,0.03486977700738974,-0.036403390043409825],[-0.07964225491160906,0.5787513361835518,0.030196810021813486]],[[-0.16148203579689563,-0.16097479561216219,0.012507756120914697],[-0.7320116925936461,0.9013565769101799,0.8092190511114018],[-0.01816093113285888,1.519344235902069,1.387286554505831]],[[0.14605578349522083,0.08302646033408524,0.5422700779035147],[-0.38816163946612037,-0.4027140103589151,0.250523530100434],[-0.18554398535539743,-0.14811651394022343,-0.20638563919242428]],[[-0.30938865420479367,-0.5822717710810372,0.81444313107912],[0.14109739853138084,-0.3437620383086649,-0.08040192057617898],[0.17221424250805656,0.9047232931107022,-0.7344153428118617]],[[0.4512093681091315,-0.06430985930914536,-0.17225242467788474],[-0.3484489957171259,0.14532694842481655,0.0540753671042183],[-0.41107635824530225,0.11041059988178982,-0.12149503019793775]],[[0.14157519005230898,0.7459990120101647,0.30270931002370893],[1.1194979603495274,0.7112634579918249,0.8675024736597303],[0.1396134482600077,-1.130344752471133,0.13607292576681476]]],"convBiases":[-0.008572697551017345,-0.04727360418154695,0.017747795380598477,-0.3944699301741373,0.06410209021793595,-0.012394390080485382,0.06343164656448877,-0.4283052144919458],"denseWeights":[[0.043456326435261386,-0.05486958580565791,-0.11052172327569554,-0.16388907652120424,-0.08803664946805397,0.06745011885602918,-0.04444955711225275,-0.07958469183739575,0.127745366449363,-0.11581391591011893,-0.06052814629021086,-0.012345067891791038,-0.14846687768731925,-0.039073015874798514,-0.06427892997962041,-0.03747521975362343,-0.12022665298741943,0.006313251843912739,-0.08449391397828213,-0.003972402407113042,-0.27011908551751346,0.23515708969916024,-0.033344327046202396,0.004602993001326892,0.003670357534628999,-0.09892069313777098,-0.04168062091142329,0.11220429982108608,-0.02365244157150378,0.0719639596931657,-0.17045430017140745,-0.23353195574122965,0.26390927464688047,-0.13809150937996054,0.2770598975476987,-0.15246282464100092,0.0013403619498816367,-0.012889803425726884,0.026626267895154345,-0.03796556245683542,0.11756336902418671,-0.04274840826049364,-0.044861427235129044,0.07986878071391842,0.017079630286194115,0.15510645713087415,-0.08773796026884577,0.2119823911097694,-0.12349500279425703,-0.02171382296968659,-0.18783541373323376,-0.13098041741354627,-0.059345212108164014,0.16909634500349377,-0.05794740505007077,-0.003161288139862514,0.08139727355467645,0.13925114889920653,-0.05753099225605988,0.04140381375864659,-0.102547672653384,-0.3281308711819925,0.16352488693622394,-0.09693261838038338,-0.1315767524726446,-0.07063365388597968,0.2937528893089108,-0.06981804776955597,0.0234532618926878,-0.14018715721355707,0.06757632248968833,-0.055014879767044576,0.0407271219538902,0.02185555627847895,-0.1494597539629676,0.28508267352515937,-0.1898966050278775,0.008896047178125156,-0.16707874706984993,0.08108277977316852,-0.2175289014089189,0.07404579419909739,0.20615785091159333,-0.00538439213794611,-0.048225571920082325,0.11214644158442297,-0.0013540833991945314,0.07052540059121541,-0.003526009599815241,-0.15302125607141726,-0.05948833009594577,-0.1372542503379369,0.05159139858649025,-0.005999389985943892,0.04893193427008202,0.10472778144191597,-0.04631049488463169,-0.049693256395749646,0.2662038333502635,0.09652249362624124,-0.034732377548846026,0.011721717558288774,0.20667553743174216,-0.011426471393553638,-0.20595818589401127,0.203760955771009,-0.053778474371772446,-0.0951676829028919,0.23096924284450465,-0.12023737075618164,-0.032547230203068864,0.0016390695587375308,0.11746882009975823,-0.029683231854770217,0.11138554065667829,0.1450134837743915,0.11237368404896401,-0.23335950116564586,0.26445162336498707,-0.07367850320736602,-0.08248510078151293,0.09439682103022072,0.07793989310618933,-0.007516342569776769,0.039307780116740895,0.2855829024070706,0.034404564733255365,0.1741803502745131,0.014432179808132158,-0.012779494762979668,-0.2266661072794117,0.043279959543605195,-0.028412831684491234,0.010725965851373157,-0.11237047814883104,0.2920398016251394,0.13173801005580754,-0.08238074132069875,0.11677976219300529,-0.007962189908849124,0.1348365274910626,0.07126361515594239,-0.000319587704862377,-0.21818278476646263,-0.030851127820510298,0.057893276031792354,-0.16294049711319494,-0.16915571246960898,-0.0166467006308525,0.06427329128118914,-0.1620526872290871,-0.07047919228741784,-0.15874032749020833,0.003153642592447379,0.029589442246215063,-0.07708101878953644,-0.15985359576577451,-0.12705184364803487,0.00646809248697437,-0.03916474382709382,0.09042534562826283,0.06384178949688193,0.046487960590956,-0.05101581941775839,0.016757728112912266,-0.02991641517431044,0.1171684463991915,0.043559458845029424,0.002945219209443933,-0.027160646717031285,0.009382324714822114,-0.017700638330301905,-0.01583858112320989,-0.013532037428578175,0.052136371508130636,0.006726355347655617,0.03967960402630524,-0.10375072605735648,-0.10452240514330699,-0.04110359997495496,0.021329979759736988,0.006881730112239822,-0.022433826390882507,-0.0844044333975544,-0.14339619229743483,-0.002126878818634158,0.1032432766385005,-0.06999860483808211,-0.014843304778879752,0.19951880303338754,0.034986036910683915,0.013706207454920672,-0.07628080202179632,0.09893785557689855,-0.05003278049038363,-0.07083573230333823,-0.07073263744019717,-0.06939460367556423,-0.04234471798968102,-0.030067396623454264,0.16780218691427634,-0.04553586286718908,0.08747650194714011,-0.13734818474198685,0.0649220876229958,-0.11027962243418449,0.01333786952830606,0.0753530152734436,0.022946931046679265,0.0624425609680036,-0.18517228547926792,-0.026635512219675105,-0.16334320173719474,-0.19111434347760414,0.13150981721961555,0.09623555336534748,-0.07769415806209314,0.07790932143400699,0.09792158866422382,0.07630259647212399,0.11281970907476084,0.0053509959575294514,0.01939748516331269,-0.12662427817774874,0.10588838582704332,0.13066714498168025,0.013066841601532516,-0.034551085996580466,0.06288014248095888,-0.17616751759255816,-0.04734408806303259,-0.0225825425947494,-0.04816200730604193,0.002536315643268612,0.11934761857583423,-0.02149142117346045,-0.11395988044060851,0.06281656590515775,0.10499695292023555,0.14846421345086316,0.02793652974994106,0.04360986985603884,-0.21127748009978375,0.007569033266396871,-0.1714132812637197,-0.0553454468236189,0.031220564374038286,-0.020554410162171893,0.11584282168084445,-0.12217468089359869,0.018105301464800164,-0.044074502469504846,0.022111773607980105,0.18054681676885806,0.12204745496925481,-0.10429468566973303,0.1491257659645549,-0.2454198512896022,-0.03232266175777403,0.13144580364688402,-0.1309642516705736,0.2286556362994679,0.06542688314369322,0.1973034272776322,-0.026863968015103195,-0.06760749418303148,0.019307329847947106,0.12480903879737658,-0.11747824637752527,-0.10898068736833011,-0.056139921581521535,0.07104663772217147,-0.042555390337544484,-0.005418293918364207,0.22472370863469898,0.006463371818040557,0.029748217508796505,0.06804727917640413,0.12014170992284752,-0.11344575026231886,0.09390231595268743,0.1623088671276431,-0.13138232874036004,-0.15149459472785695,0.11439726799016839,-0.2136917484799165,0.019341210223292875,0.1118394871323403,-0.14552190871662823,-0.0777364587563253,-0.013772902995910306,-0.02687008949404454,-0.04222108919628441,-0.12081441049794003,0.0343097724607223,0.03781405716553516,-0.14261413060784606,-0.07436430159856963,-0.2906329418319982,0.014632796036041566,0.24424549362000308,-0.0036033625295636466,-0.013471012320419851,0.126665172674561,0.23792641440553347,0.03477738889533172,0.041053389850689855,-0.060307717098048226,0.008396022367373883,0.02149297012189869,-0.058914734690939066,-0.05954690785469398,-0.017721804002484125,0.16080656297846802,0.015894666341297922,-0.10500363428713062,-0.043432195995275265,-0.04457139502799414,-0.1396991748885776,-0.07196041802687562,-0.041210275417966256,-0.007515025781322151,-0.04583531019636152,0.006061469888080106,0.0015237053223367922,-0.19245365844869225,0.04302522939154168,0.0422609032728907,-0.04031398166158232,0.015605563954041116,0.02841499260892202,-0.009166762845569821,-0.03762667782547425,0.01982855904782703,-0.005150389760349215,0.04382404624944401,-0.03483568923939056,-0.06369193601331584,-0.013057316981694732,-0.04505526062594285,0.04567592137414966,-0.04490186606030694,-0.00907527932566933,-0.015088399206394187,-0.04110434465209474,-0.015214614463905485,0.10765398064764249,-0.015897175070247408,-0.005133403808540329,0.04737459701028602,-0.013648100914784118,0.016796449943911024,-0.012885604151412237,0.00929052189016881,-0.034561227528443254,0.03896158963912413,0.03229356924937948,0.011259982000336082,0.025203336724182897,0.03147880611992014,-0.0041464302807855445,0.003119370845329976,0.016098238091930613,0.03540935222310848,-0.011192171622828631,0.012425339161327806,-0.014502384122613858,-0.03711278620680237,0.04695245050732214,-0.0643943672703979,0.07703234097474133,0.02711469859358855,0.08436134332417262,-0.01788938018639592,0.005372578563398916,-0.04948982340097691,0.006966705500723532,0.03711131541745968,-0.018435505171107536,0.046294745755237704,0.08297285645370639,0.13217782095033118,0.017703806888736778,-0.05152428610500183,0.0026011788152942752,0.06442282305573332,-0.04337834435863386,-0.04390615132373122,-0.051246543868685726,-0.014055929278258092,-0.0013086228540109656,-0.01791693775726284,-0.023298712876204882,0.03173632220983903,0.0569373032127289,0.023264938746923314,0.05377808649291961,0.04028534875924985,0.004080462373010295,0.03468406374448864,0.04880923159603546,0.0061687884098488854,-0.01487698542133457,-0.07067403528854556,-0.03989991949860646,0.0414868228086082,-0.004971576891314739,0.0636409193113933,-0.01064012511021417,0.0894633489186329,0.04114546553967449,0.09917015628741359,0.007220848966107833,-0.05773583128507792,-0.05561618233072788,0.028014111236085208,-0.04956773625798385,0.034147016343150066,0.0006666768680900012,-0.023677138257719672,-0.010110593553969273,-0.019890929981121123,0.04033869466430062,0.023033844046776122,0.004767569306684564,-0.08753546335162148,-0.013825181535791566,0.02514851721260313,-0.12385866610960018,-0.036671033465257775,0.09077396106118489,0.05003558668791882,0.07049089775375114,0.020635153119671797,-0.02008789009088414,-0.06590590597312834,0.07312145145918976,-0.019538476184125634,0.0259903116679631,0.005751444055432525,0.03773800516591675,-0.06726189164700644,0.045530833144643866,-0.0066120258919997585,-0.029791426798446378,0.014419464589912584,0.017174206299119973,-0.08950223281637729,-0.006422886095442418,-0.005673064795711127,0.03815031266576144,0.005491187515643024,-0.062194528351091675,-0.07176135709125202,-0.062102207148573084,0.01574005305432574,0.05477000835282988,0.023426317698929937,-0.030397173739372604,0.06629313889913474,0.07035585778444019,0.0693356527093923,0.027298539927340625,0.030335220060200944,-0.005869268289280609,-0.0030288407797752497,-0.05271608080757371,-0.02140061793679548,0.003992464312408973,0.014517724172868802,-0.014876242234451758,-0.058862181356255476,0.00825584578065881,-0.03893052811642194,0.044194224685359426,-0.018654730474197682,-0.023776049828199773,-0.007810654196668161,-0.05145628743292168,0.04184201976291013,-0.023001612611725206,0.03547331469992418,0.030497526432696428,0.0036821147014280365,-0.019226586777952626,0.008992619668699216,0.007113535686459868,-0.001053740251498659,0.053517427697666056,-0.0012692535289468435,-0.05884527762897661,0.012531431361171887,-0.017442257747056232,0.0009704058882345813,0.023583481775942106,0.0036889173132699632,0.0128480378729591,0.05792031540954743,-0.017813216978109968,0.02208820092026789,-0.0397241622915081,0.0646814938241296,0.02921143398310239,-0.013912044260303114,-0.009430587134456426,0.024347144658391266,0.017398665170515783,-0.077102513508198,-0.09203219324786138,-0.12130796130881945,-0.20818900381039274,0.29181200544975927,0.0432457714749694,-0.017704428230656783,0.05513865074307114,0.08565980094470452,0.018398582664638938,-0.07641196639635164,-0.017483830073276037,-0.1821735021752021,-0.07888871315198866,-0.022649340192772923,-0.0028034324568152625,-0.033358217614782325,0.10610371070033732,0.09519794260766394,-0.03802394235915496,0.38540309203781065,0.15669954662819655,-0.0788697899520172,-0.06538143661261651,0.0863277114211072,0.08012281570395882,0.07039273938350439,0.17728539294328338,-0.18003787013804343,-0.24483281090020806,-0.33926191184769305,0.14475071039118753,0.26386359699947554,0.04930078246346183,-0.031551934829125904,0.16752371421053658,0.007293659227015145,0.1537622080489832,0.24644382509654933,0.11576191887178554,0.13970409142804913,-0.08819822803567418,0.12397750528946816,0.13975331304499009,-0.13081531086772696,-0.09439314948462824,0.3007227339600563,0.1433234620436772,-0.16630995871315699,0.006861117541080026,-0.059948205449963606,0.05706370822272263,0.24157913222230365,0.12938273374314033,-0.17095993183732186,-0.04345887588305322,0.04138311158111878,0.029993689870869276,-0.22989103780633222,0.028759529829283478,-0.09739505330227705,0.22463355453223358,0.06848660496772191,-0.15917105507203752,-0.06510670397899744,-0.06805674442566197,0.16514608522284827,-0.2163556683899423,0.008075336325671045,-0.10450294630409522,0.049360217803978555,0.09328575842718473,0.13690241366698497,-0.13798460600620338,0.21212373243542612,-0.24124706368017165,-0.050982695204224465,0.02323717917280299,-0.23407407415524742,0.1794322419759885,0.07720818879678001,0.12084663244890896,-0.11720417197353793,-0.34737032822358965,-0.13975771935645404,0.14421964656524383,0.05160288135224201,0.12037591181040168,-0.14643530407108787,0.0205563534875003,0.13129642798306468,-0.20369929205327447,0.47581432218908365,0.0678097971523008,0.04605340021059062,0.021460252653206204,0.2512071600260393,-0.05741283102720742,0.23622456230801087,-0.10624954173126762,-0.14761297645496282,-0.11576277261889285,0.2809032118748537,0.011554473774757278,-0.10149750783832742,0.1729761576921567,-0.41851938711645825,-0.2898884671357376,-0.142608532670858,-0.14008469294791506,-0.2934745316231429,-0.09074647994228334,0.292067367608016,0.023991998124359863,-0.03907300630321903,-0.006398491421596374,-0.36414799183201924,0.0395947284836493,0.266834415016752,0.00859159032871016,0.06112660475280864,0.31467750057593685,0.3313128512979449,0.1452788050377735,0.08232797529481191,-0.0009762717206730734,0.09844381546024482,-0.057164082213885914,0.1635486255117852,0.009616484540556285,-0.18076207338299424,0.13875298436163783,0.0047580943096477354,-0.32034723339529014,-0.2077094854275164,0.035636181966878636,-0.30171335653857606,-0.05978856973268916,-0.0035545867206606308,0.013002618434868725,0.09892859488994109,-0.021067093427967754,-0.13704374482373202,-0.32005995761318046,0.054039673464085246,0.06620843904795007,-0.18034479177356588,0.012712923130704585,0.0641128761910519,-0.05233894815244697,-0.18730080847820801,-0.05574146212057184,-0.05735320439797204,0.13343744391410087,0.06548407803093487,-0.05708098388239941,-0.23405877194989697,-0.15553696974576503,0.12503481719403342,0.17254660540870742,0.17885066854330328,0.07424682787264697,0.08967906120607931,0.10700739033170527,0.08134326610742597,0.018026713076143463,0.03684042931712577,0.02232732275704392,-0.005455823846290401,-0.03857633406283406,0.006400355919244169,0.024693256138502335,0.06181977642307759,-0.009076141012531048,0.008175081926070178,-0.02325519261548548,-0.010386179341223883,-0.0537062618269117,-0.04146118608228289,0.004674793437499782,-0.04359936445649735,-0.000814302312788437,-0.046305511879204134,-0.01942546211138123,-0.029439812400956,-0.044144423523743107,0.01244065579878835,0.01404368497926931,-0.016595741914015215,-0.0068169080721008826,-0.08485302839834087,-0.016581431105374928,0.043978401179766334,-0.038888844932321594,0.026875281912258457,-0.04822148713821073,0.0007300481016209037,-0.024235948956993877,0.03740872707353742,-0.027810535143183265,0.08998234371374787,-0.057505228296519426,0.09383809054038571,-0.004746857255191159,0.031371040213858954,-0.004146226680745318,-0.026788651981741154,-0.034590890189148654,0.04042584357520611,-0.026819337711737282,0.013238586692292118,0.01443581471745881,-0.04530299333695225,-0.06467979206696983,-0.02589971189771116,-0.054174478112450436,-0.06287615092259408,0.07157177241393915,-0.03071773193080938,0.11479168220650424,0.04018242322932464,0.0246042383572066,-0.006996309909934725,-0.04671302463510519,0.08433926139362807,0.037415310600715665,0.010651920068762527,0.0147630605346283,0.04109255747991231,-0.005597194430632899,-0.0033857047225138315,-0.07400482879737223,-0.031518888113345946,0.0024530455640693633,-0.026687614362189253,-0.051913646655982734,-0.04720035533111267,-0.03775503754781845,0.09750431615795674,0.06936517523870593,0.034401045138541655,-0.09685623683219344,-0.03166249931570568,-0.04071879726635812,-0.061416293180816595,0.04338365583551535,-0.10772521706449972,-0.029272240921603333,-0.023985701529264217,-0.023784058938230532,-0.02473019008253188,-0.09255100402122322,0.028116501886034193,0.18214584059188454,-0.020535147288893624,-0.044591064972327205,-0.008269309676272135,-0.03163122482686556,-0.0456055530128607,-0.0922894482591268,0.023920978337919653,0.024720037150377912,-0.03121819361404656,0.033205056350766204,-0.0177538761700123,-0.06647797250311496,0.006762017528865819,-0.0003858318640397705,-0.05978596695732539,0.027586549655846036,0.025655165696500403,-0.05700548604903121,-0.08034485408972301,0.029853992285660314,0.03964311617095993,0.02175400154406356,0.10604767625888648,0.021860347537736604,0.11762488445710348,-0.026212166597483315,0.07839383143001359,0.06771546808618328,-0.055471254306689234,-0.07814069870024165,-0.013292258214901185,0.029087518065591466,0.10656123420064463,0.028860640687688285,-0.05209512947507017,0.00468589443424943,-0.08167979327323142,-0.05429377742012368,-0.008031463650331926,-0.08435292830114126,-0.012799904865876332,-0.013524570220196297,-0.028059838388885597,0.0007958314400028416,-0.07834655272959781,0.10169721758053032,0.006402743970505667,-0.0011977148872319467,0.07301540247346555,0.08092406186048207,0.033562842577802746,0.03954859629966236,0.01715184400467523,0.011869074847409818,0.004302893627786057,-0.02672895096550912,-0.000585034065216466,-0.024982022886882647,-0.03638343462002074,0.04594363795571818,0.006089404355692074,-0.02591561157743486,0.011228960104182341,-0.03256931806533665,-0.10567089530165652,0.01701038542583344,0.060261123948783885,-0.017774083665403412,-0.04102708623484063,-0.034159746752619714,0.031105897454377338,-0.10082253157541599,0.008057200089310977,-0.0003398372965302336,0.006192840837722401,-0.05159711346457339,0.020999639822458803,0.03560122392464471,-0.0031390895602747874,-0.030943328076505877,0.00039849914072019165,0.012848432837659114,-0.007688247430765058,-0.05336442386343513,-0.03860010468233082,-0.07011507824466445,-0.011258434596774355,-0.0792367284464983,-0.013108215899772038,0.07905327035479874,-0.08679372419234575,-0.020895825107544497,0.049153737561216625,0.0217832998978212,-0.010065890990606815,0.08553437303867238,-0.02516282407254955,0.02640170157509758,-0.001297175177028003,-0.006620810135751203,0.01457534098579241,0.002206568119980889,-0.12265189215748726,0.0792606080624275,0.001756779232547879,0.0294289503364674,0.025914957837255486,0.022723062336103675,-0.05295943510405885,-0.0014006441940588431,0.009401060767304839,0.05846921467248557,-0.0012152109947147507,-0.01597608945879929,-0.001862444935203605,0.07903341752362512,-0.009894057021657639,0.10699243914007957,-0.06257253467966996,-0.06825705515808256,0.07404611644490423,-0.028889634881587193,-0.0585587473667212,-0.000008859746708583779,-0.0001650923195081461,0.01457067064736007,0.015196469325853612,0.040771938368206424,0.0835987109637206,-0.0591715997841799,0.13816824780341175,0.13217923091125872,-0.031243875422254774,-0.10626021333024757,-0.10592607638459343,-0.10545225612279584,-0.07124256775742796,-0.019784280603593385,0.031257952279884225,0.15508455723807987,0.001433732539617298,0.13410756371338556,-0.017879793718682608,0.052252893297190704,0.004972814835601301,0.12522649389797458,-0.03697723821230349,0.04050664802158695,0.01994261519950192,0.05079394966481293,-0.03663761261801662,-0.03802358659873372,0.0015278015484105063,0.08029368626581565,0.044712069320453074,0.03157378384673189,0.018888362245980617,0.06701001566897725,0.0019592349865890997,-0.01064136038132333,0.012928177705337374,0.03800276653548409,0.023369958285197622,0.0445281738938207,0.027719042550073427,-0.03790811701274976,-0.011509780616285332,0.028107867604178497,0.024940823198385552,0.06471153062835401,0.05841321131718422,-0.010734393629113588,-0.14099274716803717,0.05271236818953623,-0.018060709037219503,0.016617212284330184,0.07303982806094114,0.022763709544330588,0.12414097771095935,0.0018370892378772618,0.02100168918446841,0.062156233878234116,0.03034288533747817,0.07965322755839631,-0.03980632396823184,-0.06305553276511887,0.07527057477362399,0.06272819338978022,0.04104483465481944,0.21688136225764607,-0.006463798637186155,0.057179708805710924,0.10473019017322396,0.0648686377977896,0.10346124040130707,0.09736305829889179,0.042801400978528324,0.06087649182363692,0.0822674427857531,0.053934014709659965,0.011226451758246544,0.013636496389684956,0.03298646032481455,-0.023315542897346345,0.05034590996640951,0.004584130186226093,0.08918555709285593,-0.06242261862522146,0.14795525420911296,0.19057886892462358,-0.0468791440591813,-0.004259410389878291,0.104266318238314,-0.008230593506575233,0.08130858712098868,-0.019266363049352732,0.09665131676150493,0.05621997779214465,0.1036986425153628,-0.004614433191043562,-0.0024912647335927807,-0.06007117198200348,0.0023055663184794444,-0.06969752953791694,-0.04931817028372111,-0.017255436086670965,-0.05995854306017054,0.04942069193003942,0.006293751585471443,0.06606516478348114,0.023026167529453835,0.0035894728599008474,0.06932968239848895,-0.01538325574462606,-0.0050522346562523164,0.0729288303749645,-0.0008487914693565461,-0.005129993645398326,-0.006315091835393902,-0.010599018386042392,-0.10103247876442877,0.07183379319830678,0.033648804113979365,-0.01593344483252931,-0.01627485181517592,0.03835704300549924,-0.01451047271679996,-0.017196854468640586,-0.052018252978666836,0.04289722032015562,-0.007465995776239641,-0.022891734432850034,-0.01252053508533775,-0.00412313082665841,-0.045943420904952466,-0.02238473919619729,0.002685038069330083,-0.008829153582430487,-0.019960931597343152,-0.03719998033624202,0.050039438973736916,0.0066985338350965375,0.0005979113792421439,0.04142301280464788,-0.031276759675952316,-0.04157543409336906,0.0015304554906917268,-0.0249860640512146,-0.03698009564709799,-0.04583588729864458,-0.022307635139608338,-0.053405526388141174,0.03952053688130648,-0.06502017546661479,0.018221303703759725,-0.01444568359313169,-0.033979334194607365,-0.029168078801881055,0.04316789045179936,0.019760818452653626,-0.014727052391651988,0.03272104060930307,-0.0009044152697650852,-0.038635072988563356,0.015631950641559998,0.05525990485466054,-0.014123930716757655,0.08723943546039185,0.04428198696503842,-0.017899499636419448,-0.029320702091881395,0.0024023196682634363,0.024333861107640818,-0.02670168114940999,-0.03326747088941625,0.03230232408288049,0.04060341737022113,-0.025972390480244585,-0.01358789436026744,-0.03741133785086522,-0.011419041146495282,-0.00788623840713807,0.03421487976681058,-0.026364164326498626,0.01133838906824161,-0.04018826024554553,-0.06721929542233222,0.03129551512580887,0.003437150421851312,0.09221461493279377,-0.020780627260304725,0.04830135351765904,0.01573904983324402,-0.013776386426008898,0.03875108387329839,-0.02114491813861076,0.026330242766937743,-0.02798054126485814,0.010827135393036124,-0.0084358906775822,-0.027397857068717443,-0.006099820784150865,-0.0005608158054700393,-0.0078077440652314705,0.051009543724382884,-0.00396116139484848,-0.018957509067506012,0.037861278049131454,-0.02852361951147019,0.055748576720057286,-0.01353434135091249,0.02515010316811237,-0.012553086010136737,-0.015479406558239078,-0.054028995176250164,-0.03030190610973133,0.07617473607609108,0.003552706223074092,0.02467258522991978,0.0010717496479296354,-0.05947821985589594,0.004937507807076326,-0.04504572499497971,0.021566369718547813,-0.04436197934859045,0.03519335691279085,0.002344279575363855,0.0361181142278574,0.062152963268189795,0.012590039222606708,-0.053233911709886644,0.05029822689996475,0.04095192245620824,0.06985976445033507,0.05387389227022101,0.05194123074594777,-0.04973419325462549,0.04882844040795763,-0.0006314290420536881,0.06668220809664847,0.08417418548697227,0.023268053171961057,0.031669220990441733,0.088747633249371,0.03030311222749226,0.09037525260477212,0.03481453614209065,0.009765828461785789,-0.04965179895704328,0.03294984672271656,-0.006716069039492487,0.02413633849048709,-0.01323873372365374,-0.08266580397950957,0.06089115645452692,-0.05790854148925066,-0.004132420148927057,0.017591157161271524,0.034365736517562986,0.00773826531067109,0.010600788078456304,0.007739581169501136,-0.04264129887153934,0.02689413541740069,-0.0068691880767831805,0.03589245343819051,-0.03065338568928885,0.04774800795856248,0.03656275829008355,0.013353469879872664,0.09083242348042288,0.0013274872471638788,0.014995628598568319,-0.0002796970703090729,-0.00870867288385714,0.06263752447179473,0.016628887610241408,0.01773860481386329,-0.033565650901785066,0.0025737930361718563,-0.03379932995189367,-0.06501080563251438,0.04744507124468094,-0.01621029035970057,0.07719281216475611,0.04267796576279111,-0.027584279852266743,0.03542915208251459,0.03564959072127311,0.0038200598097454332,-0.0075934939367485865,-0.008294368951270156,-0.02180994454221689,-0.028881453145754032,0.01161271484677179,-0.014342034218140964,0.0494124588700952,0.01200897547043973,-0.06541317161389192,0.00217965207420478,-0.004032768957949426,0.005246554271148256,-0.09974492891985959,-0.08637211448655446,-0.15673158723027117,-0.18571224861319507,-0.1708064132540579,0.13665094501076802,-0.04730783285573414,0.14776668217201722,-0.03349466811520524,-0.12205979222112265,0.021730423401604766,-0.08249792400298477,-0.0772831542631291,-0.012458547224741177,-0.07833204972986482,-0.041280915857537695,-0.12573843674023746,0.15261288153964214,0.274704878207728,-0.10342876287611837,0.1263453852243849,0.42532002023173304,0.1485802612116419,-0.016112964756101297,0.08380786349638265,-0.04252772910525376,-0.025662225501792712,0.04432378664883516,0.12776599594000776,-0.18077959891555465,-0.276462801731406,0.09293073876732272,0.08807967943826674,0.05705880577385504,0.0024382214852657837,-0.1842866541369368,0.03694712156630283,0.028970024002818003,0.04674359425356971,-0.008937042801922339,0.16812661632081616,0.10387391876474296,0.12058651506090451,0.026376229001910962,0.10791944891486623,-0.05971736931073178,-0.14116104790952178,0.004838500065175421,0.1055110912714157,-0.051439503780860096,0.10101633996951981,0.04794341409967673,0.10822169472385537,-0.006093796947154282,0.010445675989814102,0.017114156505858,0.1716548198173819,-0.09469398664334355,-0.18913728705488372,-0.14439882128184625,0.10241784019360159,-0.11316704385552173,-0.08831387210497664,0.14577960583551747,-0.36428320678879517,0.2718692438186897,0.06769392928005585,0.01461233639503191,-0.30628387020298353,-0.12726961190489858,0.033315167598173115,0.014464842458021045,-0.11225638468981873,-0.05512455904105784,-0.07432085823424048,0.1703177618374331,-0.04201113043003493,-0.21764358807163137,-0.03215985241636841,-0.03637127305011657,-0.04482656181417888,0.1417328693975759,-0.2922453958912956,-0.2525026680532809,-0.2609078138174638,-0.21789028020317577,0.10879045447938596,0.10854467607351796,-0.01167693767409512,-0.11730512301774529,-0.14041298657896795,-0.027734968868045006,0.1716156559644605,0.23178771309300344,0.08234039984482722,0.07809411539961296,0.012609154141186433,-0.023879190643966933,0.22647905907928703,-0.02675791139071863,-0.21213080690905717,-0.10740810585667315,-0.07944125864116346,0.08659617848152673,0.0141637544763888,0.0440797735224922,0.02174767034250937,-0.2427056315827606,-0.11070908475139152,0.009782539829760083,-0.212597239134212,-0.12712814083407628,0.05069566844474477,-0.11120697215878492,-0.021919363977633863,0.0686518005789133,0.07950280755165767,0.07358006050329208,0.21717728541189657,0.14089143732171622,-0.024904096889305283,0.27501805678170677,0.03198530580983799,0.07829403434376102,0.011949505404674479,0.119845840789858,0.15120129621208028,-0.004842453105542698,-0.010262627051170717,-0.062424669235877694,-0.1601606990872508,0.005840366392847091,0.06748883078490624,-0.087903341337303,0.02328890666327285,0.13724632315401022,0.15000922230335947,0.10776281861148185,0.20595121707711825,0.1100644190496122,0.1917419086795539,0.019565917948088388,0.02747342730635846,-0.12862085791552927,0.02235938657326181,0.10569354380744884,-0.20655098656415982,-0.1803157660039143,-0.03552757789358342,-0.026540363349046893,-0.1517533480670949,-0.1514852463780655,-0.15106696292075245,0.0373974901899771,-0.019312115437282576,-0.044723589686195726,-0.3122051769264091,-0.2697773807022149,0.0365608710121325,0.0223532520909304,0.10910067114617826,0.1641296554030667,0.11645733439855378,-0.002252802264086093,0.06645370678492415,0.0921167311886006,0.10320665364703654,0.04524184684025207,-0.06433670534135832],[-0.04284467777487426,0.02369049834889182,0.08652480798468376,-0.0061736871031793095,0.06285783581774927,0.023639375848265767,-0.056378340246797186,0.07652868028279164,-0.03718848988600603,0.09283346654340066,-0.0686428581158707,0.11529372577192479,0.16389445257136817,0.013887794659503034,0.13910837145040053,-0.09428169403730123,0.013892442988264616,0.0011273379113576707,0.07926908528158581,-0.037454300437280916,0.19083140548234245,-0.21612476992064356,0.16673791255354012,-0.14089674142402012,0.11975240609644182,0.19795029145726054,-0.011623805765124935,0.025781660781253227,-0.14724377941114702,0.05139083020008464,0.036086061305861346,0.143779599832104,-0.17106894129623262,0.14314794167434797,-0.241709077289162,0.38089753029793344,-0.07175432283956229,0.1012761825290762,0.17366482616365966,0.08830381238882337,-0.11610213498283734,0.05040888872745546,0.16365792574804316,-0.034034725931929743,0.16752271411702901,-0.15327436820276744,0.15153110204861595,-0.2483403510704983,0.2746325347048055,-0.07758280994284213,0.15448946702499208,0.1566463284750305,0.19071924159081,-0.12911620637424387,0.13633742689050032,0.02090342753438867,-0.0440073327822947,-0.15161663785089963,-0.09956544565549152,0.11585377504217316,0.028289922437404844,0.20995937195105302,-0.05281364078431926,0.14647639080848535,0.27020380983114217,0.030938489788704844,-0.1835674919795282,0.026944859565955798,-0.04056735065817105,-0.12222812109333375,-0.05512555702769221,-0.034362099902950166,-0.05966581936003341,0.04668330300751901,0.10117701528110185,-0.12318651369814755,0.10652718579604727,0.2566718057228357,0.22840347424657337,0.09362030996697193,0.09300740188415559,0.09176561584113922,-0.11386236569377167,0.06118576032474544,0.060673477169032826,-0.29102387567687615,0.14353093048578058,-0.07030656264356358,-0.038215886296830044,-0.00829163606845842,0.12138026788356684,0.2434932228957169,-0.07504062098559029,-0.016432170333068194,0.031135415536559977,-0.030407313251721045,-0.04919940263962619,0.10548603256733272,-0.15568625649862425,0.09055824618406454,0.023337766497485212,-0.08986610674558732,0.004975925709221505,0.1670586576118359,0.23147074433438639,-0.13561066695169674,0.059433732188451945,0.08968829659103254,0.06076075651946261,-0.0984206667488064,0.0708664636439034,0.12410930288134012,-0.15193842917210898,0.09972144558256382,-0.24118834577321133,-0.11191699999575902,0.25009481489900015,0.35580959613523894,-0.14974295963872183,0.052256585582726924,0.11746072628342666,-0.027059529604356,-0.00943736386524179,0.053833157466081305,0.027602611267193947,-0.234134818655287,0.03284203730559627,-0.03882413737926202,0.143337557444466,0.19587217020063877,0.2883946500322041,-0.039591523665026516,0.06475624166457647,0.20583392487183494,0.038136843936808246,-0.010828592902359708,0.0030376266954714763,0.09977889391647508,-0.040804440053872675,0.07855681576280604,-0.03059220062150903,-0.028787541502287173,-0.0003890941447931815,0.11845311062567775,-0.09472036953942987,-0.008550750180640596,0.17440855505488245,-0.0016017973933297055,-0.03830599741591053,-0.09844038398684686,0.01315246607026184,0.17722861978688878,0.06162389673146383,-0.09492896302434797,0.04378098420758194,0.033339629869325225,0.07742071860048778,-0.015856421890515466,-0.0412802870444337,-0.01896321632456038,-0.09923036323512227,0.013852669456428048,-0.004099216957284976,0.04213396456890969,0.03907101116435178,-0.055544715203259605,-0.07353223644367461,-0.03792026986620917,0.021545856677373895,-0.012289247613076204,0.04317950433356639,-0.011441110467766228,0.004919765745217885,0.007781625871373002,-0.03794225535960373,-0.06538991928488783,-0.009486176485002551,0.10078303958140669,0.020889462277279693,0.025102282299493767,0.05748740650093015,-0.0030308490874006552,0.11388693397072114,0.06308583624531128,-0.005896418362841058,-0.01849720324217822,-0.0002597431224709509,-0.0691547724004687,0.05534211044833649,-0.15605491215551837,0.03739328527454031,-0.19239061759460585,-0.021413132664315226,0.02814153144085979,0.10820716113513615,0.04775201417544466,0.14018537715512908,0.0013651423397684026,0.04125635106510351,-0.042191464362565476,-0.05233727426110881,0.049176199918058834,-0.2344210273300093,0.16302092572614232,-0.0033834204292961484,0.036277066250837715,0.06277294812207761,0.05450977690658568,0.03664111299732515,-0.06371847571774437,0.16022907857816862,0.1364898081335206,0.12690499416308423,0.007463560637299126,0.09834954240874991,-0.2431434218214664,0.04418554087997094,-0.12501475800147527,-0.025438972807369575,0.07515436553085245,0.03112729167971135,0.07300644331881823,-0.02108009390969543,0.14854732757955685,-0.019931164531524985,0.07390898437774338,0.0866995175060996,0.05364041124426575,-0.17452047856584318,0.3011431824274898,0.0533507818581503,0.012357643048960552,0.002175432093663731,0.05384464036136758,-0.10331037978458818,-0.009582801039130319,0.09156037608066717,-0.10342072175407128,-0.21870207939341246,0.08910512525086034,0.13222739055564064,-0.026685791447768014,0.1285265720250293,-0.13548745773033455,-0.06182186676863765,-0.02578750637126733,0.02267104834826221,0.12105795922031116,-0.09629346469833883,0.08074229699459658,0.11687963640822577,-0.03293788425365486,0.04527089414663429,-0.029928701868637515,0.009650454871950528,0.13026047215192932,-0.11342913928560043,-0.10123156097733352,-0.08597862491299753,0.01699533227376428,0.13804093413470553,-0.13098839118680466,0.13222454548691945,0.02005668886904669,0.011168057076905104,-0.02848860901869195,0.1002751891377585,-0.02388101138333045,0.026787068851495482,0.002742495868514139,0.04908095924728946,-0.05598940863742884,0.018206002401989278,0.03567879566112995,-0.1541669091122787,0.031487190800093275,0.034837141206586465,-0.05824962999202721,-0.0392479385747287,0.03352028954240081,0.040541990558341894,-0.14145767051868144,-0.00814494050549432,-0.08571955602706846,0.10143348712266294,0.17292650734207124,0.08162466324421891,-0.12112612717818269,0.1326426325938332,0.19349174581017647,-0.01951566699847113,0.08960983920839671,0.2055989989299032,0.05210326805441766,-0.06292241740827577,0.027655964067108435,0.06552625930859929,0.22070365373493686,0.19755763129057746,0.12099182056067873,-0.10969879074218052,0.060222419734901694,0.03196966174509959,-0.07257398552442537,-0.09269983070274149,-0.046675929424227984,0.07959754744819376,0.0638621020133725,0.10888925171940733,0.07980363629238443,0.006035252571707353,0.015504891149410272,0.07223257189947634,-0.024278244841947308,0.075322034922288,0.0763155871873595,0.009145261312540676,-0.10685924778827174,-0.005266825665370885,0.0726362657965258,-0.05373533873066114,-0.07018805873506753,0.06065505396881455,0.05003771204608376,-0.07134534523840615,0.10029050365142557,-0.04854858596555947,-0.05716435151837326,0.0002155252146021672,-0.02941586843028852,-0.07068934011745745,-0.06630220490888959,0.021885558362552033,0.037126530882119715,-0.06375890386735528,-0.07734196190879299,-0.05427545467497378,-0.012811770274490958,-0.030956315350942676,-0.007435952091234126,-0.037333320727493964,-0.040347749798849496,-0.014432218138529835,-0.002814925241037986,-0.02505817606001633,0.05060817210442414,-0.03422875816918586,0.06923443209766056,-0.031958386178417335,0.018465888158573505,0.08636174484648436,0.05215677988969011,0.022198262679867947,-0.05778936683331119,0.014686194043923968,-0.011039273211509316,-0.012658952502915863,-0.04839524385760902,-0.011613838812795676,0.02365499526497396,0.04089189159004548,-0.007826118121763802,0.03319757450435156,0.03161569256407196,0.05230675231689854,-0.07903022497662476,-0.03682804523637692,0.0399443889219972,0.010675940278074022,0.03714461162691243,-0.061937193925802575,-0.02715764291655248,-0.012818144556213742,0.0074476555761640835,0.025053359001606948,-0.014673655074571683,-0.019829972292939774,-0.009818019146145612,0.015879430872863944,0.0007005753981777608,0.02172539016192609,-0.056888328709115414,0.03255227738872132,0.009847945285477383,0.007636854175665744,-0.02243807597886645,-0.044931212605091635,0.026353878625389928,-0.03067916032424194,-0.012340885749459459,0.04899139586312837,-0.04528589782785273,-0.02185588400157497,-0.05491806115763517,-0.10799523573737943,-0.03833972084391969,-0.03278776366802733,0.0030757915980218105,0.006172795789412087,0.0107110601305558,-0.06507154311325658,0.04803532132023247,0.03640968837179486,0.048850453597604516,0.05103262484691569,-0.011910359347871956,-0.0048266648921922225,-0.05580428767622594,-0.0017779542921488583,-0.03741944626210188,-0.09142545225949514,0.0005599151414637076,-0.02131888769615417,0.007817950133619602,-0.030262118657830347,-0.0060041664207005074,0.019370300751039987,-0.02839073930333483,0.03329308049440469,-0.03359891424884029,0.0465949341820874,0.03731506514841214,0.017293142267355878,-0.004421645848801713,0.06982463627293611,-0.004659117671220134,0.013198150392778017,-0.0029117710073207543,-0.009527495759135257,-0.014917381866404114,0.005977182407340662,-0.04535209346647479,-0.06161603113092829,-0.007953534968434353,0.00940951823852768,-0.012692706735209511,-0.06242783055546556,0.022191982113764636,0.019532477684898155,0.00017414933482374565,-0.061018322695922216,0.05907591433164703,-0.03239628258804075,-0.06328838474445513,0.06790832184602348,-0.0499404900105417,-0.013009985128363826,0.0001853769162176942,-0.01224163384632237,-0.08504582895657166,-0.03728298606125985,-0.004345963662008977,0.024473069500423484,0.06131519622346716,0.04508592526737603,0.019195109943450056,-0.005208613547886984,-0.00615268869724116,0.00478164997518403,0.008058185228232657,0.047292784899288794,-0.07693019942454055,0.005830997744858304,0.020802613825086066,0.04184984180506736,0.02479507909712232,0.012944969512644938,0.019237921641178955,0.04938162297338577,0.010452450684151857,-0.01696804443623517,0.01169462131567603,0.023900742878406836,0.012018489517852256,0.03584513289549441,-0.05512380843741867,0.010494954419741307,-0.003463982052830882,-0.00591817003005033,0.04070498217830163,-0.018746986711673172,0.050552866573019294,-0.0016869620854471008,0.037403877815074246,-0.008927275884872616,-0.05458753584825205,0.007452054690077469,0.022099224517364298,0.01912964810751989,-0.048008258897195365,0.0016186385845099898,-0.03957201395225852,0.01155840495481212,0.033497702383535506,-0.0037790392966380133,-0.0192364972240141,0.014096140518272408,-0.04558826842025135,-0.006621778934339603,0.016204256591814174,0.007606192340778749,-0.03723184922673956,-0.003749400156303807,-0.007781555518824914,0.035566696593766794,-0.028701701505093522,0.027492558698349556,0.1858237157305666,0.16287463450101886,-0.04053736314445254,0.0643942487839987,-0.03365499812587518,-0.09263375799024193,0.12619150522412914,0.1492249582386677,0.03156783794835049,-0.08985926909813392,0.1925983861983088,0.14399462748968797,0.2212752430373943,0.08872614548453046,0.11759321364460049,-0.22442765686152882,-0.0781093198448732,-0.003941046700172678,-0.018985619246436547,0.11921094685595876,-0.36547180119422734,0.10536013831331749,0.08352889916450831,0.06868361407508562,0.21325277357032119,0.15044591511660488,0.0050238172138707534,-0.21985471959955002,0.08902920568589788,0.3355553578183675,0.22774179920735174,-0.05745913950414169,0.11125731761683874,-0.3541556953386786,-0.11600473462837649,-0.07831009867821183,-0.1381920293321055,0.07546796784793766,-0.04099136230118969,0.05119089190631379,-0.09353138411788858,0.17326047562060193,0.006857718632560393,-0.002438167995471058,-0.04306234072966018,0.13392280836712822,-0.2985811718450679,0.13616940320715873,0.10826162612162663,-0.04751704231313184,0.12826822913678937,0.009485366080293349,-0.17437625938426765,-0.1536278222284545,-0.02493166113295536,-0.054314601910496266,-0.2241039749841641,-0.01432913195350053,0.2071474834918617,-0.159940916892446,-0.06637843142259205,-0.3035044573227977,-0.24680849497509424,-0.031711453113976884,0.0059448228131315384,0.11720725143909015,-0.12172243820874128,0.10477259922315593,0.018624821797516653,-0.011534523785263108,0.06625249764934732,-0.061449956153156696,-0.3005668792224223,0.08691644412250947,-0.2715423318226746,-0.1396222307247515,-0.12848419915828901,0.18229898226791488,0.3697353351268994,-0.23173709992931368,0.08563096212372509,-0.038608581826450425,0.01986330640906572,0.11126555806488507,0.029388136094276575,0.023561455864165387,-0.07586579448806327,-0.1927983695748212,-0.03180231524686527,-0.12659240181187412,-0.059598114691325914,0.2787345752156262,-0.3786313652125565,-0.06743639602600443,0.040940523320542904,-0.12183437441929017,-0.24334359347326287,0.005220360099962774,0.1644654727766543,-0.15892010071005885,-0.0008116632292458698,-0.1354064873016749,0.015773591611195372,0.05688671225852201,0.17164535783770726,-0.22161378608899804,0.13889405758387163,0.34816083666356323,0.05087354207455283,-0.025492265958300227,0.40383048747965195,0.14923768869120485,-0.0372849706454833,-0.0495661250804942,-0.031547797593407576,0.313654706180905,0.3413869708005741,0.17717184627648663,-0.28721624883414554,0.07284403350132027,0.04513757924340538,-0.16334987310604115,-0.07137315868886385,0.010161034273317288,-0.037945486213548485,0.07419327367008463,0.09323236132192855,0.04249811525838757,0.04500440322976752,0.03569727144163906,0.15342173278211926,-0.0793601708783063,0.2500142132372359,0.25981230292134927,0.0911401799063343,-0.11534626232781196,-0.004697385520524469,-0.011719328572502278,-0.01877846712688193,-0.08421172082406825,0.027064642166501413,0.09504164984820855,-0.07602122803853628,0.11434491724621097,-0.048937373485642706,-0.06689401813296889,0.004480017742699084,-0.00782068293643989,-0.14258437525970136,-0.04844224454978188,0.16595251803937283,0.03771785139457014,-0.06472013535124702,-0.06916576994629872,-0.10084117209582624,-0.0565260110035924,-0.020633514391815926,0.006180753494468647,-0.05095224721239927,0.03338025650982652,-0.015901964795453204,0.0013966129301701085,0.006009838805663799,-0.055934172032850844,-0.04403944147099083,-0.035624904977603865,-0.04971420354142098,-0.028163874843272218,0.01698011478790495,-0.031294106722887925,-0.040476925431159216,-0.04174404708284104,-0.0414519709608323,-0.034434668449018796,0.008642582825839407,0.011326920305753569,-0.004805691706509471,-0.038361167360383336,0.021101737812058922,-0.00784728989014567,-0.0021707599748240626,-0.02392548912514818,0.034334131159487156,0.026628982372960307,0.020201047144959506,-0.005265729369115431,0.004400745038776735,-0.01323561418757418,-0.0420714198920493,-0.01532549555411657,0.04520087089586652,-0.07158554605128492,-0.002709061410118045,-0.0019094147033464823,0.049959311394970334,0.042495967312898564,-0.019953660394720168,-0.037138648739817315,-0.06279529478021512,0.031177394077322458,0.01368796189806614,0.03852365755692818,-0.11201126341800591,0.022925697100233174,-0.0682245170553887,0.01275196406600218,0.00981640359281718,-0.011714649640271246,-0.01727577509120491,-0.02899525997793695,-0.01769023383913493,0.05701727041500168,-0.02426840155741399,-0.032197387801618946,0.05196325295580188,-0.06478970165279842,-0.015817829646285968,-0.1179059724180574,-0.07926340269873557,-0.07635684465167858,0.010662107949041034,-0.02858120350176471,0.083405736842398,0.012671643095854696,0.00681933141582897,-0.0340981754339921,-0.015715135794695904,-0.019493827291748873,-0.04491008522008219,0.06993905723363353,-0.01388548347820043,0.09418469904245388,0.058721820274787465,0.059546710294603684,0.017758295369145572,0.05059043622262313,0.043864549480981195,-0.0282765314780698,-0.06485365120785408,0.012080667160568595,0.005991240892750931,0.0404286102016759,-0.03524221063416238,-0.03783995879032521,0.015421608817713232,-0.057921048807950244,-0.015094106296718835,0.050894681677137815,-0.03272592268508797,0.08859736442664747,-0.004979753787804727,-0.022900412312508407,0.04517509475850742,-0.03280115526904069,-0.009101428518556205,-0.06743921535967375,-0.06667042685914636,-0.0691356337661438,0.0038443792054208715,0.05127592036450368,-0.04503020769952703,-0.03305125860237472,-0.005117613135041598,-0.09043071629916948,0.03528300712038949,0.026191489046347646,0.028605939488743955,-0.0146240337577297,0.019447612455278443,-0.056636367445694544,-0.004500132150039146,-0.010423448812480545,0.03256653677892144,-0.03138739185933089,0.007812775603609897,-0.020616300888530698,0.004636318592385804,-0.06129336429562759,-0.07176062741457175,-0.058826176936218146,0.0010204345379515498,-0.02939696480542875,-0.0478025055587044,0.0029347092609552984,0.013785243635063466,0.010755219479399352,-0.09818250187822986,-0.05004084543295319,0.06658794311842993,0.04581036636284324,0.036829931719457545,0.04358141011322458,0.02832255234789577,-0.009768518772304344,0.029653223608890472,-0.08962163543776623,0.015129920125163758,0.04914816752337077,0.11290293672483243,-0.020880070122052646,-0.06069793314219343,0.004814996496211886,0.004765031034665893,0.0004892228910111827,-0.030920656658139888,0.024057752592007087,0.025749414299618738,0.04296757106266255,0.06643513471099127,0.023632311687279396,-0.0242001981367802,0.013725649209304625,0.007772921107368015,0.051841970641937095,0.010046694880916272,0.05265444120232053,0.037573142732033944,0.04449645334802714,0.0474828525541746,0.02755659625240666,-0.00717156961198255,-0.05861329438139881,0.05920364826725308,0.08321537325498125,0.0428606399550363,0.04309948941487773,0.023147386000225166,-0.000848102939453043,0.02621804153002542,0.003952800630557422,-0.05175702997214655,-0.031856601535971606,0.013304089652643228,-0.014700343525061884,-0.016516818487129596,0.015833845190066417,-0.06341998146667437,-0.0106894431662159,0.06400609490690391,0.05465618279339707,-0.0036955347854088806,-0.05413969450163505,-0.009804155421466502,-0.09721778438486321,0.005765479507513619,-0.0333791099163206,-0.042378417562957564,0.06651911289126554,0.026757504527908395,-0.039371303325107576,0.04027302869665908,-0.023712896921587973,-0.025903262933058203,-0.060007996748547426,0.046475190941336245,-0.07200272925306539,0.008407810642688177,-0.016202471748011777,-0.05868014943460318,-0.12951072676817782,0.04962841939712515,0.027960722083276143,0.009778140135751104,0.00962768942085057,0.08905620570798743,0.0038223679758058013,0.006987299281407847,0.005076179628369195,-0.07794036774948729,-0.09801307149971474,-0.05729892422903118,-0.15371892196602407,-0.09166879254493625,-0.03771533678145432,-0.08838927275337777,-0.010990490042245164,-0.03768392006464181,-0.010899229036896444,-0.11987078091997208,0.01707291235836685,-0.06947967975632023,-0.19248603230544536,-0.05590615432036281,0.006929075468716736,-0.20300909796086622,-0.148234300220567,-0.04412334620958724,-0.026866031803746606,-0.08339594751657928,-0.01771280842299544,0.028740343393316627,-0.00007716782258682358,-0.06304542695388876,-0.11451573375593617,-0.13627544068829278,-0.17187729830902665,-0.08430800882621371,-0.09801708341742849,-0.034158178143937526,-0.14656929281831677,0.0025238183340891697,-0.08905103394151634,0.06269569390028534,-0.07152839463098325,-0.02422542497549386,-0.04272053976864929,0.019302388044094163,-0.037626275774517395,-0.09697216831999854,-0.06991523632633778,-0.2209921472322922,-0.028665243680520792,-0.058898848379060405,0.00037022436839362425,-0.11281456038129511,-0.04841106665606891,-0.024146627742777583,-0.054848238311061843,-0.047693450674463705,-0.07316226014576507,-0.07261470640826777,0.012118428309269404,-0.03941272885074905,0.03430671557398603,-0.1265923633944323,-0.11476631639120431,-0.015109218339444598,0.016525949869433797,-0.06030408545747286,-0.04852331115859812,-0.05930496069314231,0.025440572254268982,-0.17055759557556674,-0.023607404039729888,-0.0952617847585308,-0.17672449570801707,-0.03920987902365448,-0.20973234646771974,-0.06562042956501662,-0.03272754995709479,-0.15088056204799744,-0.04034937832785415,-0.06844780041674303,-0.10359658261195631,0.007937284137561753,-0.07162974782555977,-0.06608666199963369,-0.10923389908001632,-0.04623538495445318,-0.1431794630161033,-0.05647023755458814,-0.09169857588399116,-0.07571140099876801,-0.10284047805980019,0.05405427232372575,-0.05602496441878439,-0.02181008340914462,-0.038818674172331785,-0.044225574707487045,-0.05885369578475823,-0.08414665007377117,0.00675997099852034,-0.12871584882745588,-0.17399606329557105,-0.08390040867325696,-0.0274104451426138,-0.05166752687701009,-0.008892207569210435,-0.061197348288044066,-0.03916167768685827,-0.06616107591824513,0.024815068406218194,-0.014321509228663822,-0.05311576430063137,-0.07379061641540863,0.021037500420772872,-0.08064803704058542,0.04677544897930847,0.021507414821502747,0.02392546394204792,-0.04431033436487962,-0.06318967564474158,0.030328543603893626,0.0051245889060847665,-0.07173152993279164,-0.007517575109457609,-0.0017008887391697855,0.06733703882358354,-0.0017740701786588423,-0.02717826459326625,0.030855906089768622,-0.013516548454799564,-0.004607824887958559,-0.018368016781549253,0.006189968792083755,-0.06690670877659158,-0.022152322424164962,0.07666913791818726,0.00648486134913039,-0.029302494064321984,0.016239907220984316,0.014814405774559839,-0.06935890466617459,-0.020726090866121826,-0.031047057628477266,-0.010821266289998284,-0.016926349122994254,-0.010842445068150621,-0.04108426186096932,-0.0002540755565805139,-0.015368091574925365,-0.01564105121431794,-0.05392027694892518,0.016064636124118514,0.02077945360917638,0.015250002551386308,-0.00633850815522596,0.005754393964784764,0.0008349917357312667,0.0212933246149651,-0.04445612053120332,0.009547509044484166,-0.02250687452395096,-0.020318460283047415,0.04358264768879499,-0.008537518614097486,-0.05168494360505283,0.0022585760794650747,-0.017800343554244643,0.022662810486761612,0.03716887259124761,0.012591489916175142,-0.036149586343431746,0.026183938437068007,0.045680108185929665,-0.022895221764295225,-0.031355026395175435,-0.05904617509815197,-0.015526161631656132,-0.012860328170166175,-0.01717085329916798,-0.026728356560768076,-0.007485527834385218,0.0422282748740008,0.0034041791253007062,-0.03856589358327037,-0.037206577585784,0.0536536842634126,-0.04454778699558312,-0.05368138708892141,-0.03151994278256897,-0.0024610126348878158,0.03306271695834073,0.03390377924895899,-0.0201654250195603,0.03780658238389126,-0.06879944602721506,-0.012306267668734569,0.008616556143870193,0.005936531581785201,-0.010744080747695969,-0.07011271796747862,-0.04592831995591683,-0.05521561729772998,-0.01885303379585784,-0.003098692905984568,0.022974723705636267,0.026215955694220375,0.006123287747101164,-0.013631778590048901,0.023971877847672497,-0.004001764782631033,0.00782645822916655,-0.02512903222505046,0.030620647997330764,0.01738067769688133,0.015005236242097444,-0.061988106246258126,-0.04976739325010001,0.017168369142123902,-0.05734711934138006,-0.005855105569283696,0.014342629679326199,-0.040762037507045885,0.04713854083156935,-0.0585871067577863,0.01289699081752373,0.0251640759702948,-0.002538110904815859,-0.01080682027673272,-0.057672714106062076,0.010736561082781083,0.03827563365116548,-0.024983160648253434,0.025121598681385,-0.021980377187675322,-0.007248756047746868,-0.015049072038149446,-0.028724036379692106,0.01875848851806294,-0.014439704451239542,-0.06039113500106844,-0.05856929542072046,-0.05176035064089708,0.00633993366078197,-0.0023077773156927545,-0.013775188859174763,-0.04901689679336846,-0.031433599678606185,-0.07079328804268156,-0.01350469710069861,-0.06264947226747594,-0.031066000602331185,-0.03931008859983547,-0.02880657921200465,-0.0527399608342459,-0.009703973413918826,-0.0471765262112408,-0.03568847974144407,-0.06427514559084233,-0.007073350134948447,-0.010631237181011215,-0.04796894567204616,-0.03309013171952534,0.01984227965093,-0.0699485119553677,0.012542926046531943,-0.011235997435387902,-0.06267392566508198,-0.015185770228163285,-0.014332673058628067,-0.008770133734481841,-0.005291186572362519,-0.06841724205282929,-0.0500369031308573,-0.022282124982146094,-0.011252737265438077,0.0026996308583421514,0.020877902410946253,0.01680123288421757,0.01487637444323642,-0.003244848383487927,0.013471197959693936,-0.015239927218621217,0.022947475266691046,-0.016903022313384204,-0.004903483174820813,-0.027808113538491403,0.014392848046093039,0.013849678525783754,0.051826468879947754,0.00272470114180947,0.007050020829655736,0.03568471285148932,-0.0310579426051898,0.003523332899901078,-0.04701494287828861,-0.013951995476060118,-0.04109864221381917,-0.018662167413898728,-0.041276487126319254,0.04370551816422575,0.05842505430499237,-0.042498053958281365,-0.012414526370528721,0.023895116217176524,-0.03179534715793887,0.027709611637098473,0.037041209189911474,-0.017143107435726054,-0.034728640022016935,0.05479339887448092,0.029607473730969478,-0.034081246648454154,-0.029793077513822923,-0.031110029013179902,0.10039078213736254,0.14167832509280104,0.08014588340840935,0.1578335734773538,0.27498020354900027,-0.0028886991061559033,-0.06159765757722216,-0.017767074883029838,0.14773933913770446,0.008995201417627283,0.07306913952485403,0.1959887286950619,0.2298712364547439,0.08224034776914071,0.14525940447608068,0.08842390824410153,-0.052291081074411334,-0.04953652864389835,-0.1564514962153155,0.07033822438466056,0.1031042927070751,-0.10622494196965854,-0.04746689560738255,-0.005999314806407883,0.105223562869543,0.1742927457982657,0.03437135433540902,0.048263031047147555,-0.17351776823023543,0.06703371299432102,0.2996391937793547,-0.021356538301050262,-0.011097213022836266,-0.028920438152572513,-0.012621124672600058,0.24556158334649672,-0.03967301674516646,-0.0449677887737588,0.22379220993374718,0.057418586278940593,-0.12601134887743468,-0.16815377960568004,0.1099801003021867,0.08521141336760633,-0.049353603959223336,0.04999859654526154,0.11002371712069467,-0.028492924160852445,0.13192500671717577,0.10493721743858156,0.15933539223047172,0.09631235335924343,-0.027242184512760954,0.07294072700301443,0.0351726762032931,0.07535602252866866,0.10753141729069096,-0.10545016594331359,0.1457766191428346,0.026592324043310016,-0.05226532831629491,0.055925622861220596,-0.15753373634151907,-0.1808717765015372,0.1908215510361926,-0.15376330440385616,-0.0731443414104867,-0.1176111042801525,-0.02333975411254485,0.021945080415404314,-0.07966217790042086,-0.00037949507003845695,-0.06741903386202905,-0.1541740761483849,0.05776841990365312,-0.09278556160889391,0.028878167450310653,0.2064985836237278,0.061248133120077965,0.2189426487965202,0.045515697056668775,0.08341493193759825,0.17008074203883283,0.1229791730312255,0.003254158644533365,0.0910264620960442,-0.13569525200379304,-0.14050554310090854,-0.186102668403402,-0.24351180590603994,0.030889214033557606,-0.01852358714970509,-0.0926383721774668,-0.2488771394362675,-0.06960234591573515,-0.06170201644706469,-0.05139272248251286,-0.035187339639675935,-0.0020545963022015226,-0.05726192385043434,-0.16431118402582212,0.07027645862914988,0.07571542166392825,-0.03220763581306604,0.010775041752165369,-0.003353740122804988,-0.1485115425643224,0.14239697253309413,0.10231027487733334,-0.10198541118945388,0.10303853725785525,0.12716909940403034,-0.011557830096578972,0.01779676055302558,-0.17781038740087882,-0.11709378468504025,0.08892156652057669,0.08557283671617417,-0.13065689724965202,-0.12943539027099088,0.08310064810721066,0.011667759263571524,-0.0007967361053724938,0.19076882663163477,0.017366548141175015,-0.10448325856025352,-0.02234334515962544,-0.0018246543563208348,0.07258561066750775,0.22942823783246924,0.21283973490702068,0.005793696688242463,-0.032082763782078465,0.19263159172040517,0.09197203749117384,-0.13227686796780314,-0.10212990988249901,-0.024004226056244197,-0.17286408700403072,-0.08725671950350342,-0.13275899645886718,-0.09575561736605143,0.045971664179978895,0.10211386918185472,-0.063724004890767,0.08670511445928,0.036493433055729496,0.09825803523711184,-0.12754658152011344,-0.10354293660514209,0.18036217710613064,0.1685284906767419,0.06799922996590888,-0.07111265017015858,-0.015111608202072349,-0.05596794103189265,0.10358044971950155,0.1119832727647331,0.010233070020163125,0.049643870379546984,0.035574710243677594,-0.008795544450217031,0.022900620458564815,0.019004619770613303,0.01621312503093566,-0.055016362701346314,-0.04131989383568847,-0.08663899929519088,-0.02188862895048086],[0.017425593117523137,0.015107139092881125,-0.017871691775138872,0.0574419138821683,0.06713124004303439,-0.08968433446222657,0.06638579066983397,0.009504555042550617,-0.11950764170833976,0.021008918785321972,0.021348077491672106,-0.06656651941587731,-0.05230901149195715,0.04887613166619193,-0.10226243744598376,0.0602026345791481,0.08980302893135299,0.00857341747397803,-0.0044059362590731415,0.06103684312042804,0.09805991439046206,0.03165219629977401,-0.14626634689821497,0.1389168844966407,-0.09016534647507427,-0.158198600801233,-0.05639243208998216,-0.04678954340269543,0.15757709978635834,-0.005959460240463002,0.09868495696195503,0.15256379482375645,-0.09647447885224518,-0.011399381429260592,0.020413584120829702,-0.23446546319291878,-0.00659786470850827,-0.11528926731512139,-0.15981167382139044,-0.01324537268965107,-0.06596878510068516,-0.010342000920833283,-0.07884669059523527,-0.058124114401947403,-0.09295670054706726,-0.013764147728293928,-0.048608461272808996,-0.009340621124729699,-0.07563727461570875,0.042599975383931746,-0.06318211474802823,-0.00964790513551174,-0.17714498337311554,0.032380105985243775,-0.059982554170389876,-0.11098799074092063,-0.096470355214963,-0.041500250830967936,0.17613471486031332,-0.171194757458487,0.06760564213508274,0.09804367091049626,-0.03384985395417202,0.07019807429901279,-0.10004564516644936,-0.0506196834790268,-0.18889488933147697,-0.0009784793681428855,-0.017376606149702996,0.2973327495855179,0.02857539617941245,0.19544251223431064,0.0021827395824044523,0.0002058615670938862,0.04253508650849663,-0.14698975524188546,0.14026592566307353,-0.2507945469994796,-0.04132452361915391,-0.09246331615307991,0.05012743304656871,-0.22001353628616832,-0.15046123598879638,-0.021502341382207576,0.008871679629495277,0.19626279970017643,-0.18564331010443125,0.059671556009863375,-0.0023602519984107814,0.1699380725021836,-0.04124923371070723,-0.08865364374195328,-0.04471359088008936,-0.0007543076135201347,-0.06965579964187077,-0.10042173214362123,0.09497164171405795,0.04025048090363432,-0.10484545950984923,-0.1647421880876408,0.04875672831179369,0.10093858207231304,-0.16237739882935365,-0.15250442223367078,-0.05630889469236746,-0.12331578725133435,0.07683774311831212,0.07636671375889673,-0.26279329835047727,0.13080138212969897,0.03677867730826574,-0.1428929644437736,-0.06236000012512532,-0.017513933270946996,0.12260402205853016,-0.04133427717436356,-0.3175515517144661,-0.07674906333975338,-0.13253579860660356,0.07355053118558186,-0.06531228113239929,-0.06808805932236858,0.004312817958060869,-0.028177819311294,-0.08225699724231257,-0.0005404853508236668,-0.04621995865110035,-0.10097733396884874,-0.18486029768742576,-0.15108491050321415,-0.05094379282826999,0.011553045652811407,-0.0371488703201666,-0.16312871218016528,0.08196133075224511,-0.25638450243443356,-0.04492249787836713,0.00016682416126680347,-0.06584657678622281,-0.10005122305956403,-0.10885820551262733,-0.13073248825253264,0.0039592263930582315,0.022447078865698283,0.1272932587280122,-0.1392062635079215,0.029648177647284717,0.08151004358529795,0.08500576020860291,0.05655936682592388,0.18652131183745274,-0.08286023485777136,0.05167798342822378,-0.039498971305986404,-0.0864737716415526,0.12761526585823776,0.07167240521890195,0.08055288556129338,-0.0016791931137587757,0.05361747048842703,-0.04685762649706506,-0.02751774967291768,-0.0007874703722818313,0.031010112858134424,-0.019529718216045447,0.0745423226491829,-0.0623016001718485,-0.023817306723154522,0.03768754506791028,-0.032932906317692634,-0.013059224671556664,-0.02205989311043178,0.01876607102769975,-0.0037126456743705677,0.033877496275191724,0.017405154833595077,-0.003618040210592821,0.14111758840181088,0.1392397590817421,0.033590682907484015,0.03419391943033012,-0.021860555310832048,-0.028061860144247216,0.007833804289529374,0.08053487921110652,-0.016465346555276515,-0.018131813527409172,0.09354763888824975,0.003780962934113639,-0.05188852231828812,-0.060330343185676136,0.10837318366818982,0.10581305758342978,-0.14796418490653046,0.009692471114022857,-0.02876863866897653,-0.006611351135121955,0.048761042161751776,-0.0003450223035198712,0.07741014794773242,-0.11502303859703131,-0.054690165631461986,0.23945065876843813,-0.005131478093879153,-0.00032615019054615344,0.07847151692861751,-0.1386603725826297,-0.19286604193160453,-0.007518209817157018,0.05985626752267001,0.004014233877993754,-0.060295645611055784,0.015778720459225275,0.10581591702943277,-0.2868697499268287,0.10311395618660477,0.00413462662904812,0.0099619102115216,-0.022065059036387728,-0.039507348593286966,-0.20009275142948452,-0.04700083570346722,-0.0953677984245944,-0.0028933005064931704,-0.08886971574189986,-0.18097835935876855,-0.0623376954392957,-0.010216888036609868,0.11433968590675347,-0.08034640402954407,0.012771336264992928,0.015127265053357842,0.007572736283580849,-0.10404114335897754,0.05993192175366107,0.06930939202849201,0.0693310235645075,0.06116238433623416,0.0858480500854966,-0.15142181659032772,-0.0881375285305421,-0.02883890286666525,0.16208310624588,0.19146015632418953,0.24219427082963485,0.10014329462708611,-0.07349338399131321,-0.09535906812173449,-0.004862388586014584,-0.06548956425756249,-0.09934765946346855,0.06587462996313262,0.01926712081559263,-0.12783857487188433,-0.04847896579307228,-0.043041288495568736,-0.000323572770909659,0.2997728283436243,0.15508866213487776,-0.12274610263590391,-0.01413057438000886,-0.11726927668414595,-0.14874698137012976,-0.17986995296829233,0.006428533152396616,0.08457540573482175,-0.12364018209055905,-0.10466311964780887,-0.013384919867275883,0.05748158387177469,0.028891451363234226,-0.030878869559303787,0.05514999912222468,-0.0042781596816673885,-0.023104346365812985,-0.039413364346276625,-0.1493866366974893,0.026746869610864015,-0.032140464958102244,0.0026348717159596735,-0.125798685088959,-0.05626067005979489,0.11757507308780744,0.218162683187007,-0.17617786720422612,0.07322793176767356,-0.0324212841884034,0.11703345529034176,-0.006530429253855077,-0.05551188556529568,0.007300184157174635,-0.03858387612658379,-0.029452799188439738,0.07344309601782928,-0.04170418021724786,-0.026742164534698826,-0.026704850560578672,-0.15103291395509552,0.1049478583475578,-0.1073700268589368,-0.12188087926536592,-0.15512503620771206,-0.02237058602128876,-0.09097021106316401,-0.13386863100467286,-0.05581354890373676,-0.0977584167995788,0.017086269719620864,-0.07521626972303515,-0.02703618579464677,0.039624421100222365,0.00492074084899082,0.0013343713387669934,-0.05074323244158565,-0.08006669460027414,0.005788584240128392,0.06684945507148779,0.08149206767678913,0.0949486802598808,0.0476222584313813,0.013005413660099384,0.041596491492878127,-0.06339146982027147,-0.009260619480804172,0.12383629949245745,0.14510484438506635,-0.015071458499025844,0.06533824825557372,0.061095947628258374,-0.031115288952724567,0.051213977324920966,0.13080905050684907,-0.04877715575198895,0.06345148797508293,0.14144750618475915,0.001994389824552293,0.0248648974199238,0.0751952860373109,0.04359830753321472,0.029628449693565474,-0.02547180454328897,0.08409801253777309,-0.0281494409589601,0.0376228993573087,-0.0094500863956663,-0.030353895071450795,-0.088568891067365,0.056695276929174934,-0.02428919049821786,0.009266920491820074,-0.05746403445540329,0.02358924025565912,0.031275912123772016,-0.040256959184985744,-0.05778991917403697,0.0008984945220596069,-0.03086216942131567,-0.015987405899888396,-0.03768498484007412,-0.07527763811033136,-0.003810363435716292,-0.02223958933165589,0.01211227000717165,-0.030034270913916906,0.01861268574669375,0.008602760485426257,-0.012365656197293278,-0.09379580377991914,-0.011923092540114718,-0.006747169704737681,-0.020791889926441903,-0.023349332305000962,0.003483234694619541,-0.0533190857113217,0.03204785211608551,0.04999241867410716,0.003216142855541107,0.02756959211557241,0.0035948396714690287,-0.04728036351260115,-0.0006818428003111875,-0.0016128070813765994,-0.047281301644276866,-0.014749543167336312,0.0154367075691737,-0.0030806807607815082,0.02501300787756398,0.035237180678564566,0.07238879281445264,-0.018481109720973755,-0.004410162003625181,-0.02355556063986869,0.004452539512676698,0.029165688626139243,0.07047739113791479,-0.07205828210963333,0.0031934428373553234,0.04750125587154941,-0.002543859483886848,0.023106110196696443,-0.04283258332348588,0.001452024988199801,-0.025216240026104272,0.0293231705790908,0.02635189194553963,-0.031915636709210715,-0.05024546926162032,-0.034406193020570734,-0.053940613094587274,-0.06340951117924208,0.021339851796319736,-0.03172000225659203,-0.04470482675044608,0.07104857548341101,0.03563664246656598,-0.04841254189040081,-0.03827874192502454,-0.033239636980414225,0.007255280942844632,-0.0023889236883559753,-0.020174648110461105,0.01422717345204207,-0.054567136942531375,-0.0045826389393901976,0.014432071358456044,0.044335922005917724,0.07327662739143158,0.02820996271016061,0.06022915992194281,-0.027390008337080093,0.002177071478331635,0.021934000848928985,-0.05310391052895862,0.07242617011969776,0.04543004472635025,-0.006084237835303365,0.01797655257665193,0.012845021580113584,-0.00954128105209346,0.013425463932610333,0.0068537454482974465,0.026395296768713886,0.03935665733471334,0.03552916332871876,0.0005798572509596319,0.04541624859456963,-0.02338061075280518,0.026964789853829243,-0.005936286178082434,-0.017625196526192977,0.0523555921277621,-0.033833289919309395,0.021348922856554806,0.026697936391691015,-0.04143951133146786,-0.00790469769212073,0.0348419629178815,0.026349402007741288,-0.06369776791061027,0.02014240546614409,-0.025792514546119766,-0.027662505485632368,0.009364131443419762,0.007181164836413756,-0.03147770185410479,0.001025366083774516,0.0689362347552547,-0.012613348098011642,0.0430497686574046,-0.03919085896988207,0.030526172407825652,-0.07002390018003248,-0.0004043607069312428,0.02486054263128082,0.059699100844627255,0.0003590134139333382,-0.046449314657349625,-0.0022165575556473745,-0.032662780552982475,-0.042036527479357426,0.012168089048481493,-0.014699369217057253,-0.0133746762996466,-0.000045842050212592696,0.03998401003781272,0.01047455506705173,-0.023241542095237534,-0.016652965091005306,0.04749421206833823,0.04670964035831875,-0.01691950849355748,0.029301109885539727,-0.0008674748704571703,0.05141553906730466,0.025527440871860305,-0.02963186020136418,-0.013981064484701166,-0.03793554461165948,-0.018252118711651118,-0.010408946814577979,-0.03582247321794614,0.001283942345057905,-0.01418143560928939,-0.01860796891502412,-0.02403808601566576,0.010417679603116101,0.03221424868371672,0.010868787982194202,-0.08189069699194237,0.18121224539885147,0.12147064523970681,-0.2917960709040033,0.06758305555217625,-0.08543884834887895,-0.13747085979882392,-0.07248136201359516,0.021005878248284114,-0.06331250935584386,-0.23259005767839883,-0.08563329364028374,-0.11190257240162671,-0.07725387299359554,0.21415086247858559,0.04002743170531105,-0.17678863192161828,-0.06695377294962808,-0.13010580810157413,-0.06055050649858025,-0.20469384930540033,0.005839691338507828,-0.013730754204419451,-0.22173193062320176,-0.3029214965041041,-0.15824113320743818,0.11525608424226297,0.14333782740148207,-0.17479174288543678,0.16134623509755608,-0.012718932902884257,-0.37240376649246054,0.25708547130088943,0.1040006912030379,-0.09408944357477059,0.08093894433452388,-0.26360046617125865,-0.24992637787674118,-0.14899193938525263,0.000415535324587043,-0.06505865979354014,-0.1902999567300411,-0.11623704440929528,0.22580588380222558,-0.06620752089680142,0.07263724851670014,-0.22941984294152917,0.1392279162117801,0.07954470499234756,-0.09792490375477977,-0.07016625254650492,-0.0051009836449522495,0.048605808136957016,0.1485661306589601,0.021893297212515508,0.2618739905805395,-0.10706157418390455,0.027036580170175063,0.10897168056526266,0.213751424326769,0.11099880122716689,0.128934061097637,0.16177083685780128,0.010213106915955453,-0.02333018021432254,-0.03431093225669575,0.15413442150515927,-0.13936976067059795,0.12193114228222227,-0.03196228485515837,-0.07772465738686464,0.18143778505700323,0.07974541653117602,0.02598806383584945,0.39787871148285947,0.15973597947178148,-0.19183979050427394,-0.11495084624620823,0.052859819570224155,-0.0947592406969893,-0.08865362692839342,0.03827361139082624,0.2520546923416982,0.13073823802901202,-0.16798316589813103,-0.030902460494414155,0.01766259616046655,0.14425474281427866,0.08353034557268353,-0.09238251225481844,-0.10396187328147792,-0.1321739721160426,-0.017387204942568284,-0.05639310866943956,0.0025468146659993435,0.00562004326829252,0.021983671911693436,-0.26539769769978283,0.17968281975336675,0.16527533606180245,0.24548212670625683,-0.30762058235834283,-0.020122096969636277,-0.028628945181313835,0.12615213403511283,0.2312510152530935,-0.16048046323280585,0.09712973014461367,0.08236628721519015,-0.06091567839962334,0.025729820729001106,-0.20496653211870935,-0.016729500299558896,0.08480226683436162,-0.3024416226168542,0.04674255931053446,-0.24011976754235603,-0.062492633538367315,-0.055224327309065326,-0.19885572636564922,-0.12836915380354577,-0.24970225285617062,-0.1585503020477952,-0.15254941525632518,-0.13047493692443676,-0.21010159338791745,-0.06507365709530258,-0.19209052829215362,-0.10013473675229241,-0.06546672973621441,-0.02869134436954175,-0.23125524342788126,0.0286317845305038,0.10474225811736608,0.0009343705982392294,0.23778016024929927,0.12680843308684164,-0.044627605739803115,0.06096562876747062,-0.08965699977273556,-0.07806850313086194,0.16751866705637325,0.22281631534267948,-0.07171335700224642,0.05353801963794545,0.19106599084545703,-0.010225246892611893,-0.02067348174483729,0.15820988505261047,0.04895114355629764,0.04329601207920627,0.09481757820981578,-0.0523550653409001,0.08585434519562858,0.17854969284108974,0.2939537435880176,0.16558177371887833,-0.09005032290820875,-0.1348983554024335,-0.15052255343628904,-0.05404446124578694,-0.07275244489212351,-0.08275689100898358,-0.09999106630763643,-0.08382373217423604,-0.043372756651874365,0.040703363618669074,-0.032932161551780324,-0.008777301446431301,0.04846906446649858,0.015694901127625213,-0.06138157817996386,-0.02820992776770359,0.0405080430088117,0.01266261437807598,-0.05339479054474552,0.0009889959822017179,-0.008680573925255772,-0.009641973703156193,0.05129081655446372,0.00793348613034072,-0.045891676382184506,-0.020317588011749016,-0.025593209382249158,-0.03001249935099659,-0.07271048472642151,0.0741064185888027,-0.018218111958080967,0.04642663577710699,0.04301766180632955,0.02797190663057443,-0.014592857380982142,-0.07134787940159247,0.04154649205548869,0.0029141760405012075,-0.03375589256368663,0.06603079066604355,0.018931995396725222,0.009979601182766766,-0.04119862783034971,0.022430228682835757,0.032762900700425264,-0.08941350186115574,-0.06003734915297099,-0.03965811237286428,-0.036991613089863165,-0.051138088727656836,0.0011730754983568991,-0.009828093706033344,-0.028449414587986426,0.046405269178504065,0.029618847100221116,-0.01342187945847061,-0.008241623404392622,0.0444462045809499,0.04795394739532166,0.00917886385266771,0.09459994593873362,-0.03217877456427816,-0.004779190221926376,-0.018877459968321648,-0.036435031483231586,-0.024113710858390234,-0.03478329103933457,-0.10621610045092503,-0.042602074975906845,-0.0562971908037453,0.02625507078625127,-0.10274570996052126,-0.0341481704766375,0.01734482384048233,0.03588073614102446,-0.023456472976795786,0.03797053640245264,0.027943910196014696,0.03828334750527179,0.06854255670541048,0.02838758034779948,-0.00020656356609440255,-0.00998280128802781,0.05033108377450948,0.03284221027671353,0.010125345917154338,0.13886056664286953,-0.0531502948290281,0.001457124530912284,-0.01800453110328977,-0.041230934532004895,-0.06205260214465008,0.007796091280730123,0.12165629489045478,-0.03157178033019923,-0.12007055289913293,-0.04321774218866973,0.11500835493808663,0.04250960825517641,0.11848741476247587,0.03603078082749559,0.05671761128192907,-0.0319468536590198,0.05275272648637051,0.003503620260067351,-0.024245907108643345,0.003111342649738361,-0.00975646803887485,-0.0434450491026549,0.020211782054641848,-0.010293569417014535,0.06433731340046032,0.012310654567151387,0.013334788447624766,-0.023100802386071474,0.05102931032380034,-0.033473025600504,0.017819946514450607,-0.0530279398053381,0.05850674085798045,-0.02840806937124236,0.0023568289836838215,-0.05282641526336033,-0.0253904966869575,0.09165830031618459,0.015283496952862618,-0.07325340881525051,0.07921460970816356,0.036646687476542046,0.07542402038978943,0.037275959826711,-0.056304505482701536,0.009290310028006152,-0.03078927466209706,-0.03430129452647673,0.013680936063353777,-0.010330346730798119,0.08365968911915976,0.004082985610192989,-0.015020674795929539,0.029580267647199503,-0.0403050448078833,-0.022553121198320553,-0.00903920731752096,0.00934888068234453,-0.04844372380604947,-0.055252693946783935,-0.011796963889937728,-0.03353028103382313,-0.020278014247140506,-0.06878170828405884,0.022370450978594063,0.030675248089937424,0.015358202396847149,-0.03440648515917244,-0.03481199134316907,-0.035175275319847925,0.005208206988716472,-0.055745988596642426,-0.027722553652343346,0.002905806110700815,-0.04154765225588187,0.0417399328840481,0.04639051028047535,0.015644760729177597,-0.04581690193137525,0.010359636834877468,0.019050658064003738,-0.038586695199782736,0.05459630764030263,0.054885645948121925,0.07131791394055412,0.027117436983172215,0.0747193496768056,0.004912335844970397,-0.006974083249102308,0.07857138234121518,0.022240859800250264,0.040119794299732385,0.06268455353627472,-0.051055181947976716,-0.004103949012887378,-0.029203104948717726,0.12813021336081185,0.0030966140636338865,0.004317096007054658,0.08188476924521312,0.06899008730000179,0.036395621227706995,-0.0922714796023655,-0.006806701868506054,-0.060573470693766285,-0.06182070732617132,-0.02306931119910528,0.00981463709150676,0.024174443816717968,0.03043515923031688,0.04063647772673921,0.14421790649804772,0.008339172066283129,0.05503652532870323,0.05239420448233799,0.00771790829770009,-0.042414696277511044,-0.07389197737478172,-0.015277479741512279,0.01331692940609856,-0.045682937981013916,0.02843117774712684,0.04913953728317049,0.0113800151732925,0.049911951836711536,0.08442982579850623,0.0730560589633732,0.1421883438509337,0.05024883211516825,0.06749688332291624,0.060120063816606395,0.06464768966499664,0.028997861686059393,0.08605433410639189,0.02499799205758305,0.10185143563010511,0.11667890082057819,-0.04970798252891192,0.03820237318651813,0.08811261966986496,0.025566051324280976,0.13521540632430842,0.16171450276957308,0.13393725041122426,0.08739051203938564,0.04955463239540012,0.03345727891516222,0.04026792033191442,0.03582291743268036,0.10161234646357212,0.10691961132730311,0.04200309196990319,0.028401934554048498,0.08654568363316473,0.031988142798586854,0.07434453066489992,0.0792722832669155,-0.03700390749463695,0.0527820397643757,0.009253570890323576,0.024787718019292432,0.035804892199728426,0.010860521694109414,0.031068901257260872,0.08046475915575226,0.10708016166911884,0.03426016751751012,0.11930479606913386,0.02648116050981012,0.1471690768097306,0.03237114391786128,-0.010446892745302642,0.04834178169496749,-0.03125831659539104,0.0977690049012013,0.007327955094162728,-0.06708219706010211,-0.04909642828442308,-0.05265088696472055,0.01776390593424262,0.030288649020613602,0.09247971865826689,-0.06054412584471531,0.15704624541840656,0.033621938733179285,0.026286618970905666,0.01617041840971552,0.08815070345547568,-0.01858481815688905,0.12582636361947952,0.06828780815903936,0.006898817872328902,0.06884746609019342,0.07456815470411686,0.0017216927267149416,0.0523206428930656,-0.03925189305607119,-0.013515205203392531,-0.05372598508630455,0.02780734763515436,0.023687419290549495,-0.07552300343933388,0.024628927327679496,0.01287283938002913,0.03553154538049534,0.012899150593255002,0.07577416404092899,-0.04760617384074818,0.045181913439774485,-0.03945205548591979,0.0026703447750444686,0.010349144511106688,-0.013737953893853646,-0.09896834309961454,0.06138190588623771,-0.04192207334319573,0.020814695439312445,0.0012323107636003534,0.015397440187850572,0.13391419714238284,-0.04631437320348225,-0.04157656758425456,0.011609148552026948,-0.037174773338082036,0.0905373985613087,0.010113845576120934,-0.010074229541378687,-0.02946728676238026,0.09928246762117601,0.02159655037416526,0.05869196179451219,-0.0056781126809220175,0.061385600984800774,-0.03042205053191468,0.03196288469452885,0.057062343663828174,0.030998088035549293,-0.04286587778068596,-0.058741453321762595,0.008716793129844904,-0.05520856805298732,-0.07555681327632942,-0.037200192812816925,-0.03400739067671274,-0.004579808217524026,-0.04618046289642405,0.008119689934910006,0.015149350292286284,0.014134897214434556,0.0057323474298083925,-0.04834795444471431,0.04374074883517254,-0.01376656207883234,-0.06321005606249033,0.005250666189258942,0.03901133592004528,-0.033727025617372355,0.07608286107759173,0.007373094067150886,0.019379539961927126,-0.014690769496173612,-0.02898607393426313,0.03493594942562486,0.006192558999426623,0.006984801073732316,-0.0031133153186189734,-0.03585070812212169,-0.001853668057064725,-0.015569015932878792,-0.032934603981863424,-0.018013830263607655,-0.03703316657822358,-0.011325345720988837,0.017491632279057808,0.011047965873832883,0.040609011887526525,0.04507876550969449,-0.022181912961645765,0.03256275243777561,-0.024910893469359874,0.0011777997589849944,-0.01029630999494705,0.016449544852842543,-0.06542488897280271,0.04108643762983979,-0.04601201391548928,0.012379715652233829,0.036604654207757256,-0.022715335772511742,-0.025374864987257224,0.03731898214464319,0.0042972533120564,0.044422477408366544,0.03804222733208192,-0.010581679323882903,0.035920064938748546,-0.03182016029911429,0.007336926908417649,0.019566917056382103,-0.017137371265113985,0.004315990872382811,-0.03637983298250221,-0.004680992367237053,-0.03725297021409136,0.010387005927252968,-0.03138373582491217,0.04423169433286979,-0.005564452596975797,0.06617602090119325,0.05069290829596064,0.08249555319851677,0.025842620207492836,0.024326287588980738,0.023885763623500245,0.007130006417809608,0.020460315576977035,0.03687219418226038,-0.035292891041079764,0.020398268522607246,-0.008261038458599136,0.062954926030058,-0.01694672523504922,-0.06633521608497417,-0.049750975818148184,-0.04862090789593216,-0.02233742390983876,0.00562313682601235,-0.012747884843995978,-0.01992751445130355,0.03013768719490001,0.04440975553552365,-0.007681315701585424,-0.0338777226825497,0.047513331109726935,-0.03458344440377079,0.04160643036673322,-0.0028252052212552757,0.05410841560666696,-0.0207852947171917,0.020328168790867365,-0.018038141560885207,-0.023967565848358005,-0.020131546858333665,0.025514399554601934,0.04843142045748144,0.007674121273621759,-0.004227346161756906,-0.006068664206972633,0.07735349139039707,0.009210839856768113,-0.01811818407737378,0.0983069868815479,0.022747417589554585,-0.026235913810120002,0.006870245588558581,0.05928955938789314,-0.015571811328717674,0.0040295143477116786,0.01322661481354095,-0.026781453531780314,0.05044324133312488,0.052500662501024524,0.03764416415073162,-0.023494745515368357,-0.021717498144572484,0.018738249373268245,0.025943136201447524,0.0022064285327969262,-0.05199485003751025,-0.04059530047911298,-0.0014775808287879307,-0.0008679522963259123,-0.013723157602421375,0.05513368136612578,-0.01218737657752053,-0.06101906667166031,0.020116326594373287,-0.0008727270356426143,0.051380054170888506,-0.013236252741583946,0.04315609967950461,0.04531288874590152,0.006389358933714821,0.0283032922932636,0.022138428908882377,-0.006014860194608254,0.0485436914098501,0.024147434640981034,0.04330067905165594,0.013291719015791518,-0.050043095413590824,-0.040698056747264955,-0.025274396845976273,0.037435260888833866,0.03596292817922878,-0.04408919887457455,-0.059001144078483746,0.005566232813461449,-0.01603374306135026,-0.02613117505500817,0.0025082526749328395,-0.04699500118641657,-0.022030913181550697,-0.010063703584427855,0.04070558836461731,0.04415575035651439,-0.005013174304975669,0.045081197400920515,-0.025063949034707267,-0.012217672066347872,0.0006417255017305229,-0.017469290931719048,0.018232789660674025,0.01927170192312728,0.011126358307143862,-0.04141745263576802,0.02020069477552885,-0.012022553248137053,0.04662716464530442,-0.02581494171866084,-0.003443197693256006,-0.010344578092444245,-0.016856328156838653,0.026607784516557823,-0.015485386423533356,0.009374626371804829,0.030603840290064795,-0.030081299440472355,-0.011569247513673674,-0.037170844067484145,-0.005090498266500731,0.005364577216241896,-0.014795730150217012,0.007680546504315555,0.01677363327892379,-0.05365417713973248,-0.13571088565603134,0.14188619770267413,-0.06456346411927626,-0.09724709804413169,0.08417468534116751,-0.043737883642841306,-0.14163970826405495,-0.05198987946897292,-0.0563298515982782,-0.10529277910307917,-0.046493075552277274,0.16530505811696528,-0.10442963412492075,-0.15320426635163487,0.005336087116714683,-0.2358341111354685,-0.36707156419834885,-0.09628026085452177,-0.027531621995358968,-0.225907943116564,-0.12543967768520337,-0.025382849243750182,-0.03821231944826696,0.1323678431372993,0.1364552186797409,-0.06437179122226248,-0.11360683837474184,-0.17038714516260417,-0.011543979292163081,-0.054426990519365545,-0.08064701338495107,0.007414093171875327,-0.07228765385067247,-0.2867760640078572,-0.13056457960193182,-0.0436941373196706,0.07660679047752905,-0.19361612725583238,-0.16693370365216662,-0.004584554554421141,-0.08643576444482196,-0.09022487899122862,-0.04402950950146318,-0.23702735013904747,-0.06880859142258773,-0.21835404675318676,-0.0945001267336108,-0.0762064351304034,-0.09021025062436906,-0.01150289086311085,-0.05981223391111323,-0.2745522339720272,0.20849683637902997,0.08757734346484485,0.154619799413386,-0.0511789489281605,0.19693063239737912,0.24075112681089383,0.055798233393495496,0.16100513417403217,-0.09569415215640821,0.07953554550461946,0.08951493991923083,0.31340678723410853,0.11086422836073265,0.07713283545820912,0.052655702174393554,0.10464000690900643,0.2321367090398436,0.04098124095394092,-0.08128443708453226,0.029840055221135437,-0.019596004121309154,-0.06211974653057505,-0.17156292268516551,0.06065532060045424,-0.18880205024065846,0.1217372341322704,0.19451443527199966,0.22830637607650342,0.15789204508467652,-0.030001937143463484,0.0987017781452551,0.23475425416528783,0.3766036467098914,0.03806916716359399,0.038145140355894516,-0.05559184158902263,-0.026540686566480155,0.011952507889117982,0.06713392533407682,0.027471953009418158,0.04225088687162835,-0.15042122251270262,0.08460525543880308,0.33960120912853975,0.003954206869765095,0.04265075955049035,-0.09463520766598466,-0.07604320137600255,0.011009128043189823,0.07244711080780601,0.147831342265734,0.03615875060550982,0.17741154161149805,0.14041369980136945,-0.039257997278293974,-0.018334350630037376,0.16787178576385897,0.2422649622238584,0.05981332623290177,-0.11813296772454528,-0.17352553290903538,-0.039707979792588464,0.0028262602367913397,-0.04662171868654153,-0.28173772797117486,0.0488797239692797,-0.23176294498413322,-0.052780811230900136,-0.049172707340155115,-0.0680110120817184,0.03973559852389538,-0.10922896686898768,-0.16849566492898221,-0.029102355961453354,-0.09197974246636326,-0.048679853201033955,-0.11442229929893137,-0.07164774410136013,-0.08420687964370692,0.05443134426746795,-0.12838906244441878,0.0009347489077222044,-0.009255540318732538,-0.07914280073331054,0.03213059731148605,-0.07797957590175952,0.07105077361495896,0.00015934446747795894,-0.15464996782272258,0.06267750359665374,0.17888839527260042,0.1347411072663798,0.12471473192910373,0.00918150339408785,-0.05930312443485085,0.023615818583974317,-0.008308492665569397,0.002600843220084743,0.14893034593083876,0.18003265090459405,0.168681050846816,-0.03487096424093264,-0.08578766126824199,-0.15385990907669983,-0.1925602935276036,-0.16354595519225834,-0.08494924277476885,-0.03455323196211305,-0.02279427560198991,-0.06877558607873671,0.019221464078472264,0.0695299813541405]],"denseBiases":[-0.026186458338948684,-0.03604252711789141,0.06222898545683956]}

function loadModel() {
    // Try to retrieve the saved JSON string
    const savedData = localStorage.getItem('cnn_pretrained_weights');
    
    // Use LocalStorage if available (from a previous session), 
    // otherwise fall back to the hardcoded pre-trained weights
    const modelData = savedData ? JSON.parse(savedData) : PRETRAINED_WEIGHTS;
    
    convLayer.filters = modelData.convFilters;
    convLayer.biases = modelData.convBiases;
    denseLayer.weights = modelData.denseWeights;
    denseLayer.biases = modelData.denseBiases;

    const statusDiv = document.getElementById('trainingStatus');
    if (statusDiv) {
        statusDiv.innerText = savedData
            ? "Model loaded from LocalStorage. Ready to predict!"
            : "Pre-trained model loaded. Ready to predict!";
        statusDiv.style.color = "green";
        statusDiv.style.fontWeight = "bold";
    }
}

// === Global Model Variables ===
let convLayer, reluLayer, poolLayer, flattenLayer, denseLayer, softmaxLayer;
let currentNumFilters = null;
let currentFilterSize = null;
let trainingData = []; 

// === Model Builder Function ===
function buildModel(numFilters, filterSize) {
    const convOutputSize = MODEL_INPUT_SIZE - filterSize + 1;
    const poolOutputSize = Math.floor(convOutputSize / 2);
    const flattenedSize = (poolOutputSize ** 2) * numFilters;
    
    convLayer = new Conv2D(numFilters, filterSize);
    reluLayer = new ReLU();
    poolLayer = new MaxPool2D(2, 2);
    flattenLayer = new Flatten();
    denseLayer = new Dense(flattenedSize, 3);
    softmaxLayer = new Softmax();
    
    currentNumFilters = numFilters;
    currentFilterSize = filterSize;
    console.log(`Model built: ${numFilters} filters, size ${filterSize}, flattened size: ${flattenedSize}`);
}

// Initialize with default UI values on page load
buildModel(
    parseInt(document.getElementById('filtersInput').value),
    parseInt(document.getElementById('filterSizeInput').value)
);

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

trainBtn.addEventListener('click', () => {
    const numFilters = parseInt(document.getElementById('filtersInput').value);
    const filterSize = parseInt(document.getElementById('filterSizeInput').value);
    const lr = parseFloat(document.getElementById('lrInput').value);
    const epochs = parseInt(document.getElementById('epochsInput').value);
    
    // Rebuild only if architecture changed
    if (numFilters !== currentNumFilters || filterSize !== currentFilterSize) {
        if (!confirm("Changing architecture will reset model weights. Continue?")) return;
        trainingData = []; 
        buildModel(numFilters, filterSize);
    }
    
    let label = prompt("What did you draw? (0: Circle, 1: Square, 2: Triangle)");
    if (label === null || label === "") return;
    const targetIndex = parseInt(label);
    
    const pixels = getPixelData(); 
    const inputMatrix = toMatrix(pixels, MODEL_INPUT_SIZE); 
    
    let finalLoss = 0;
    
    for (let e = 1; e <= epochs; e++) {
        const predictions = modelForward(inputMatrix);
        const { loss, gradient } = computeLossAndGradient(predictions, targetIndex);
        finalLoss = loss;
        modelBackward(gradient, lr);
    }
    
    saveModel();
    document.getElementById('trainingStatus').innerText = "Training completed!";
    document.getElementById('currentLoss').innerText = finalLoss.toFixed(4);
    alert(`Trained! Loss: ${finalLoss.toFixed(4)}`);
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
