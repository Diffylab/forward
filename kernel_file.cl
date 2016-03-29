#define TANH 1

// expected defines:
// one of: [ TANH | RELU | LINEAR | SIGMOID | SCALEDTANH | ELU ]
#ifdef TANH
    #define ACTIVATION_FUNCTION(output) (tanh(output))
#elif defined SCALEDTANH
    #define ACTIVATION_FUNCTION(output) (1.7159f * tanh(0.66667f * output))
#elif SIGMOID
    #define ACTIVATION_FUNCTION(output) (1.0f / (1 + exp(-output)))
#elif defined RELU
    #define ACTIVATION_FUNCTION(output) (output> 0 ? output : 0)
#elif defined ELU
    #define ACTIVATION_FUNCTION(output) (output> 0 ? output : exp(output) - 1)
#elif defined LINEAR
    #define ACTIVATION_FUNCTION(output) (output)
#endif

#ifdef ACTIVATION_FUNCTION // protect against not defined
kernel void forwardNaive(const int N, global float * restrict out, global const float * restrict in) {
    const int globalId = get_global_id(0);
    if (globalId >= N) {
        return;
    }
    out[globalId] = ACTIVATION_FUNCTION(in[globalId]);
}
#endif

void kernel convolve_imagecubes_float2(
    const int numExamples,
    global const float * restrict inputs, global const float * restrict filters, 
    global float * restrict output, const int isFirstTime) {
    int globalId = get_global_id(0);

    int gInputSizeSquared;
    int gHalfFilterSize;
    int gInputSize;
    int gEven;
    int gOutputSizeSquared;
    int gNumFilters;
    int gOutputSize;
    int gNumInputPlanes;
    int gFilterSize;
    int gFilterSizeSquared;
    int gPadZeros;

    if (isFirstTime) {
    	gInputSizeSquared = 1024;
    	gHalfFilterSize = 2;
    	gInputSize = 32;
    	gEven = 0;
    	gOutputSizeSquared = 1024;
    	gNumFilters = 8;
    	gOutputSize = 32;
    	gNumInputPlanes = 1;
    	gFilterSize = 5;
    	gFilterSizeSquared = 25;
    	gPadZeros = 1;
    } else {
    	gInputSizeSquared = 1024;
    	gHalfFilterSize = 16;
    	gInputSize = 32;
    	gEven = 1;
    	gOutputSizeSquared = 1;
    	gNumFilters = 7;
    	gOutputSize = 1;
    	gNumInputPlanes = 8;
    	gFilterSize = 32;
    	gFilterSizeSquared = 1024;
    	gPadZeros = 0;
    }

    int outputImage2Id = globalId / gOutputSizeSquared;
    int exampleId = outputImage2Id / gNumFilters;
    int filterId = outputImage2Id % gNumFilters;

    // intraimage coords
    int localid = globalId % gOutputSizeSquared;
    int outputRow = localid / gOutputSize;
    int outputCol = localid % gOutputSize;

    global float const*inputCube = inputs + exampleId * gNumInputPlanes * gInputSizeSquared;
    global float const*filterCube = filters + filterId * gNumInputPlanes * gFilterSizeSquared;

    float sum = 0;
    if (exampleId < numExamples) {
        for (int inputPlaneIdx = 0; inputPlaneIdx < gNumInputPlanes; inputPlaneIdx++) {
            global float const*inputPlane = inputCube + inputPlaneIdx * gInputSizeSquared;
            global float const*filterPlane = filterCube + inputPlaneIdx * gFilterSizeSquared;
            for (int u = -gHalfFilterSize; u <= gHalfFilterSize - gEven; u++) {
                // trying to reduce register pressure...
		int inputRowIdx;
		if (gPadZeros) {
			inputRowIdx = outputRow + u;
		} else {
			inputRowIdx = outputRow + u + gHalfFilterSize;
		}
                global float const *inputRow = inputPlane + inputRowIdx * gInputSize;
                global float const *filterRow = filterPlane + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
                bool rowOk = inputRowIdx >= 0 && inputRowIdx < gInputSize;
                #pragma unroll
                for (int v = -gHalfFilterSize; v <= gHalfFilterSize - gEven; v++) {
		    int inputColIdx;
		    if (gPadZeros) {
			inputColIdx = (outputCol + v);
		    } else {
			inputColIdx = (outputCol + v + gHalfFilterSize);
		    }
                    bool process = rowOk && inputColIdx >= 0 && inputColIdx < gInputSize;
                    if (process) {
                            sum += inputRow[inputColIdx] * filterRow[v];
                    }
                }
            }
        }
    }

    if (exampleId < numExamples) {
        output[globalId] = sum;
    }
}


kernel void repeated_add(const int N, const int sourceSize, const int repeatSize, global float * restrict target, global const float * restrict source) {
    const int globalId = get_global_id(0);
    if (globalId >= N) {
        return;
    }
    target[globalId] += source[ (globalId / repeatSize) % sourceSize ];
}

