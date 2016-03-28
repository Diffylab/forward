package main

import "C"

import (
    "cl"
    "io/ioutil"
    "fmt"
    "io"
    "os"
    "strings"
    "strconv"
)

func check (e error) {
   if e != nil {
        panic(e)
    }
}

func floatsFromFile(file_path string, split_char string) []float32 {
    // Read a bunch of files for inputs
    dat, err := ioutil.ReadFile(file_path)
    s_result := strings.Split(string(dat), split_char)
    check(err)

    var result []float32
    for i := 0; i < len(s_result); i++ {
        if len(s_result[i]) == 0 {
            continue
        }
        f, _ := strconv.ParseFloat(s_result[i], 32)
        result = append(result, float32(f))
    }

    return result
}

func floatsToFile(fp string, arr []float32) {
    f, err := os.Create(fp)
    check(err)

    for i := 0; i < len(arr); i++ {
        s := strconv.FormatFloat(float64(arr[i]), 'f', -1, 32)
        io.WriteString(f, s + "\n")
    }
    f.Close()
}

func main() {
    if len(os.Args) != 2 {
        fmt.Printf("Usage: %s <path to floats files>\n", os.Args[0])
        return
    }
    base_path := os.Args[1]
    // Read a bunch of files for inputs
    filters := floatsFromFile(base_path + "/filters.txt", "\n")
    biases := floatsFromFile(base_path + "/biases.txt", "\n")
    inputs := floatsFromFile(base_path + "/input.txt", "\n")
    output := floatsFromFile(base_path + "/out.txt", "\n")

    // Run the kernel
    nacl := cl.NewNaCL()
    outputBuffer := convolve(nacl, filters, inputs, len(output))
    output_ := repeatedAdd(nacl, biases, outputBuffer, len(output))
    _, tanhOut := tanh(nacl, outputBuffer, len(output_))
    floatsToFile("tanhout.txt", tanhOut)
    floatsToFile("convout.txt", output_)

    // Verify the output
    //fmt.Printf("%v\n", tanhOut)
}

func tanh(nacl *cl.NaCL, inputBuffer *cl.MemObject, N int) (*cl.MemObject, []float32) {
    kernel := compileKernel(nacl, "forwardNaive")

	outputBuffer, err := nacl.Context.CreateEmptyBuffer(cl.MemReadWrite, N * 32/8)
    check(err)

    err = kernel.SetArgs(int32(N), outputBuffer, inputBuffer)
    check(err);

	_, err = nacl.Queue.EnqueueNDRangeKernel(kernel, nil, []int{8192}, []int{1024}, nil)
    check(err)
    fmt.Printf("Ran tanh kernel\n")

    output := make([]float32, N)
    _, err = nacl.Queue.EnqueueReadBufferFloat32(outputBuffer, true, 0, output, nil)
    check(err)

    fmt.Printf("Output length: %d\n", len(output))
    return outputBuffer, output
}


func repeatedAdd(nacl *cl.NaCL, biases []float32, outputBuffer *cl.MemObject, output_size int) []float32 {
    kernel := compileKernel(nacl, "repeated_add")
    var N int32 = 8192
    var sourceSize int32 = 8
    var repeatSize int32 = 1024

	source_buffer, err := nacl.Context.CreateEmptyBuffer(cl.MemReadWrite, len(biases) * 32/8)
    check(err)

    _, err = nacl.Queue.EnqueueWriteBufferFloat32(source_buffer, true, 0, biases, nil)
    check(err)

    err = kernel.SetArgs(N, sourceSize, repeatSize, outputBuffer, source_buffer)
    check(err);

    // todo: function to get work size
	_, err = nacl.Queue.EnqueueNDRangeKernel(kernel, nil, []int{8192}, []int{64}, nil)
    check(err)
    fmt.Printf("Ran kernel\n")


    output := make([]float32, output_size)
    _, err = nacl.Queue.EnqueueReadBufferFloat32(outputBuffer, true, 0, output, nil)
    check(err)

    // fmt.Printf("Output: %d\n", len(output))
    return output
}

func convolve(nacl *cl.NaCL, filters []float32, inputs []float32, output_size int) *cl.MemObject {
    kernel := compileKernel(nacl, "convolve_imagecubes_float2")
    var num_examples int32 = 1

	input_buffer, err := nacl.Context.CreateEmptyBuffer(cl.MemReadWrite, len(inputs) * 32/8)
    check(err)

    _, err = nacl.Queue.EnqueueWriteBufferFloat32(input_buffer, true, 0, inputs, nil)
    check(err)

	weights_buffer, err := nacl.Context.CreateEmptyBuffer(cl.MemReadWrite, len(filters) * 32/8)
    check(err)
    _, err = nacl.Queue.EnqueueWriteBufferFloat32(weights_buffer, true, 0, filters, nil)
    check(err)

	outputBuffer, err := nacl.Context.CreateEmptyBuffer(cl.MemReadWrite, output_size * 32/8)
    check(err)

    var outputSizeSquared, numFilters, outputSize, numInputPlanes int32
    outputSizeSquared = 1024
    numFilters = 8
    outputSize = 32
    numInputPlanes = 1

    err = kernel.SetArgs(num_examples, input_buffer, weights_buffer, outputBuffer,
                         outputSizeSquared, numFilters, outputSize,
                         numInputPlanes)
    check(err);

    // todo: function to get work size
	_, err = nacl.Queue.EnqueueNDRangeKernel(kernel, nil, []int{8192}, []int{1024}, nil)
    check(err)
    fmt.Printf("Ran convolve kernel\n")

    /*
    output := make([]float32, output_size)
    _, err = nacl.Queue.EnqueueReadBufferFloat32(outputBuffer, true, 0, output, nil)
    check(err)

    fmt.Printf("Output: %d\n", len(output))
    */

    return outputBuffer
}

func getCLContext() *cl.Context {
	platforms, err := cl.GetPlatforms()
	if err != nil {
		panic(err)
	}

	devices, err := cl.GetDevices(platforms[0], cl.DeviceTypeAll)
	if err != nil {
		panic(err)
	}

	context, err := cl.CreateContext(devices[:1])
	if err != nil {
		panic(err)
	}
    return context
}

func compileKernel(nacl *cl.NaCL, kernelName string) *cl.Kernel {
	filename := "./forward.cl"

	filebytes, err := ioutil.ReadFile(filename)
	if err != nil {
		panic(err)
	}

	fileString := string(filebytes)

    /*
	command_queue, err := context.CreateCommandQueue(devices[0], cl.CommandQueueProfilingEnable)
	if err != nil {
		panic(err)
	}
	buffer, err := context.CreateEmptyBuffer(cl.MemReadWrite, 128)
	if err != nil {
		panic(err)
	}
    */

	program, err := nacl.Context.CreateProgramWithSource([]string{fileString})
	if err != nil {
		panic(err)
	}

	err = program.BuildProgram(nil, "")
	if err != nil {
		panic(err)
	}

	kernel, err := program.CreateKernel(kernelName)
	if err != nil {
		panic(err)
	}

    return kernel
    /*
	err = kernel.SetArgs(size)
	if err != nil {
        fmt.Printf("Error setting args.\n")
		panic(err)
	}

	_, err = command_queue.EnqueueNDRangeKernel(kernel, []int{1,1,1}, []int{1,1,1}, []int{1,1,1}, nil)
	if err != nil {
		panic(err)
	}

    resPtr := C.CString(string(make([]byte, 128)))

	_, err = command_queue.EnqueueReadBuffer(buffer, true, 0, 128, unsafe.Pointer(resPtr), nil)
	if err != nil {
		panic(err)
	}

	result := C.GoString(resPtr)

	fmt.Printf("Result: %s\n", result)

	kernel.Release()
	program.Release()
	buffer.Release()
	command_queue.Release()
	context.Release()
    return context
    */
}
