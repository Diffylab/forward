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

    filters2 := floatsFromFile(base_path + "/filters2.txt", "\n")
    biases2 := floatsFromFile(base_path + "/bias2.txt", "\n")

    // Compile needed kernels
    nacl := cl.NewNaCL()
    convKernel := compileKernel(nacl, "convolve_imagecubes_float2")
    tanhKernel := compileKernel(nacl, "forwardNaive")
    repeatedAddKernel := compileKernel(nacl, "repeated_add")

    inputBuffer := createBuffer(nacl, inputs)

    // Run the kernel
    convBuffer := convolve(nacl, convKernel, filters, inputBuffer, len(output), int32(1), 1024, 8192)
    output_ := repeatedAdd(nacl, repeatedAddKernel, biases, convBuffer, len(output), 64, 8192)
    tanBuffer, _ := tanh(nacl, tanhKernel, convBuffer, len(output_))

    outputBuffer := convolve(nacl, convKernel, filters2, tanBuffer, 7, int32(0), 7, 7)
    output_ = repeatedAdd(nacl, repeatedAddKernel, biases2, outputBuffer, 7, 64, 64)

    // Verify the output
    /*
    floatsToFile("tanhout.txt", tanhOut)
    floatsToFile("out.txt", output_)
    */
    fmt.Printf("Results: %v\n", output_)
}

func createBuffer(nacl *cl.NaCL, arr []float32) *cl.MemObject {
	buffer, err := nacl.Context.CreateEmptyBuffer(cl.MemReadWrite, len(arr) * 32/8)
    check(err)

    _, err = nacl.Queue.EnqueueWriteBufferFloat32(buffer, true, 0, arr, nil)
    check(err)
    return buffer
}

func tanh(nacl *cl.NaCL, kernel *cl.Kernel, inputBuffer *cl.MemObject, N int) (*cl.MemObject, []float32) {
	outputBuffer, err := nacl.Context.CreateEmptyBuffer(cl.MemReadWrite, N * 32/8)
    check(err)

    err = kernel.SetArgs(int32(N), outputBuffer, inputBuffer)
    check(err);

	_, err = nacl.Queue.EnqueueNDRangeKernel(kernel, nil, []int{8192}, []int{1024}, nil)
    check(err)

    output := make([]float32, N)
    _, err = nacl.Queue.EnqueueReadBufferFloat32(outputBuffer, true, 0, output, nil)
    check(err)

    return outputBuffer, output
}


func repeatedAdd(nacl *cl.NaCL, kernel *cl.Kernel, biases []float32, outputBuffer *cl.MemObject, output_size int, localWorkSize int, globalWorkSize int) []float32 {
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
	_, err = nacl.Queue.EnqueueNDRangeKernel(kernel, nil, []int{globalWorkSize}, []int{localWorkSize}, nil)
    check(err)

    output := make([]float32, output_size)
    _, err = nacl.Queue.EnqueueReadBufferFloat32(outputBuffer, true, 0, output, nil)
    check(err)

    return output
}

func convolve(nacl *cl.NaCL, kernel *cl.Kernel, filters []float32, inputBuffer *cl.MemObject, output_size int, firstFlag int32, localWorkSize int, globalWorkSize int) *cl.MemObject {
    var num_examples int32 = 1

    weights_buffer := createBuffer(nacl, filters)

	outputBuffer, err := nacl.Context.CreateEmptyBuffer(cl.MemReadWrite, output_size * 32/8)
    check(err)

    err = kernel.SetArgs(num_examples, inputBuffer, weights_buffer, outputBuffer, firstFlag)
    check(err);

	_, err = nacl.Queue.EnqueueNDRangeKernel(kernel, nil, []int{globalWorkSize}, []int{localWorkSize}, nil)
    check(err)

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
	filename := "./kernel_file.cl"

	filebytes, err := ioutil.ReadFile(filename)
	if err != nil {
		panic(err)
	}

	fileString := string(filebytes)

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
}
