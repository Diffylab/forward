package main

import "C"

import (
	"cl"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
	"time"
)

func check(e error) {
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
		io.WriteString(f, s+"\n")
	}
	f.Close()
}

type naclContext struct {
	NaCl             *cl.Nacl
	weights_buffer_1 *cl.MemObject
	weights_buffer_2 *cl.MemObject
	bias_buffer_1    *cl.MemObject
	bias_buffer_2    *cl.MemObject
}

func main() {
	if len(os.Args) != 2 {
		fmt.Printf("Usage: %s <path to floats files>\n", os.Args[0])
		return
	}
	base_path := os.Args[1]

	/*
		SETUP
	*/

	start := time.Now()

	// Read a bunch of files for inputs
	filters := floatsFromFile(base_path+"/filters.txt", "\n")
	biases := floatsFromFile(base_path+"/biases.txt", "\n")
	inputs := floatsFromFile(base_path+"/input.txt", "\n")
	output := floatsFromFile(base_path+"/out.txt", "\n")

	filters2 := floatsFromFile(base_path+"/filters2.txt", "\n")
	biases2 := floatsFromFile(base_path+"/bias2.txt", "\n")

	// Compile needed kernels
	nacl := cl.NewNaCL()
	convKernel := compileKernel(nacl, "convolve_imagecubes_float2")
	tanhKernel := compileKernel(nacl, "forwardNaive")

	weights_buffer_1 := createBuffer(nacl, filters)

	weights_buffer_2 := createBuffer(nacl, filters2)

	bias_buffer_1 := createBuffer(nacl, biases)

	bias_buffer_2 := createBuffer(nacl, biases2)

	nacl.Queue.Finish()

	time_diff := time.Since(start)

	fmt.Printf("Setup time is (%s)\n", time_diff)

	/*
		END OF SETUP
	*/

	inputBuffer := createBuffer(nacl, inputs)

	start = time.Now()
	// Run the kernel
	fmt.Printf("output size %d\n", len(biases2))
	convBuffer := convolve(nacl, convKernel, weights_buffer_1, inputBuffer, len(output), int32(1), 1024, 8192)

	_ = repeatedAdd(nacl, nil, bias_buffer_1, convBuffer, len(output), 64, 8192, false, 8192, 8, 1024)

	tanBuffer := tanh(nacl, tanhKernel, convBuffer)
	nacl.Queue.Finish()

	outputBuffer := convolve(nacl, convKernel, weights_buffer_2, tanBuffer, 8, int32(0), 8, 8)

	output_ := repeatedAdd(nacl, nil, bias_buffer_2, outputBuffer, 8, 64, 64, true, 8, 8, 1)

	nacl.Queue.Finish()

	fmt.Printf("run time time is (%s)\n", time.Since(start))

	// Verify the output
	fmt.Printf("Results: %v\n", output_)
}

func createBuffer(nacl *cl.NaCL, arr []float32) *cl.MemObject {
	buffer, err := nacl.Context.CreateEmptyBuffer(cl.MemReadWrite, len(arr)*32/8)
	check(err)

	_, err = nacl.Queue.EnqueueWriteBufferFloat32(buffer, true, 0, arr, nil)
	check(err)
	return buffer
}

func tanh(nacl *cl.NaCL, kernel *cl.Kernel, inputBuffer *cl.MemObject) *cl.MemObject {
	outputBuffer, err := nacl.Context.CreateEmptyBuffer(cl.MemReadWrite, 8192*32/8)
	check(err)

	err = kernel.SetArgs(outputBuffer, inputBuffer)
	check(err)
	_, err = nacl.Queue.EnqueueNDRangeKernel(kernel, nil, []int{8192}, []int{1024}, nil)
	check(err)

	return outputBuffer
}

func repeatedAdd(nacl *cl.NaCL, kernel *cl.Kernel, source_buffer *cl.MemObject, outputBuffer *cl.MemObject, output_size int, localWorkSize int, globalWorkSize int, readOutput bool, N int32, sourceSize int32, repeatSize int32) []float32 {

	source := make([]float32, 8)
	output := make([]float32, output_size)

	_, err := nacl.Queue.EnqueueReadBufferFloat32(source_buffer, true, 0, source, nil)
	check(err)

	_, err = nacl.Queue.EnqueueReadBufferFloat32(outputBuffer, true, 0, output, nil)
	check(err)

	nacl.Queue.Finish()

	for i, _ := range output {
		output[i] += source[(int32(i)/repeatSize)%sourceSize]
	}

	if readOutput {
		return output
	}

	_, err = nacl.Queue.EnqueueWriteBufferFloat32(outputBuffer, true, 0, output, nil)
	check(err)

	return nil
}

func convolve(nacl *cl.NaCL, kernel *cl.Kernel, weights_buffer *cl.MemObject, inputBuffer *cl.MemObject, output_size int, firstFlag int32, localWorkSize int, globalWorkSize int) *cl.MemObject {
	var num_examples int32 = 1

	outputBuffer, err := nacl.Context.CreateEmptyBuffer(cl.MemReadWrite, output_size*32/8)
	check(err)

	err = kernel.SetArgs(num_examples, inputBuffer, weights_buffer, outputBuffer, firstFlag)
	check(err)

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
	filename := "./kernel_file.aocx"

	filebytes, err := ioutil.ReadFile(filename)
	if err != nil {
		panic(err)
	}

	program, err := nacl.Context.CreateProgramWithBinary(nacl.Device, filebytes)
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
