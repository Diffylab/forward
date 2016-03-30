package main

import (
	"cl"
	"fmt"
	"io/ioutil"
	"time"
)

type naclContext struct {
	NaCl           *cl.NaCL
	WeightsBuffer1 *cl.MemObject
	WeightsBuffer2 *cl.MemObject
	BiasBuffer1    *cl.MemObject
	BiasBuffer2    *cl.MemObject
	convKernel1    *cl.Kernel
	convKernel2    *cl.Kernel
	tanhKernel     *cl.Kernel
	Queue          *cl.CommandQueue
}

var (
	gNaCl           *cl.NaCL
	gWeightsBuffer1 *cl.MemObject
	gWeightsBuffer2 *cl.MemObject
	gBiasBuffer1    *cl.MemObject
	gBiasBuffer2    *cl.MemObject
	gProgram        *cl.Program
)

func init() {
	gNaCl = cl.NewNaCL()
	nacl := gNaCl
	commandQueue, err := nacl.Context.CreateCommandQueue(nacl.Device, cl.CommandQueueProfilingEnable)
	check(err)
	base_path := "txt"
	if gWeightsBuffer1 == nil {
		filters := floatsFromFile(base_path+"/filters.txt", "\n")
		gWeightsBuffer1 = createBuffer(nacl, commandQueue, filters)
	}

	if gWeightsBuffer2 == nil {
		filters2 := floatsFromFile(base_path+"/filters2.txt", "\n")
		gWeightsBuffer2 = createBuffer(nacl, commandQueue, filters2)
	}

	if gBiasBuffer1 == nil {
		biases := floatsFromFile(base_path+"/biases.txt", "\n")
		gBiasBuffer1 = createBuffer(nacl, commandQueue, biases)
	}

	if gBiasBuffer2 == nil {
		biases2 := floatsFromFile(base_path+"/bias2.txt", "\n")
		gBiasBuffer2 = createBuffer(nacl, commandQueue, biases2)
	}

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

	gProgram = program

	commandQueue.Finish()
	commandQueue.Release()
}

func createNaclContext() *naclContext {
	nacl := gNaCl
	convKernel1 := compileKernel(gProgram, "convolve_imagecubes_float2")
	convKernel2 := compileKernel(gProgram, "convolve_imagecubes_float2")
	tanhKernel := compileKernel(gProgram, "forwardNaive")

	commandQueue, err := nacl.Context.CreateCommandQueue(nacl.Device, cl.CommandQueueProfilingEnable)
	check(err)

	return &naclContext{
		NaCl:           nacl,
		WeightsBuffer1: gWeightsBuffer1,
		WeightsBuffer2: gWeightsBuffer2,
		BiasBuffer1:    gBiasBuffer1,
		BiasBuffer2:    gBiasBuffer2,
		convKernel1:    convKernel1,
		convKernel2:    convKernel2,
		tanhKernel:     tanhKernel,
		Queue:          commandQueue,
	}
}

func (ctx *naclContext) run(input []float32) []float32 {
	nacl := ctx.NaCl

	inputBuffer := createBuffer(nacl, ctx.Queue, input)

	convBuffer := ctx.convolve(true, inputBuffer)
	ctx.repeatedAdd(ctx.BiasBuffer1, convBuffer, 8192, false, 8, 1024)
	tanBuffer := ctx.tanh(convBuffer)
	outputBuffer := ctx.convolve(false, tanBuffer)
	output_ := ctx.repeatedAdd(ctx.BiasBuffer2, outputBuffer, 8, true, 8, 1)

	ctx.Queue.Finish()
	ctx.Queue.Release()
	return output_
}

func (ctx *naclContext) convolve(isFirst bool, inputBuffer *cl.MemObject) *cl.MemObject {

	ctx.Queue.Finish()
	start := time.Now()

	var kernel *cl.Kernel
	var firstFlag int32
	var output_size int
	var weights_buffer *cl.MemObject
	var globalWorkSize int
	var localWorkSize int

	if isFirst {
		kernel = ctx.convKernel1
		firstFlag = 1
		output_size = 8192
		weights_buffer = ctx.WeightsBuffer1
		globalWorkSize = 8192
		localWorkSize = 1024
	} else {
		kernel = ctx.convKernel2
		firstFlag = 0
		output_size = 8
		weights_buffer = ctx.WeightsBuffer2
		globalWorkSize = 8
		localWorkSize = 8
	}

	nacl := ctx.NaCl

	outputBuffer, err := nacl.Context.CreateEmptyBuffer(cl.MemReadWrite, output_size*32/8)
	check(err)

	err = kernel.SetArgs(int32(1), inputBuffer, weights_buffer, outputBuffer, firstFlag)
	check(err)

	_, err = ctx.Queue.EnqueueNDRangeKernel(kernel, nil, []int{globalWorkSize}, []int{localWorkSize}, nil)
	check(err)

	ctx.Queue.Finish()

	fmt.Printf("Convolve run time: %s\n", time.Since(start))

	return outputBuffer
}

func (ctx *naclContext) tanh(inputBuffer *cl.MemObject) *cl.MemObject {
	ctx.Queue.Finish()
	start := time.Now()

	nacl := ctx.NaCl
	kernel := ctx.tanhKernel
	outputBuffer, err := nacl.Context.CreateEmptyBuffer(cl.MemReadWrite, 8192*32/8)
	check(err)

	err = kernel.SetArgs(outputBuffer, inputBuffer)
	check(err)
	_, err = ctx.Queue.EnqueueNDRangeKernel(kernel, nil, []int{8192}, []int{1024}, nil)
	check(err)

	ctx.Queue.Finish()

	fmt.Printf("tanh run time: %s\n", time.Since(start))

	return outputBuffer
}

func (ctx *naclContext) repeatedAdd(source_buffer *cl.MemObject, outputBuffer *cl.MemObject, output_size int, readOutput bool, sourceSize int32, repeatSize int32) []float32 {
	ctx.Queue.Finish()
	start := time.Now()

	source := make([]float32, 8)
	output := make([]float32, output_size)

	_, err := ctx.Queue.EnqueueReadBufferFloat32(source_buffer, true, 0, source, nil)
	check(err)

	_, err = ctx.Queue.EnqueueReadBufferFloat32(outputBuffer, true, 0, output, nil)
	check(err)

	ctx.Queue.Finish()

	for i, _ := range output {
		output[i] += source[(int32(i)/repeatSize)%sourceSize]
	}

	if readOutput {
		return output
	}

	_, err = ctx.Queue.EnqueueWriteBufferFloat32(outputBuffer, true, 0, output, nil)
	check(err)

	ctx.Queue.Finish()

	fmt.Printf("repeatedAdd run time: %s\n", time.Since(start))

	return nil
}
