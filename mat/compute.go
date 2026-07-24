package mat

import (
	"fmt"

	"github.com/gogpu/wgpu"
)

type computeDispatch struct {
	x uint32
	y uint32
	z uint32
}

func createComputePipeline(
	device *wgpu.Device,
	bindGroupLayout *wgpu.BindGroupLayout,
	operation string,
	shaderSource string,
	deps matMulWGPUDeps,
) (*wgpu.ComputePipeline, error) {
	shader, err := deps.createShaderModule(device, &wgpu.ShaderModuleDescriptor{
		Label: fmt.Sprintf("go-wgpu-mat-%s-shader", operation),
		WGSL:  shaderSource,
		SPIRV: nil,
	})
	if err != nil {
		return nil, wrapError(err, "create %s shader", operation)
	}
	defer deps.releaseShaderModule(shader)

	pipelineLayout, err := deps.createPipelineLayout(device, &wgpu.PipelineLayoutDescriptor{
		Label:            fmt.Sprintf("go-wgpu-mat-%s-pipeline-layout", operation),
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		return nil, wrapError(err, "create %s pipeline layout", operation)
	}
	defer deps.releasePipelineLayout(pipelineLayout)

	pipeline, err := deps.createComputePipeline(device, &wgpu.ComputePipelineDescriptor{
		Label:      fmt.Sprintf("go-wgpu-mat-%s-pipeline", operation),
		Layout:     pipelineLayout,
		Module:     shader,
		EntryPoint: "main",
	})
	if err != nil {
		return nil, wrapError(err, "create %s compute pipeline", operation)
	}

	return pipeline, nil
}

func encodeAndSubmitCompute(
	device *wgpu.Device,
	pipeline *wgpu.ComputePipeline,
	bindGroup *wgpu.BindGroup,
	dispatch computeDispatch,
	operation string,
	deps matMulWGPUDeps,
) error {
	encoder, err := deps.createCommandEncoder(device, &wgpu.CommandEncoderDescriptor{
		Label: fmt.Sprintf("go-wgpu-mat-%s-encoder", operation),
	})
	if err != nil {
		return wrapError(err, "create %s command encoder", operation)
	}

	pass, err := deps.beginComputePass(encoder, nil)
	if err != nil {
		return wrapError(err, "begin %s compute pass", operation)
	}

	deps.setPipeline(pass, pipeline)
	deps.setBindGroup(pass, 0, bindGroup, nil)
	deps.dispatch(pass, dispatch.x, dispatch.y, dispatch.z)

	err = deps.endComputePass(pass)
	if err != nil {
		return wrapError(err, "end %s compute pass", operation)
	}

	commandBuffer, err := deps.finishCommandEncoder(encoder)
	if err != nil {
		return wrapError(err, "finish %s command encoder", operation)
	}
	defer deps.releaseCommandBuffer(commandBuffer)

	err = deps.submit(device, commandBuffer)
	if err != nil {
		return wrapError(err, "submit %s command buffer", operation)
	}

	return nil
}
