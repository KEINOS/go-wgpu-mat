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

//nolint:funlen // Backend construction stages keep cleanup adjacent to ownership.
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
		if shader != nil {
			deps.releaseShaderModule(shader)
		}

		return nil, wrapError(err, "create %s shader", operation)
	}

	if shader == nil {
		return nil, sentinelError(ErrBackendUnavailable, "create %s shader returned nil", operation)
	}
	defer deps.releaseShaderModule(shader)

	pipelineLayout, err := deps.createPipelineLayout(device, &wgpu.PipelineLayoutDescriptor{
		Label:            fmt.Sprintf("go-wgpu-mat-%s-pipeline-layout", operation),
		BindGroupLayouts: []*wgpu.BindGroupLayout{bindGroupLayout},
	})
	if err != nil {
		if pipelineLayout != nil {
			deps.releasePipelineLayout(pipelineLayout)
		}

		return nil, wrapError(err, "create %s pipeline layout", operation)
	}

	if pipelineLayout == nil {
		return nil, sentinelError(ErrBackendUnavailable, "create %s pipeline layout returned nil", operation)
	}
	defer deps.releasePipelineLayout(pipelineLayout)

	pipeline, err := deps.createComputePipeline(device, &wgpu.ComputePipelineDescriptor{
		Label:                         fmt.Sprintf("go-wgpu-mat-%s-pipeline", operation),
		Layout:                        pipelineLayout,
		Module:                        shader,
		EntryPoint:                    "main",
		Constants:                     nil,
		ZeroInitializeWorkgroupMemory: nil,
	})
	if err != nil {
		if pipeline != nil {
			deps.releaseComputePipeline(pipeline)
		}

		return nil, wrapError(err, "create %s compute pipeline", operation)
	}

	if pipeline == nil {
		return nil, sentinelError(ErrBackendUnavailable, "create %s compute pipeline returned nil", operation)
	}

	return pipeline, nil
}

//nolint:cyclop,funlen // Each backend stage has a distinct cleanup path.
func encodeAndSubmitCompute(
	ctx *Context,
	pipeline *wgpu.ComputePipeline,
	bindGroup *wgpu.BindGroup,
	dispatch computeDispatch,
	operation string,
	deps matMulWGPUDeps,
) error {
	device := ctx.device

	encoder, err := deps.createCommandEncoder(device, &wgpu.CommandEncoderDescriptor{
		Label: fmt.Sprintf("go-wgpu-mat-%s-encoder", operation),
	})
	if err != nil {
		if encoder != nil {
			deps.discardCommandEncoder(encoder)
		}

		return wrapError(err, "create %s command encoder", operation)
	}

	if encoder == nil {
		return sentinelError(ErrBackendUnavailable, "create %s command encoder returned nil", operation)
	}

	pass, err := deps.beginComputePass(encoder, nil)
	if err != nil {
		if pass != nil {
			_ = deps.endComputePass(pass)
		}

		deps.discardCommandEncoder(encoder)

		return wrapError(err, "begin %s compute pass", operation)
	}

	if pass == nil {
		deps.discardCommandEncoder(encoder)

		return sentinelError(ErrBackendUnavailable, "begin %s compute pass returned nil", operation)
	}

	deps.setPipeline(pass, pipeline)
	deps.setBindGroup(pass, 0, bindGroup, nil)
	deps.dispatch(pass, dispatch.x, dispatch.y, dispatch.z)

	err = deps.endComputePass(pass)
	if err != nil {
		deps.discardCommandEncoder(encoder)

		return wrapError(err, "end %s compute pass", operation)
	}

	commandBuffer, err := deps.finishCommandEncoder(encoder)
	if err != nil {
		if commandBuffer != nil {
			deps.releaseCommandBuffer(commandBuffer)
		}

		return wrapError(err, "finish %s command encoder", operation)
	}

	if commandBuffer == nil {
		return sentinelError(ErrBackendUnavailable, "finish %s command encoder returned nil", operation)
	}
	defer deps.releaseCommandBuffer(commandBuffer)

	err = ctx.withQueue(func() error {
		return deps.submit(device, commandBuffer)
	})
	if err != nil {
		return wrapError(err, "submit %s command buffer", operation)
	}

	ctx.recordComputeSubmission()

	return nil
}
