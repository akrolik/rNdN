#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "CUDA/Data.h"

namespace CUDA {

class ConstantMappedBuffer : public Data
{
public:
	ConstantMappedBuffer(const void *buffer) : m_buffer(buffer) {}

	const void *GetBuffer() const { return m_buffer; }
	void SetBuffer(const void *buffer) { m_buffer = buffer; }

	const void *GetAddress() const override;

private:
	const void *m_buffer = nullptr;
	mutable void *m_device = nullptr;
};

}
