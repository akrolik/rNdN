#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "CUDA/Data.h"

namespace CUDA {

class MappedBuffer : public Data
{
public:
	MappedBuffer(void *buffer) : m_buffer(buffer) {}

	void *GetBuffer() const { return m_buffer; }
	void SetBuffer(void *buffer) { m_buffer = buffer; }

	const void *GetAddress() const override;

private:
	void *m_buffer = nullptr;
	mutable void *m_device = nullptr;
};

}
