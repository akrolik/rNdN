#include "CUDA/ConstantMappedBuffer.h"

#include "CUDA/Utils.h"

namespace CUDA {

const void *ConstantMappedBuffer::GetAddress() const
{
	checkRuntimeError(cudaHostGetDevicePointer(&m_device, const_cast<void *>(m_buffer), 0));
	return &m_device;
}

}
