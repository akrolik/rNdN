#include "CUDA/MappedBuffer.h"

#include "CUDA/Utils.h"

namespace CUDA {

const void *MappedBuffer::GetAddress() const
{
	checkRuntimeError(cudaHostGetDevicePointer(&m_device, m_buffer, 0));
	return &m_device;
}

}
