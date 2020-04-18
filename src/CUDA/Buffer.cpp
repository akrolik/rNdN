#include "CUDA/Buffer.h"

#include "CUDA/Utils.h"

#include "Utils/Chrono.h"

namespace CUDA {

void Buffer::Copy(Buffer *destination, Buffer *source, size_t size)
{
	auto start = Utils::Chrono::StartCUDA("CUDA copy (" + std::to_string(size) + " bytes)");

	checkDriverResult(cuMemcpy(destination->m_GPUBuffer, source->m_GPUBuffer, size));

	Utils::Chrono::End(start);
}

Buffer::Buffer(void *buffer, size_t size) : m_CPUBuffer(buffer), m_size(size)
{
	const auto multiple = 1024;
	if (m_size == 0)
	{
		m_allocatedSize = multiple;
	}
	else
	{
		m_allocatedSize = (((m_size + multiple - 1) / multiple) * multiple);
	}
}

Buffer::~Buffer()
{
	auto start = Utils::Chrono::StartCUDA("CUDA free (" + std::to_string(m_allocatedSize) + " bytes)");

	checkDriverResult(cuMemFree(m_GPUBuffer));

	Utils::Chrono::End(start);
}

void Buffer::AllocateOnGPU()
{
	auto start = Utils::Chrono::StartCUDA("CUDA allocation (" + std::to_string(m_allocatedSize) + " bytes)");

	checkDriverResult(cuMemAlloc(&m_GPUBuffer, m_allocatedSize));

	Utils::Chrono::End(start);
}

void Buffer::Clear()
{
	auto start = Utils::Chrono::StartCUDA("CUDA clear (" + std::to_string(m_allocatedSize) + " bytes)");

	checkDriverResult(cuMemsetD8(m_GPUBuffer, 0, m_allocatedSize));

	Utils::Chrono::End(start);
}

void Buffer::TransferToGPU()
{
	auto start = Utils::Chrono::StartCUDA("CUDA transfer (" + std::to_string(m_size) + " bytes) ->");

	checkDriverResult(cuMemcpyHtoD(m_GPUBuffer, m_CPUBuffer, m_size));

	Utils::Chrono::End(start);
}

void Buffer::TransferToCPU()
{
	auto start = Utils::Chrono::StartCUDA("CUDA transfer (" + std::to_string(m_size) + " bytes) <-");

	checkDriverResult(cuMemcpyDtoH(m_CPUBuffer, m_GPUBuffer, m_size));

	Utils::Chrono::End(start);
}

}
