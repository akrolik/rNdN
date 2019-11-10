#include "CUDA/Buffer.h"

#include "CUDA/Chrono.h"
#include "CUDA/Utils.h"

#include "Utils/Logger.h"

namespace CUDA {

void Buffer::Copy(Buffer *destination, Buffer *source, size_t size)
{
	auto start = Chrono::Start();

	checkDriverResult(cuMemcpy(destination->m_GPUBuffer, source->m_GPUBuffer, size));

	auto time = Chrono::End(start);
	Utils::Logger::LogTiming("CUDA copy (" + std::to_string(size) + " bytes)", time);
}

Buffer::Buffer(void *buffer, size_t size) : m_CPUBuffer(buffer), m_size(size)
{
	const auto multiple = 1024;
	if (m_size == 0)
	{
		m_paddedSize = multiple;
	}
	else
	{
		m_paddedSize = (((m_size + multiple - 1) / multiple) * multiple);
	}
}

Buffer::~Buffer()
{
	auto start = Chrono::Start();

	checkDriverResult(cuMemFree(m_GPUBuffer));

	auto time = Chrono::End(start);
	Utils::Logger::LogTiming("CUDA free (" + std::to_string(m_paddedSize) + " bytes)", time);
}

void Buffer::AllocateOnGPU()
{
	auto start = Chrono::Start();

	checkDriverResult(cuMemAlloc(&m_GPUBuffer, m_paddedSize));

	auto time = Chrono::End(start);
	Utils::Logger::LogTiming("CUDA allocation (" + std::to_string(m_paddedSize) + " bytes)", time);
}

void Buffer::Clear()
{
	auto start = Chrono::Start();

	checkDriverResult(cuMemsetD8(m_GPUBuffer, 0, m_paddedSize));

	auto time = Chrono::End(start);
	Utils::Logger::LogTiming("CUDA clear (" + std::to_string(m_paddedSize) + " bytes)", time);
}

void Buffer::TransferToGPU()
{
	auto start = Chrono::Start();

	checkDriverResult(cuMemcpyHtoD(m_GPUBuffer, m_CPUBuffer, m_size));

	auto time = Chrono::End(start);
	Utils::Logger::LogTiming("CUDA transfer (" + std::to_string(m_size) + " bytes) ->", time);
}

void Buffer::TransferToCPU()
{
	auto start = Chrono::Start();

	checkDriverResult(cuMemcpyDtoH(m_CPUBuffer, m_GPUBuffer, m_size));

	auto time = Chrono::End(start);
	Utils::Logger::LogTiming("CUDA transfer (" + std::to_string(m_size) + " bytes) <-", time);
}

}
