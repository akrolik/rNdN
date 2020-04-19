#include "CUDA/Buffer.h"

#include "CUDA/Utils.h"

#include "Utils/Chrono.h"

namespace CUDA {

void Buffer::Copy(Buffer *destination, Buffer *source, size_t size)
{
	std::string description = "CUDA copy ";
	if (source->m_tag != "")
	{
		description += "'" + source->m_tag + "' ";
	}
	if (destination->m_tag != "")
	{
		description += "to '" + destination->m_tag + "' ";
	}
	description += "(" + std::to_string(size) + " bytes)";

	auto start = Utils::Chrono::StartCUDA(description);

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

std::string Buffer::ChronoDescription(const std::string& operation, size_t size)
{
	if (m_tag == "")
	{
		return "CUDA " + operation + " (" + std::to_string(size) + " bytes)";
	}
	return "CUDA " + operation + " '" + m_tag + "' (" + std::to_string(size) + " bytes)";
}

Buffer::~Buffer()
{
	auto start = Utils::Chrono::StartCUDA(ChronoDescription("free", m_allocatedSize));

	checkDriverResult(cuMemFree(m_GPUBuffer));

	Utils::Chrono::End(start);
}

void Buffer::AllocateOnGPU()
{
	auto start = Utils::Chrono::StartCUDA(ChronoDescription("allocation", m_allocatedSize));

	checkDriverResult(cuMemAlloc(&m_GPUBuffer, m_allocatedSize));

	Utils::Chrono::End(start);
}

void Buffer::Clear()
{
	auto start = Utils::Chrono::StartCUDA(ChronoDescription("clear", m_allocatedSize));

	checkDriverResult(cuMemsetD8(m_GPUBuffer, 0, m_allocatedSize));

	Utils::Chrono::End(start);
}

void Buffer::TransferToGPU()
{
	auto start = Utils::Chrono::StartCUDA(ChronoDescription("transfer", m_size) + " ->");

	checkDriverResult(cuMemcpyHtoD(m_GPUBuffer, m_CPUBuffer, m_size));

	Utils::Chrono::End(start);
}

void Buffer::TransferToCPU()
{
	auto start = Utils::Chrono::StartCUDA(ChronoDescription("transfer", m_size) + " <-");

	checkDriverResult(cuMemcpyDtoH(m_CPUBuffer, m_GPUBuffer, m_size));

	Utils::Chrono::End(start);
}

}
