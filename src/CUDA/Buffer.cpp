#include "CUDA/Buffer.h"

#include "CUDA/Utils.h"

#include "Utils/Chrono.h"

namespace CUDA {

void Buffer::Copy(Buffer *destination, ConstantBuffer *source, size_t size, size_t destinationOffset, size_t sourceOffset)
{
	std::string description = "CUDA copy ";
	if (source->GetTag() != "")
	{
		description += "'" + source->GetTag() + "' ";
	}
	if (destination->GetTag() != "")
	{
		description += "to '" + destination->GetTag() + "' ";
	}
	description += "(" + std::to_string(size) + " bytes";
	if (destinationOffset > 0)
	{
		description += "; d_offset " + std::to_string(destinationOffset) + " bytes";
	}
	if (sourceOffset > 0)
	{
		description += "; s_offset " + std::to_string(sourceOffset) + " bytes";
	}
	description += ")";

	auto start = Utils::Chrono::StartCUDA(description);

	checkDriverResult(cuMemcpy(destination->GetGPUBuffer() + destinationOffset, source->GetGPUBuffer() + sourceOffset, size));

	Utils::Chrono::End(start);
}

ConstantBuffer::ConstantBuffer(const void *buffer, size_t size) : m_CPUBuffer(buffer), m_size(size)
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

std::string ConstantBuffer::ChronoDescription(const std::string& operation, size_t size)
{
	if (m_tag == "")
	{
		return "CUDA " + operation + " (" + std::to_string(size) + " bytes)";
	}
	return "CUDA " + operation + " '" + m_tag + "' (" + std::to_string(size) + " bytes)";
}

ConstantBuffer::~ConstantBuffer()
{
	auto start = Utils::Chrono::StartCUDA(ChronoDescription("free", m_allocatedSize));

	checkDriverResult(cuMemFree(m_GPUBuffer));

	Utils::Chrono::End(start);
}

void ConstantBuffer::AllocateOnGPU()
{
	auto start = Utils::Chrono::StartCUDA(ChronoDescription("allocation", m_allocatedSize));

	checkDriverResult(cuMemAlloc(&m_GPUBuffer, m_allocatedSize));

	Utils::Chrono::End(start);
}

void ConstantBuffer::Clear()
{
	auto start = Utils::Chrono::StartCUDA(ChronoDescription("clear", m_allocatedSize));

	checkDriverResult(cuMemsetD8(m_GPUBuffer, 0, m_allocatedSize));

	Utils::Chrono::End(start);
}

void ConstantBuffer::TransferToGPU()
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
