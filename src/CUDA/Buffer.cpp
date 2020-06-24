#include "CUDA/Buffer.h"

#include "CUDA/Utils.h"

#include "Utils/Chrono.h"

namespace CUDA {

void Buffer::Copy(Buffer *destination, Buffer *source, size_t size, size_t destinationOffset, size_t sourceOffset)
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

void Buffer::TransferToCPU()
{
	auto start = Utils::Chrono::StartCUDA(m_buffer.ChronoDescription("transfer", m_buffer.GetSize()) + " <-");

	checkDriverResult(cuMemcpyDtoH(m_CPUBuffer, m_buffer.GetGPUBuffer(), m_buffer.GetSize()));

	Utils::Chrono::End(start);
}

}
