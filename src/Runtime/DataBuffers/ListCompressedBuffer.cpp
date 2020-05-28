#include "Runtime/DataBuffers/ListCompressedBuffer.h"

#include "HorseIR/Utils/PrettyPrinter.h"

namespace Runtime {

ListCompressedBuffer::ListCompressedBuffer(const TypedVectorBuffer<CUdeviceptr> *dataAddresses, const TypedVectorBuffer<CUdeviceptr> *sizeAddresses, const TypedVectorBuffer<std::int32_t> *sizes, VectorBuffer *values)
	: m_dataAddresses(dataAddresses), m_sizeAddresses(sizeAddresses), m_sizes(sizes), m_values(values)
{
	m_type = new HorseIR::ListType(m_values->GetType()->Clone());

	std::vector<const Analysis::Shape *> cellShapes;
	for (const auto size : sizes->GetCPUReadBuffer()->GetValues())
	{
		cellShapes.push_back(new Analysis::VectorShape(new Analysis::Shape::ConstantSize(size)));
	}
	m_shape = new Analysis::ListShape(new Analysis::Shape::ConstantSize(cellShapes.size()), cellShapes);
}

ListCompressedBuffer::~ListCompressedBuffer()
{
}

ListCompressedBuffer *ListCompressedBuffer::Clone() const
{
	return nullptr;
	//TODO: Reallocation requires re-offsetting
	// return new ListCompressedBuffer(m_dataAddresses->Clone(), m_sizeAddresses->Clone(), m_values->Clone());
}

void ListCompressedBuffer::ValidateCPU(bool recursive) const
{
	DataBuffer::ValidateCPU(recursive);

	m_dataAddresses->ValidateCPU();
	m_sizeAddresses->ValidateCPU();
	m_sizes->ValidateCPU();
	m_values->ValidateCPU();
}

void ListCompressedBuffer::ValidateGPU(bool recursive) const
{
	DataBuffer::ValidateGPU(recursive);

	m_dataAddresses->ValidateGPU();
	m_sizeAddresses->ValidateGPU();
	m_sizes->ValidateGPU();
	m_values->ValidateGPU();
}

CUDA::Buffer *ListCompressedBuffer::GetGPUWriteBuffer()
{
	ValidateGPU();
	m_values->GetGPUWriteBuffer();
	return m_dataAddresses->GetGPUReadBuffer();
}

CUDA::Buffer *ListCompressedBuffer::GetGPUReadBuffer() const
{
	ValidateGPU();
	m_values->GetGPUReadBuffer();
	return m_dataAddresses->GetGPUReadBuffer();
}

std::string ListCompressedBuffer::Description() const
{
	std::string description = HorseIR::PrettyPrinter::PrettyString(m_type) + "{";
	description += "<compressed> x " + std::to_string(GetCellCount());
	return description + "}";
}

std::string ListCompressedBuffer::DebugDump() const
{
	std::string string = "[";
	string += " < Data offset = " + std::to_string(m_values->GetGPUReadBuffer()->GetGPUBuffer()) + " >\n";
	string += " - Data addresses: " + m_dataAddresses->DebugDump() + "\n";
	string += " - Data: " + m_values->DebugDump() + "\n";
	string += " < Size offset = " + std::to_string(m_sizes->GetGPUReadBuffer()->GetGPUBuffer()) + " >\n";
	string += " - Size addresses: " + m_sizeAddresses->DebugDump() + "\n";
	string += " - Sizes: " + m_sizes->DebugDump() + "\n";
	return string + "]";
}

void ListCompressedBuffer::Clear(ClearMode mode)
{
	m_values->Clear(mode);
}

}
