#include "Runtime/DataBuffers/ListCompressedBuffer.h"

#include "HorseIR/Utils/PrettyPrinter.h"

namespace Runtime {

ListCompressedBuffer::ListCompressedBuffer(const TypedVectorBuffer<std::int32_t> *sizes, VectorBuffer *values) : m_sizes(sizes), m_values(values)
{
	// Type and shape

	const auto& cellSizes = sizes->GetCPUReadBuffer()->GetValues();

	m_type = new HorseIR::ListType(m_values->GetType()->Clone());
	m_shape = new Analysis::ListShape(
			new Analysis::Shape::ConstantSize(cellSizes.size()),
			{new Analysis::VectorShape(new Analysis::Shape::RangedSize(cellSizes))}
	);

	// Construct cell addresses on GPU

	auto compressedSize = cellSizes.size();

	auto dataAddressesBuffer = new TypedVectorBuffer<CUdeviceptr>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), compressedSize);
	auto sizeAddressesBuffer = new TypedVectorBuffer<CUdeviceptr>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), compressedSize);

	auto dataOffset = m_values->GetGPUReadBuffer()->GetGPUBuffer();
	auto sizeOffset = m_sizes->GetGPUReadBuffer()->GetGPUBuffer();

	auto dataAddresses = dataAddressesBuffer->GetCPUWriteBuffer();
	auto sizeAddresses = sizeAddressesBuffer->GetCPUWriteBuffer();

	auto timeOffsets_start = Utils::Chrono::Start("Compute offsets");

	auto offset = 0u;
	auto index = 0u;
	for (auto size : cellSizes)
	{
		dataAddresses->SetValue(index, dataOffset + offset * m_values->GetElementSize());
		sizeAddresses->SetValue(index, sizeOffset + index * m_sizes->GetElementSize());

		index++;
		offset += size;
	}

	Utils::Chrono::End(timeOffsets_start);

	m_dataAddresses = dataAddressesBuffer;
	m_sizeAddresses = sizeAddressesBuffer;
}

ListCompressedBuffer::~ListCompressedBuffer()
{
}

ListCompressedBuffer *ListCompressedBuffer::Clone() const
{
	return new ListCompressedBuffer(m_sizes->Clone(), m_values->Clone());
}

const std::vector<DataBuffer *>& ListCompressedBuffer::GetCells() const
{
	AllocateCells();
	return m_cells;
}

DataBuffer *ListCompressedBuffer::GetCell(unsigned int index) const
{
	AllocateCells();
	return m_cells.at(index);
}

size_t ListCompressedBuffer::GetCellCount() const
{
	return m_dataAddresses->GetElementCount();
}

void ListCompressedBuffer::ValidateCPU(bool recursive) const
{
	DataBuffer::ValidateCPU(recursive);
	if (recursive)
	{
		m_values->ValidateCPU();
	}
}

void ListCompressedBuffer::ValidateGPU(bool recursive) const
{
	DataBuffer::ValidateGPU(recursive);
	if (recursive)
	{
		m_values->ValidateGPU();
	}
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

CUDA::Buffer *ListCompressedBuffer::GetGPUSizeBuffer() const
{
	m_sizes->ValidateGPU();
	return m_sizeAddresses->GetGPUReadBuffer();
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

void ListCompressedBuffer::AllocateCells() const
{
	if (m_cells.size() == 0)
	{
		auto offset = 0u;
		for (auto size : m_sizes->GetCPUReadBuffer()->GetValues())
		{
			m_cells.push_back(m_values->Slice(offset, size));
			offset += size;
		}
	}
}

}
