#include "Runtime/DataBuffers/ListCompressedBuffer.h"

#include "CUDA/Constant.h"
#include "CUDA/KernelInvocation.h"

#include "Runtime/Runtime.h"

#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Chrono.h"

namespace Runtime {

ListCompressedBuffer *ListCompressedBuffer::CreateEmpty(const HorseIR::BasicType *type, const Analysis::Shape::RangedSize *size)
{
	auto values = VectorBuffer::CreateEmpty(type, size);
	auto sizes = new TypedVectorBuffer(new TypedVectorData(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), size->GetValues()));
	return new ListCompressedBuffer(sizes, values);
}

ListCompressedBuffer::ListCompressedBuffer(const TypedVectorBuffer<std::int64_t> *offsets, VectorBuffer *values) : m_values(values)
{
	// Get the set utility kernel

	auto& gpuManager = Runtime::GetInstance()->GetGPUManager();
	auto libr3d3 = gpuManager.GetLibrary();

	auto kernel = libr3d3->GetKernel("init_list");
	CUDA::KernelInvocation invocation(kernel);

	// Configure the runtime thread layout

	const auto compressedSize = offsets->GetElementCount();

	const auto maxBlockSize = gpuManager.GetCurrentDevice()->GetMaxThreadsDimension(0);
	const auto blockSize = (compressedSize < maxBlockSize) ? compressedSize : maxBlockSize;
	const auto blockCount = (compressedSize + blockSize - 1) / blockSize;

	invocation.SetBlockShape(blockSize, 1, 1);
	invocation.SetGridShape(blockCount, 1, 1);

	auto dataAddressesBuffer = new TypedVectorBuffer<CUdeviceptr>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), compressedSize);
	auto sizeAddressesBuffer = new TypedVectorBuffer<CUdeviceptr>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), compressedSize);
	auto sizesBuffer = new TypedVectorBuffer<std::int32_t>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int32), compressedSize);

	CUDA::TypedConstant<std::uint32_t> sizeConstant(compressedSize);
	CUDA::TypedConstant<std::uint32_t> dataSizeConstant(values->GetElementCount());

	invocation.AddParameter(*dataAddressesBuffer->GetGPUWriteBuffer());
	invocation.AddParameter(*sizeAddressesBuffer->GetGPUWriteBuffer());
	invocation.AddParameter(*sizesBuffer->GetGPUWriteBuffer());

	invocation.AddParameter(*offsets->GetGPUReadBuffer());
	invocation.AddParameter(*values->GetGPUReadBuffer());
	invocation.AddParameter(sizeConstant);
	invocation.AddParameter(dataSizeConstant);

	invocation.SetDynamicSharedMemorySize(0);
	invocation.Launch();

	CUDA::Synchronize();

	m_dataAddresses = dataAddressesBuffer;
	m_sizeAddresses = sizeAddressesBuffer;
	m_sizes = sizesBuffer;

	// Type and shape

	const auto& cellSizes = m_sizes->GetCPUReadBuffer()->GetValues();

	m_type = new HorseIR::ListType(m_values->GetType()->Clone());
	m_shape = new Analysis::ListShape(
			new Analysis::Shape::ConstantSize(cellSizes.size()),
			{new Analysis::VectorShape(new Analysis::Shape::RangedSize(cellSizes))}
	);
}

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

void ListCompressedBuffer::SetTag(const std::string& tag)
{
	ListBuffer::SetTag(tag);

	//TODO: sizes
	// m_sizes->SetTag(tag + "_sizes");
	m_values->SetTag((tag == "") ? "" : tag + "_values");
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

void ListCompressedBuffer::ValidateCPU() const
{
	auto timeStart = Utils::Chrono::Start(TransferString("CPU"));

	DataBuffer::ValidateCPU();
	m_values->ValidateCPU();

	Utils::Chrono::End(timeStart);
}

void ListCompressedBuffer::ValidateGPU() const
{
	auto timeStart = Utils::Chrono::Start(TransferString("GPU"));

	DataBuffer::ValidateGPU();
	m_values->ValidateGPU();

	Utils::Chrono::End(timeStart);
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
