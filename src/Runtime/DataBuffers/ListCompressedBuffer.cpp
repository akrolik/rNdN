#include "Runtime/DataBuffers/ListCompressedBuffer.h"

#include "CUDA/Constant.h"
#include "CUDA/KernelInvocation.h"

#include "Runtime/Runtime.h"

#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Chrono.h"

namespace Runtime {

ListCompressedBuffer *ListCompressedBuffer::CreateEmpty(const HorseIR::BasicType *type, const HorseIR::Analysis::Shape::RangedSize *size)
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
	m_shape = new HorseIR::Analysis::ListShape(
			new HorseIR::Analysis::Shape::ConstantSize(cellSizes.size()),
			{new HorseIR::Analysis::VectorShape(new HorseIR::Analysis::Shape::RangedSize(cellSizes))}
	);

	m_gpuConsistent = true;
}

ListCompressedBuffer::ListCompressedBuffer(TypedVectorBuffer<std::int32_t> *sizes, VectorBuffer *values) : m_sizes(sizes), m_values(values)
{
	// Type and shape

	const auto& cellSizes = sizes->GetCPUReadBuffer()->GetValues();

	m_type = new HorseIR::ListType(m_values->GetType()->Clone());
	m_shape = new HorseIR::Analysis::ListShape(
			new HorseIR::Analysis::Shape::ConstantSize(cellSizes.size()),
			{new HorseIR::Analysis::VectorShape(new HorseIR::Analysis::Shape::RangedSize(cellSizes))}
	);

	// Construct cell addresses on GPU

	auto compressedSize = cellSizes.size();

	auto dataAddressesBuffer = new TypedVectorBuffer<CUdeviceptr>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), compressedSize);
	auto sizeAddressesBuffer = new TypedVectorBuffer<CUdeviceptr>(new HorseIR::BasicType(HorseIR::BasicType::BasicKind::Int64), compressedSize);

	auto dataOffset = m_values->GetGPUBufferAddress();
	auto sizeOffset = m_sizes->GetGPUBufferAddress();

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

	m_gpuConsistent = true;
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

	m_sizes->SetTag((tag == "") ? "" : tag + "_list_size");
	m_values->SetTag((tag == "") ? "" : tag + "_list_values");
}

std::vector<const DataBuffer *> ListCompressedBuffer::GetCells() const
{
	AllocateCells();
	return { std::begin(m_cells), std::end(m_cells) };
}

std::vector<DataBuffer *>& ListCompressedBuffer::GetCells()
{
	AllocateCells();
	return m_cells;
}

const DataBuffer *ListCompressedBuffer::GetCell(unsigned int index) const
{
	AllocateCells();
	return m_cells.at(index);
}

DataBuffer *ListCompressedBuffer::GetCell(unsigned int index)
{
	AllocateCells();
	return m_cells.at(index);
}

size_t ListCompressedBuffer::GetCellCount() const
{
	return m_dataAddresses->GetElementCount();
}

void ListCompressedBuffer::RequireCPUConsistent(bool exclusive) const
{
	auto timeStart = Utils::Chrono::Start(TransferString("CPU list"));

	m_values->RequireCPUConsistent(exclusive);
	SetCPUConsistent(exclusive);

	Utils::Chrono::End(timeStart);
}

void ListCompressedBuffer::RequireGPUConsistent(bool exclusive) const
{
	auto timeStart = Utils::Chrono::Start(TransferString("GPU list"));

	m_values->RequireGPUConsistent(exclusive);
	SetGPUConsistent(exclusive);

	Utils::Chrono::End(timeStart);
}

CUDA::Buffer *ListCompressedBuffer::GetGPUWriteBuffer()
{
	RequireGPUConsistent(true);
	return m_dataAddresses->GetGPUWriteBuffer();
}

const CUDA::Buffer *ListCompressedBuffer::GetGPUReadBuffer() const
{
	RequireGPUConsistent(false);
	return m_dataAddresses->GetGPUReadBuffer();
}

const CUDA::Buffer *ListCompressedBuffer::GetGPUSizeBuffer() const
{
	m_sizes->RequireGPUConsistent(false);
	return m_sizeAddresses->GetGPUReadBuffer();
}

CUDA::Buffer *ListCompressedBuffer::GetGPUSizeBuffer()
{
	m_sizes->RequireGPUConsistent(false);
	return m_sizeAddresses->GetGPUWriteBuffer();
}

std::string ListCompressedBuffer::Description() const
{
	std::string description = HorseIR::PrettyPrinter::PrettyString(m_type) + "{";
	description += "<compressed> x " + std::to_string(GetCellCount());
	return description + "}";
}

std::string ListCompressedBuffer::DebugDump(unsigned int indent, bool preindent) const
{
	std::string indentString(indent * Utils::Logger::IndentSize, ' ');
	std::string indentStringP1((indent + 1) * Utils::Logger::IndentSize, ' ');

	std::string string;
	if (!preindent)
	{
		string += indentString;
	}
       
	string += "[\n";
	string += indentStringP1 + " < Data offset = " + std::to_string(m_values->GetGPUReadBuffer()->GetGPUBuffer()) + " >\n";
	string += indentStringP1 + " - Data addresses: " + m_dataAddresses->DebugDump() + "\n";
	string += indentStringP1 + " - Data: " + m_values->DebugDump() + "\n";
	string += indentStringP1 + " < Size offset = " + std::to_string(m_sizes->GetGPUReadBuffer()->GetGPUBuffer()) + " >\n";
	string += indentStringP1 + " - Size addresses: " + m_sizeAddresses->DebugDump() + "\n";
	string += indentStringP1 + " - Sizes: " + m_sizes->DebugDump() + "\n";
	return string + indentString + "]";
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
