#include "Runtime/DataBuffers/ListCellBuffer.h"

#include "CUDA/BufferManager.h"

#include "HorseIR/Analysis/Shape/ShapeUtils.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "Runtime/DataBuffers/BufferUtils.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Runtime {

ListCellBuffer *ListCellBuffer::CreateEmpty(const HorseIR::ListType *type, const HorseIR::Analysis::ListShape *shape)
{
	auto elementTypes = type->GetElementTypes();
	auto elementShapes = shape->GetElementShapes();

	auto typeCount = elementTypes.size();
	auto shapeCount = elementShapes.size();

	if (const auto listSize = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(shape->GetListSize()))
	{
		shapeCount = listSize->GetValue();
	}

	if (typeCount != 1 && typeCount != shapeCount)
	{
		Utils::Logger::LogError("Mismatch between list type and shape cell count [" + HorseIR::PrettyPrinter::PrettyString(type) + "; " + HorseIR::Analysis::ShapeUtils::ShapeString(shape) + "]");
	}

	std::vector<DataBuffer *> cellBuffers;
	for (auto i = 0u; i < shapeCount; ++i)
	{
		auto elementType = (typeCount == 1) ? elementTypes.at(0) : elementTypes.at(i);
		auto elementShape = (elementShapes.size() == 1) ? elementShapes.at(0) : elementShapes.at(i);

		cellBuffers.push_back(DataBuffer::CreateEmpty(elementType, elementShape));
	}

	return new ListCellBuffer(cellBuffers);
}

ListCellBuffer::ListCellBuffer(const std::vector<DataBuffer *>& cells) : m_cells(cells)
{
	std::vector<HorseIR::Type *> cellTypes;
	std::vector<const HorseIR::Analysis::Shape *> cellShapes;
	for (const auto& cell : cells)
	{
		cellTypes.push_back(cell->GetType()->Clone());
		cellShapes.push_back(cell->GetShape());
	}
	m_type = new HorseIR::ListType(cellTypes);
	m_shape = new HorseIR::Analysis::ListShape(new HorseIR::Analysis::Shape::ConstantSize(cells.size()), cellShapes);

	m_cpuConsistent = true; // Always CPU consistent
}

ListCellBuffer::~ListCellBuffer()
{
	delete m_gpuBuffer;
	delete m_gpuSizeBuffer;

	delete m_gpuDataPointers;
	delete m_gpuSizePointers;
}

ListCellBuffer *ListCellBuffer::Clone() const
{
	std::vector<DataBuffer *> cells;
	for (const auto cell : m_cells)
	{
		cells.push_back(cell->Clone());
	}
	return new ListCellBuffer(cells);
}

void ListCellBuffer::SetTag(const std::string& tag)
{
	ListBuffer::SetTag(tag);

	auto i = 0u;
	for (auto cell : m_cells)
	{
		cell->SetTag((tag == "") ? "" : tag + "_" + std::to_string(i));
	}

	if (IsAllocatedOnGPU())
	{
		m_gpuBuffer->SetTag((tag == "") ? "" : tag + "_list");
		m_gpuSizeBuffer->SetTag((tag == "") ? "" : + "_list_size");
	}
}

void ListCellBuffer::ResizeCells(unsigned int size)
{
	auto oldDescription = Description();
	auto changed = false;

	for (auto cell : m_cells)
	{
		if (auto vectorBuffer = BufferUtils::GetBuffer<VectorBuffer>(cell, false))
		{
			changed |= vectorBuffer->Resize(size);
		}
		else
		{
			Utils::Logger::LogError("List buffer resize may only apply vector cells, received " + cell->Description());
		}
	}

	if (changed)
	{
		// Invalidate the GPU content as the cell buffers may have been reallocated

		SetCPUConsistent(true);

		// Propagate shape change

		delete m_shape;

		std::vector<const HorseIR::Analysis::Shape *> cellShapes;
		for (const auto& cell : m_cells)
		{
			cellShapes.push_back(cell->GetShape());
		}
		m_shape = new HorseIR::Analysis::ListShape(new HorseIR::Analysis::Shape::ConstantSize(m_cells.size()), cellShapes);

		if (Utils::Options::IsDebug_Print())
		{
			Utils::Logger::LogDebug("Resized list buffer [" + oldDescription + "] to [" + Description() + "]");
		}
	}
}

void ListCellBuffer::RequireCPUConsistent(bool exclusive) const
{
	auto timeStart = Utils::Chrono::Start(TransferString("CPU list"));

	for (const auto buffer : m_cells)
	{
		buffer->RequireCPUConsistent(exclusive);
	}
	DataBuffer::RequireCPUConsistent(exclusive);

	Utils::Chrono::End(timeStart);
}

void ListCellBuffer::RequireGPUConsistent(bool exclusive) const
{
	auto timeStart = Utils::Chrono::Start(TransferString("GPU list"));

	for (const auto buffer : m_cells)
	{
		buffer->RequireGPUConsistent(exclusive);
	}
	DataBuffer::RequireGPUConsistent(exclusive);

	Utils::Chrono::End(timeStart);
}

CUDA::Buffer *ListCellBuffer::GetGPUWriteBuffer()
{
	RequireGPUConsistent(true);
	return m_gpuBuffer;
}

const CUDA::Buffer *ListCellBuffer::GetGPUReadBuffer() const
{
	RequireGPUConsistent(false);
	return m_gpuBuffer;
}

const CUDA::Buffer *ListCellBuffer::GetGPUSizeBuffer() const
{
	RequireGPUConsistent(false);
	return m_gpuSizeBuffer;
}

CUDA::Buffer *ListCellBuffer::GetGPUSizeBuffer()
{
	RequireGPUConsistent(false);
	return m_gpuSizeBuffer;
}

size_t ListCellBuffer::GetGPUBufferSize() const
{
	return (sizeof(CUdeviceptr) * m_cells.size());
}

bool ListCellBuffer::ReallocateGPUBuffer()
{
	// Resize all cells in the list individually

	auto oldDescription = Description();
	auto changed = false;

	for (auto cell : m_cells)
	{
		changed |= cell->ReallocateGPUBuffer();
	}

	if (changed)
	{
		// Invalidate the GPU content as the cell buffers may have been reallocated

		SetCPUConsistent(true);

		// Propagate shape change

		delete m_shape;

		std::vector<const HorseIR::Analysis::Shape *> cellShapes;
		for (const auto& cell : m_cells)
		{
			cellShapes.push_back(cell->GetShape());
		}
		m_shape = new HorseIR::Analysis::ListShape(new HorseIR::Analysis::Shape::ConstantSize(m_cells.size()), cellShapes);

		if (Utils::Options::IsDebug_Print())
		{
			Utils::Logger::LogDebug("Resized list buffer [" + oldDescription + "] to [" + Description() + "]");
		}
	}
	return changed;
}

std::string ListCellBuffer::Description() const
{
	std::string description = HorseIR::PrettyPrinter::PrettyString(m_type) + "{";
	bool first = true;
	for (const auto& object : m_cells)
	{
		if (!first)
		{
			description += ", ";
		}
		first = false;
		description += object->Description();
	}
	return description + "}";
}

std::string ListCellBuffer::DebugDump(unsigned int indent, bool preindent) const
{
	std::string indentString(indent * Utils::Logger::IndentSize, ' ');
	std::string indentStringP1((indent + 1) * Utils::Logger::IndentSize, ' ');

	std::string string;
	if (!preindent)
	{
		string += indentString;
	}
       
	string += "[";
	if (m_cells.size() > 0)
	{
		string += "\n";

		auto index = 0u;
		for (const auto& cell : m_cells)
		{
			string += indentStringP1 + "[" + std::to_string(index) + "] ";
			string += cell->DebugDump(indent + 1, true);
			string += "\n";

			index++;
		}
		string += indentString;
	}
	return string + "]";
}

void ListCellBuffer::Clear(ClearMode mode)
{
	for (auto i = 0u; i < m_cells.size(); ++i)
	{
		m_cells.at(i)->Clear(mode);
	}
}

void ListCellBuffer::AllocateGPUBuffer() const
{
	auto cellCount = m_cells.size();
	size_t bufferSize = cellCount * sizeof(CUdeviceptr);

	m_gpuDataPointers = new CUdeviceptr[bufferSize];
	m_gpuSizePointers = new CUdeviceptr[bufferSize];

	for (auto i = 0u; i < cellCount; ++i)
	{
		auto buffer = m_cells.at(i);
		if (auto vectorBuffer = BufferUtils::GetBuffer<VectorBuffer>(buffer, false))
		{
			m_gpuDataPointers[i] = vectorBuffer->GetGPUReadBuffer()->GetGPUBuffer();
			m_gpuSizePointers[i] = vectorBuffer->GetGPUSizeBuffer()->GetGPUBuffer();
		}
		else
		{
			Utils::Logger::LogError("GPU list buffers may only have vector cells, received " + buffer->Description());
		}
	}

	// Data

	m_gpuBuffer = CUDA::BufferManager::CreateBuffer(bufferSize);
	m_gpuBuffer->AllocateOnGPU();
	m_gpuBuffer->SetCPUBuffer(m_gpuDataPointers);
	m_gpuBuffer->TransferToGPU();

	// Size

	m_gpuSizeBuffer = CUDA::BufferManager::CreateBuffer(bufferSize);
	m_gpuSizeBuffer->AllocateOnGPU();
	m_gpuSizeBuffer->SetCPUBuffer(m_gpuSizePointers);
	m_gpuSizeBuffer->TransferToGPU();

	// Tags

	if (m_tag != "")
	{
		m_gpuBuffer->SetTag(m_tag + "_list");
		m_gpuSizeBuffer->SetTag(m_tag + "_list_size");
	}
}

void ListCellBuffer::TransferToGPU() const
{
	// Init

	auto cellCount = m_cells.size();
	for (auto i = 0u; i < cellCount; ++i)
	{
		auto buffer = m_cells.at(i);
		if (auto vectorBuffer = BufferUtils::GetBuffer<VectorBuffer>(buffer, false))
		{
			m_gpuDataPointers[i] = vectorBuffer->GetGPUReadBuffer()->GetGPUBuffer();
			m_gpuSizePointers[i] = vectorBuffer->GetGPUSizeBuffer()->GetGPUBuffer();
		}
		else
		{
			Utils::Logger::LogError("GPU list buffers may only have vector cells, received " + buffer->Description());
		}
	}

	// Data

	m_gpuBuffer->TransferToGPU();
	m_gpuSizeBuffer->TransferToGPU();
}

}
