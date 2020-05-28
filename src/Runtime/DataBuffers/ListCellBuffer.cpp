#include "Runtime/DataBuffers/ListCellBuffer.h"

#include "Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Utils/PrettyPrinter.h"

#include "Runtime/DataBuffers/BufferUtils.h"
#include "Runtime/DataBuffers/VectorBuffer.h"

#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace Runtime {

ListCellBuffer *ListCellBuffer::CreateEmpty(const HorseIR::ListType *type, const Analysis::ListShape *shape)
{
	auto elementTypes = type->GetElementTypes();
	auto elementShapes = shape->GetElementShapes();

	auto typeCount = elementTypes.size();
	auto shapeCount = elementShapes.size();

	if (const auto listSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(shape->GetListSize()))
	{
		shapeCount = listSize->GetValue();
	}

	if (typeCount != 1 && typeCount != shapeCount)
	{
		Utils::Logger::LogError("Mismatch between list type and shape cell count [" + HorseIR::PrettyPrinter::PrettyString(type) + "; " + Analysis::ShapeUtils::ShapeString(shape) + "]");
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
	std::vector<const Analysis::Shape *> cellShapes;
	for (const auto& cell : cells)
	{
		cellTypes.push_back(cell->GetType()->Clone());
		cellShapes.push_back(cell->GetShape());
	}
	m_type = new HorseIR::ListType(cellTypes);
	m_shape = new Analysis::ListShape(new Analysis::Shape::ConstantSize(cells.size()), cellShapes);

	m_cpuConsistent = true; // Always CPU consistent
}

ListCellBuffer::~ListCellBuffer()
{
	delete m_gpuBuffer;
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

		InvalidateGPU();

		// Propagate shape change

		delete m_shape;

		std::vector<const Analysis::Shape *> cellShapes;
		for (const auto& cell : m_cells)
		{
			cellShapes.push_back(cell->GetShape());
		}
		m_shape = new Analysis::ListShape(new Analysis::Shape::ConstantSize(m_cells.size()), cellShapes);

		if (Utils::Options::Present(Utils::Options::Opt_Print_debug))
		{
			Utils::Logger::LogDebug("Resized list buffer [" + oldDescription + "] to [" + Description() + "]");
		}
	}
}

void ListCellBuffer::ValidateCPU(bool recursive) const
{
	DataBuffer::ValidateCPU(recursive);
	if (recursive)
	{
		for (const auto buffer : m_cells)
		{
			buffer->ValidateCPU(true);
		}
	}
}

void ListCellBuffer::ValidateGPU(bool recursive) const
{
	DataBuffer::ValidateGPU(recursive);
	if (recursive)
	{
		for (const auto buffer : m_cells)
		{
			buffer->ValidateGPU(true);
		}
	}
}

CUDA::Buffer *ListCellBuffer::GetGPUWriteBuffer()
{
	ValidateGPU();
	for (auto i = 0u; i < m_cells.size(); ++i)
	{
		m_cells.at(i)->GetGPUWriteBuffer();
	}
	return m_gpuBuffer;
}

CUDA::Buffer *ListCellBuffer::GetGPUReadBuffer() const
{
	ValidateGPU();
	for (auto i = 0u; i < m_cells.size(); ++i)
	{
		m_cells.at(i)->GetGPUReadBuffer();
	}
	return m_gpuBuffer;
}

size_t ListCellBuffer::GetGPUBufferSize() const
{
	return (sizeof(CUdeviceptr) * m_cells.size());
}

CUDA::Buffer *ListCellBuffer::GetGPUSizeBuffer() const
{
	ValidateGPU();
	return m_gpuSizeBuffer;
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

		InvalidateGPU();

		// Propagate shape change

		delete m_shape;

		std::vector<const Analysis::Shape *> cellShapes;
		for (const auto& cell : m_cells)
		{
			cellShapes.push_back(cell->GetShape());
		}
		m_shape = new Analysis::ListShape(new Analysis::Shape::ConstantSize(m_cells.size()), cellShapes);

		if (Utils::Options::Present(Utils::Options::Opt_Print_debug))
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

std::string ListCellBuffer::DebugDump() const
{
	std::string string = "[";
	bool first = true;
	for (const auto& cell : m_cells)
	{
		if (first)
		{
			string += "\n";
		}
		string += " - ";
		first = false;
		string += cell->DebugDump();
		string += "\n";
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

	m_gpuBuffer = new CUDA::Buffer(bufferSize);
	m_gpuBuffer->AllocateOnGPU();

	m_gpuSizeBuffer = new CUDA::Buffer(bufferSize);
	m_gpuSizeBuffer->AllocateOnGPU();
}

void ListCellBuffer::TransferToGPU() const
{
	auto cellCount = m_cells.size();
	size_t bufferSize = cellCount * sizeof(CUdeviceptr);

	if (m_gpuDataPointers == nullptr)
	{
		m_gpuDataPointers = new CUdeviceptr[bufferSize];
	}

	if (m_gpuSizePointers == nullptr)
	{
		m_gpuSizePointers = new CUdeviceptr[bufferSize];
	}

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

	m_gpuBuffer->SetCPUBuffer(m_gpuDataPointers);
	m_gpuBuffer->TransferToGPU();

	// Size

	m_gpuSizeBuffer->SetCPUBuffer(m_gpuSizePointers);
	m_gpuSizeBuffer->TransferToGPU();
}

}
