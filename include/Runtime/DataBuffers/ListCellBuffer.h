#pragma once

#include "Runtime/DataBuffers/ListBuffer.h"

namespace Runtime {

class ListCellBuffer : public ListBuffer
{
public:
	static ListCellBuffer *CreateEmpty(const HorseIR::ListType *type, const HorseIR::Analysis::ListShape *shape);

	ListCellBuffer(DataBuffer *cell) : ListCellBuffer(std::vector<DataBuffer *>({cell})) {}
	ListCellBuffer(const std::vector<DataBuffer *>& cells);
	~ListCellBuffer() override;
	
	ListCellBuffer *Clone() const override;

	// Tag

	void SetTag(const std::string& tag) override;

	// Cells

	std::vector<const DataBuffer *> GetCells() const override { return { std::begin(m_cells), std::end(m_cells) }; }
	std::vector<DataBuffer *>& GetCells() override { return m_cells; }

	const DataBuffer *GetCell(unsigned int index) const override { return m_cells.at(index); }
	DataBuffer *GetCell(unsigned int index) override { return m_cells.at(index); }

	size_t GetCellCount() const override { return m_cells.size(); }

	// Sizing

	void ResizeCells(unsigned int size) override;

	// CPU/GPU management

	void RequireCPUConsistent(bool exclusive) const override;
	void RequireGPUConsistent(bool exclusive) const override;

	CUDA::Buffer *GetGPUWriteBuffer() override;
	const CUDA::Buffer *GetGPUReadBuffer() const override;

	const CUDA::Buffer *GetGPUSizeBuffer() const override;
	CUDA::Buffer *GetGPUSizeBuffer() override;

	size_t GetGPUBufferSize() const override;

	bool ReallocateGPUBuffer() override;

	// Printers

	std::string Description() const override;
	std::string DebugDump(unsigned int indent = 0, bool preindent = false) const override;

	// Clear

	void Clear(ClearMode mode = ClearMode::Zero) override;

private:
	// CPU/GPU management

	bool IsAllocatedOnCPU() const override { return true; }
	bool IsAllocatedOnGPU() const override { return (m_gpuBuffer != nullptr); }

	void AllocateCPUBuffer() const override {} // Do nothing
	void AllocateGPUBuffer() const override;

	void TransferToCPU() const override {} // Always consistent
	void TransferToGPU() const override;

	// Data

	std::vector<DataBuffer *> m_cells;

	mutable CUDA::Buffer *m_gpuBuffer = nullptr;
	mutable CUDA::Buffer *m_gpuSizeBuffer = nullptr;

	mutable CUdeviceptr *m_gpuDataPointers = nullptr;
	mutable CUdeviceptr *m_gpuSizePointers = nullptr;
};

}
