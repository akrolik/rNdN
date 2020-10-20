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

	const std::vector<DataBuffer *>& GetCells() const override { return m_cells; }
	DataBuffer *GetCell(unsigned int index) const override { return m_cells.at(index); }
	size_t GetCellCount() const override { return m_cells.size(); }

	// Sizing

	void ResizeCells(unsigned int size) override;

	// CPU/GPU management

	void ValidateCPU() const override;
	void ValidateGPU() const override;

	CUDA::Buffer *GetGPUWriteBuffer() override;
	CUDA::Buffer *GetGPUReadBuffer() const override;
	CUDA::Buffer *GetGPUSizeBuffer() const override;

	size_t GetGPUBufferSize() const override;

	bool ReallocateGPUBuffer() override;

	// Printers

	std::string Description() const override;
	std::string DebugDump() const override;

	// Clear

	void Clear(ClearMode mode = ClearMode::Zero) override;

private:
	bool IsAllocatedOnCPU() const override { return true; }
	bool IsAllocatedOnGPU() const override { return (m_gpuBuffer != nullptr); }

	void AllocateCPUBuffer() const override {} // Do nothing
	void AllocateGPUBuffer() const override;

	void TransferToCPU() const override {} // Always consistent
	void TransferToGPU() const override;

	std::vector<DataBuffer *> m_cells;

	mutable CUDA::Buffer *m_gpuBuffer = nullptr;
	mutable CUDA::Buffer *m_gpuSizeBuffer = nullptr;

	mutable CUdeviceptr *m_gpuDataPointers = nullptr;
	mutable CUdeviceptr *m_gpuSizePointers = nullptr;
};

}
