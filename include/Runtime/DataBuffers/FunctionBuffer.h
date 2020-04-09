#pragma once

#include "Runtime/DataBuffers/DataBuffer.h"

#include "Analysis/Shape/Shape.h"

#include "HorseIR/Tree/Tree.h"

namespace Runtime {

class FunctionBuffer : public DataBuffer
{
public:
	constexpr static DataBuffer::Kind BufferKind = DataBuffer::Kind::Function;

	FunctionBuffer(const HorseIR::FunctionDeclaration *function);
	~FunctionBuffer() override;

	const HorseIR::FunctionType *GetType() const override { return m_type; }
	const Analysis::Shape *GetShape() const override { return m_shape; }

	// Keys/values

	const HorseIR::FunctionDeclaration *GetFunction() const { return m_function; }

	// Printers

	std::string Description() const override;
	std::string DebugDump() const override;

private:
	const HorseIR::FunctionType *m_type = nullptr;
	const Analysis::Shape *m_shape = nullptr;

	const HorseIR::FunctionDeclaration *m_function = nullptr;
};

}
