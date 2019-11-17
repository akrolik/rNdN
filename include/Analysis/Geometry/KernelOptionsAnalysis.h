#pragma once

#include "Analysis/Shape/ShapeAnalysis.h"

#include "Codegen/InputOptions.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

class KernelOptionsAnalysis
{
public:
	void Analyze(const HorseIR::Function *function, const ShapeAnalysis& shapeAnalysis);

	Codegen::InputOptions *GetInputOptions() const { return m_inputOptions; }

private:
	std::uint32_t GetAverageCellSize(const Analysis::ListShape *shape) const;

	Codegen::InputOptions *m_inputOptions = nullptr;
};

}
