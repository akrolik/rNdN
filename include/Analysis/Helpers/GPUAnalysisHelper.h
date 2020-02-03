#pragma once

#include <utility>

#include "HorseIR/Traversal/ConstVisitor.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

class GPUAnalysisHelper : public HorseIR::ConstVisitor
{
public:
	std::pair<bool, bool> IsGPU(const HorseIR::Statement *statement);
	bool IsSynchronized(const HorseIR::Statement *source, const HorseIR::Statement *destination, unsigned int index);

	void Visit(const HorseIR::DeclarationStatement *declarationS) override;
	void Visit(const HorseIR::AssignStatement *assignS) override;
	void Visit(const HorseIR::ExpressionStatement *expressionS) override;
	void Visit(const HorseIR::ReturnStatement *returnS) override;

	void Visit(const HorseIR::CallExpression *call) override;
	void Visit(const HorseIR::CastExpression *cast) override;
	void Visit(const HorseIR::Literal *literal) override;
	void Visit(const HorseIR::Identifier *identifier) override;

private:
	enum Device {
		CPU,
		GPU,
		GPULibrary
	};

	enum Synchronization {
		None       = 0,
		In         = (1 << 0),
		Out        = (1 << 1),
		Reduction  = (1 << 2),
		Raze       = (1 << 3)
	};

	friend Synchronization operator|(Synchronization a, Synchronization b);

	std::pair<Device, Synchronization> AnalyzeCall(const HorseIR::FunctionDeclaration *function, const std::vector<HorseIR::Operand *>& arguments, unsigned int index);
	std::pair<Device, Synchronization> AnalyzeCall(const HorseIR::Function *function, const std::vector<HorseIR::Operand *>& arguments, unsigned int index);
	std::pair<Device, Synchronization> AnalyzeCall(const HorseIR::BuiltinFunction *function, const std::vector<HorseIR::Operand *>& arguments, unsigned int index);

	unsigned int m_index = 0;
	Device m_device = Device::CPU;
	Synchronization m_synchronization = Synchronization::None;
};

inline GPUAnalysisHelper::Synchronization operator|(GPUAnalysisHelper::Synchronization a, GPUAnalysisHelper::Synchronization b)
{
	return static_cast<GPUAnalysisHelper::Synchronization>(static_cast<int>(a) | static_cast<int>(b));
}

}
