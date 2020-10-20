#pragma once

#include <utility>

#include "HorseIR/Traversal/ConstVisitor.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {
namespace Analysis {

class GPUAnalysisHelper : public ConstVisitor
{
public:
	enum Device {
		CPU,
		GPU,
		GPULibrary
	};

	Device IsGPU(const Statement *statement);
	bool IsSynchronized(const Statement *source, const Statement *destination, unsigned int index);

	void Visit(const DeclarationStatement *declarationS) override;
	void Visit(const AssignStatement *assignS) override;
	void Visit(const ExpressionStatement *expressionS) override;
	void Visit(const ReturnStatement *returnS) override;

	void Visit(const CallExpression *call) override;
	void Visit(const CastExpression *cast) override;
	void Visit(const Literal *literal) override;
	void Visit(const Identifier *identifier) override;

private:
	enum Synchronization {
		None       = 0,
		In         = (1 << 0),
		Out        = (1 << 1),
		Reduction  = (1 << 2),
		Raze       = (1 << 3)
	};

	friend Synchronization operator|(Synchronization a, Synchronization b);

	std::pair<Device, Synchronization> AnalyzeCall(const FunctionDeclaration *function, const std::vector<Operand *>& arguments, unsigned int index);
	std::pair<Device, Synchronization> AnalyzeCall(const Function *function, const std::vector<Operand *>& arguments, unsigned int index);
	std::pair<Device, Synchronization> AnalyzeCall(const BuiltinFunction *function, const std::vector<Operand *>& arguments, unsigned int index);

	unsigned int m_index = 0;
	Device m_device = Device::CPU;
	Synchronization m_synchronization = Synchronization::None;
};

inline GPUAnalysisHelper::Synchronization operator|(GPUAnalysisHelper::Synchronization a, GPUAnalysisHelper::Synchronization b)
{
	return static_cast<GPUAnalysisHelper::Synchronization>(static_cast<int>(a) | static_cast<int>(b));
}

}
}
