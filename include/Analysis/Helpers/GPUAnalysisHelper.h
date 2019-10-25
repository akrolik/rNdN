#pragma once

#include <tuple>

#include "HorseIR/Traversal/ConstVisitor.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

class GPUAnalysisHelper : public HorseIR::ConstVisitor
{
public:
	bool IsGPU(const HorseIR::Statement *statement);
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
	std::tuple<bool, bool, bool> AnalyzeCall(const HorseIR::FunctionDeclaration *function, const std::vector<HorseIR::Operand *>& arguments, unsigned int index);
	std::tuple<bool, bool, bool> AnalyzeCall(const HorseIR::Function *function, const std::vector<HorseIR::Operand *>& arguments, unsigned int index);
	std::tuple<bool, bool, bool> AnalyzeCall(const HorseIR::BuiltinFunction *function, const std::vector<HorseIR::Operand *>& arguments, unsigned int index);

	bool m_gpu = false;
	bool m_synchronizedIn = false;
	bool m_synchronizedOut = false;
	unsigned int m_index = 0;
};

}
