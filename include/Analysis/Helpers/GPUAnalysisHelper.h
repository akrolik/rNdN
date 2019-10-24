#pragma once

#include <utility>
#include <vector>

#include "HorseIR/Traversal/ConstVisitor.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

class GPUAnalysisHelper : public HorseIR::ConstVisitor
{
public:
	void Analyze(const HorseIR::Expression *expression);
	void Analyze(const HorseIR::Statement *statement);

	bool IsCapable() const { return m_capable; }
	bool IsSynchronizedIn(unsigned int index) const { return m_synchronizedIn.at(index); }
	bool IsSynchronizedOut() const { return m_synchronizedOut; }

	void Visit(const HorseIR::DeclarationStatement *declarationS) override;
	void Visit(const HorseIR::AssignStatement *assignS) override;
	void Visit(const HorseIR::ExpressionStatement *expressionS) override;
	void Visit(const HorseIR::ReturnStatement *returnS) override;

	void Visit(const HorseIR::CallExpression *call) override;
	void Visit(const HorseIR::CastExpression *cast) override;
	void Visit(const HorseIR::Literal *literal) override;
	void Visit(const HorseIR::Identifier *identifier) override;

private:
	std::pair<bool, bool> AnalyzeCall(const HorseIR::FunctionDeclaration *function, const std::vector<HorseIR::Operand *>& arguments);
	std::pair<bool, bool> AnalyzeCall(const HorseIR::Function *function, const std::vector<HorseIR::Operand *>& arguments);
	std::pair<bool, bool> AnalyzeCall(const HorseIR::BuiltinFunction *function, const std::vector<HorseIR::Operand *>& arguments);

	bool m_capable = false;
	std::vector<bool> m_synchronizedIn;
	bool m_synchronizedOut = false;
};

}
