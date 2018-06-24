#pragma once

#include <unordered_map>

#include "HorseIR/Traversal/ForwardTraversal.h"

#include "HorseIR/Tree/Method.h"
#include "HorseIR/Tree/Parameter.h"
#include "HorseIR/Tree/Expressions/Identifier.h"
#include "HorseIR/Tree/Statements/AssignStatement.h"
#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class SymbolTable : public ForwardTraversal
{
public:
	using ForwardTraversal::ForwardTraversal;

	void Build(HorseIR::Method *method);
	HorseIR::Type *GetType(const std::string& identifier);

	void Visit(Parameter *parameter) override;
	void Visit(AssignStatement *assign) override;
	void Visit(Identifier *identifier) override;

private:
	std::unordered_map<std::string, HorseIR::Type *> m_table;
};

}
