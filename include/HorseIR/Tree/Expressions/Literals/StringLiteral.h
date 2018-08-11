#pragma once

#include "HorseIR/Tree/Expressions/Literals/Literal.h"

#include <string>
#include <vector>

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Types/BasicType.h"

namespace HorseIR {

class StringLiteral : public Literal<std::string>
{
public:
	StringLiteral(const std::string& value) : Literal<std::string>(value, new BasicType(BasicType::Kind::String)) {}
	StringLiteral(const std::vector<std::string>& values) : Literal<std::string>(values, new BasicType(BasicType::Kind::String)) {}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }
};

}
