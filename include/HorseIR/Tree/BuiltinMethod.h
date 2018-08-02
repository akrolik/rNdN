#pragma once

#include "HorseIR/Tree/MethodDeclaration.h"

namespace HorseIR {

class BuiltinMethod : public MethodDeclaration
{
public:
	BuiltinMethod(const std::string& name) : MethodDeclaration(MethodDeclaration::Kind::Builtin, name) {}

	std::string SignatureString() const override
	{
		return "def " + m_name + "() BUILTIN";
	}

	std::string ToString() const override
	{
		return SignatureString();
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
};

}
