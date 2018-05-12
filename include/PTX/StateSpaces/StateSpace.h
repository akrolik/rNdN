#pragma once

#include "PTX/Statements/DirectiveStatement.h"
#include "PTX/Type.h"
#include "PTX/Operands/Variable.h"
#include "PTX/Operands/VariableSet.h"

namespace PTX {

template<class T>
class StateSpace : public DirectiveStatement
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	virtual std::string Specifier() const = 0;
	virtual std::string Directives() const { return ""; }

	std::string ToString() const
	{
		return "\t" + Specifier() + " " + T::Name() + " " + Directives() + VariableNames();
	}

protected:
	virtual std::string VariableNames() const = 0;
};

}
