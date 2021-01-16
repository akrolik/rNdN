#pragma once

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Declarations/VariableDeclaration.h"
#include "PTX/Tree/Operands/Variables/Variable.h"

namespace PTX {

template<class T, bool Assert = true>
using ParameterVariable = Variable<T, ParameterSpace, Assert>;

DispatchInterface(ParameterVariable)

template<class T, bool Assert>
class Variable<T, ParameterSpace, Assert> : DispatchInherit(ParameterVariable), public VariableBase<T, ParameterSpace, Assert>
{
	friend class TypedVariableDeclaration<T, ParameterSpace>;
public:
	// Visitors

	void Accept(OperandVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstOperandVisitor& visitor) const override { visitor.Visit(this); }

protected:
	using VariableBase<T, ParameterSpace, Assert>::VariableBase;

	DispatchMember_Type(T);
};

DispatchImplementation(ParameterVariable)

}
