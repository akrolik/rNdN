#pragma once

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Declarations/VariableDeclaration.h"
#include "PTX/Tree/Operands/Variables/Variable.h"

namespace PTX {

template<class T, bool Assert = true>
using ConstVariable = Variable<T, ConstSpace, Assert>;

DispatchInterface(ConstVariable)

template<class T, bool Assert>
class Variable<T, ConstSpace, Assert> : DispatchInherit(ConstVariable), public VariableBase<T, ConstSpace, Assert>
{
	friend class TypedVariableDeclaration<T, ConstSpace>;
public:
	// Visitors

	void Accept(OperandVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstOperandVisitor& visitor) const override { visitor.Visit(this); }

protected:
	using VariableBase<T, ConstSpace, Assert>::VariableBase;

	DispatchMember_Type(T);
};

DispatchImplementation(ConstVariable)

}
