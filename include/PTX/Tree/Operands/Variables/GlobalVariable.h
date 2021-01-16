#pragma once

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Declarations/VariableDeclaration.h"
#include "PTX/Tree/Operands/Variables/Variable.h"

namespace PTX {

template<class T, bool Assert = true>
using GlobalVariable = Variable<T, GlobalSpace, Assert>;

DispatchInterface(GlobalVariable)

template<class T, bool Assert>
class Variable<T, GlobalSpace, Assert> : DispatchInherit(GlobalVariable), public VariableBase<T, GlobalSpace, Assert>
{
	friend class TypedVariableDeclaration<T, GlobalSpace>;
public:
	// Visitors

	void Accept(OperandVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstOperandVisitor& visitor) const override { visitor.Visit(this); }

protected:
	using VariableBase<T, GlobalSpace, Assert>::VariableBase;

	DispatchMember_Type(T);
};

DispatchImplementation(GlobalVariable)

}
