#pragma once

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Declarations/VariableDeclaration.h"
#include "PTX/Tree/Operands/Variables/Variable.h"

namespace PTX {

template<class T, bool Assert = true>
using LocalVariable = Variable<T, LocalSpace, Assert>;

DispatchInterface(LocalVariable)

template<class T, bool Assert>
class Variable<T, LocalSpace, Assert> : DispatchInherit(LocalVariable), public VariableBase<T, LocalSpace, Assert>
{
	friend class TypedVariableDeclaration<T, LocalSpace>;
public:
	// Visitors

	void Accept(OperandVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstOperandVisitor& visitor) const override { visitor.Visit(this); }

protected:
	using VariableBase<T, LocalSpace, Assert>::VariableBase;

	DispatchMember_Type(T);
};

DispatchImplementation(LocalVariable)

}
