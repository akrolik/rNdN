#pragma once

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Declarations/VariableDeclaration.h"
#include "PTX/Tree/Operands/Variables/Variable.h"

namespace PTX {

template<class T, bool Assert = true>
using SharedVariable = Variable<T, SharedSpace, Assert>;

DispatchInterface_Using(SharedVariable)

template<class T, bool Assert>
class Variable<T, SharedSpace, Assert> : DispatchInherit(SharedVariable), public VariableBase<T, SharedSpace, Assert>
{
	friend class TypedVariableDeclaration<T, SharedSpace>;
public:
	// Visitors

	void Accept(OperandVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstOperandVisitor& visitor) const override { visitor.Visit(this); }

protected:
	using VariableBase<T, SharedSpace, Assert>::VariableBase;

	DispatchMember_Type(T);
};

}
