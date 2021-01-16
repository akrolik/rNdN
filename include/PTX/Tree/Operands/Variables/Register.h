#pragma once

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Declarations/VariableDeclaration.h"
#include "PTX/Tree/Operands/Variables/Variable.h"

namespace PTX {

template<class T, bool Assert = true>
using Register = Variable<T, RegisterSpace, Assert>;

DispatchInterface_Using(Register)

template<class T, bool Assert>
class Variable<T, RegisterSpace, Assert> : DispatchInherit(Register), public VariableBase<T, RegisterSpace, Assert>, public TypedOperand<T, Assert>
{
	friend class TypedVariableDeclaration<T, SpecialRegisterSpace>;
	friend class TypedVariableDeclaration<T, RegisterSpace>;
public:
	constexpr static bool TypeSupported = VariableBase<T, RegisterSpace, Assert>::TypeSupported;

	// Visitors

	void Accept(OperandVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstOperandVisitor& visitor) const override { visitor.Visit(this); }

protected:
	using VariableBase<T, RegisterSpace, Assert>::VariableBase;

	DispatchMember_Type(T);
};

}
