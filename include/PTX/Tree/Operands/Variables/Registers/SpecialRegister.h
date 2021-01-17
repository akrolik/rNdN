#pragma once

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Declarations/VariableDeclaration.h"
#include "PTX/Tree/Operands/Variables/Variable.h"

namespace PTX {

template<class T, bool Assert = true>
using SpecialRegister = Variable<T, SpecialRegisterSpace, Assert>;

DispatchInterface_Using(SpecialRegister)

template<class T, bool Assert>
class Variable<T, SpecialRegisterSpace, Assert> : DispatchInherit(SpecialRegister), public VariableBase<T, SpecialRegisterSpace, Assert>
{
	friend class TypedVariableDeclaration<T, SpecialRegisterSpace>;
public:
	constexpr static bool TypeSupported = VariableBase<T, SpecialRegisterSpace, Assert>::TypeSupported;

	// Visitors

	void Accept(OperandVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstOperandVisitor& visitor) const override { visitor.Visit(this); }

protected:
	using VariableBase<T, SpecialRegisterSpace, Assert>::VariableBase;

	DispatchMember_Type(T);
};

}
