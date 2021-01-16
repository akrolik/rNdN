#pragma once

#include "PTX/Tree/Operands/Variables/Register.h"

namespace PTX {

DispatchInterface(SinkRegister)

template<class T, bool Assert = true>
class SinkRegister : DispatchInherit(SinkRegister), public Register<T, Assert>
{
public:
	SinkRegister() : Register<T, Assert>(new NameSet("_"), 0) {}

	// Visitors

	void Accept(OperandVisitor& visitor) override { visitor.Visit(static_cast<_SinkRegister *>(this)); }
	void Accept(ConstOperandVisitor& visitor) const override { visitor.Visit(static_cast<const _SinkRegister *>(this)); }

protected:
	DispatchMember_Type(T);
};

}
