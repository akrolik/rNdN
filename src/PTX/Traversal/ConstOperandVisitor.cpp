#include "PTX/Traversal/ConstOperandVisitor.h"

#include "PTX/Tree/Tree.h"

namespace PTX {

bool ConstOperandVisitor::Visit(const DispatchBase *operand)
{
	return true;
}

bool ConstOperandVisitor::Visit(const Label *label)
{
	return Visit(static_cast<const DispatchBase *>(label));
}

// Constants

bool ConstOperandVisitor::Visit(const _Constant *constant)
{
	return Visit(static_cast<const DispatchBase *>(constant));
}

bool ConstOperandVisitor::Visit(const _ParameterConstant *constant)
{
	return Visit(static_cast<const DispatchBase *>(constant));
}

bool ConstOperandVisitor::Visit(const _Value *value)
{
	return Visit(static_cast<const DispatchBase *>(value));
}

// Extended

bool ConstOperandVisitor::Visit(const _BracedOperand *operand)
{
	return Visit(static_cast<const DispatchBase *>(operand));
}

bool ConstOperandVisitor::Visit(const _DualOperand *operand)
{
	return Visit(static_cast<const DispatchBase *>(operand));
}

bool ConstOperandVisitor::Visit(const _InvertedOperand *operand)
{
	return Visit(static_cast<const DispatchBase *>(operand));
}

bool ConstOperandVisitor::Visit(const ListOperand *operand)
{
	return Visit(static_cast<const DispatchBase *>(operand));
}

bool ConstOperandVisitor::Visit(const StringOperand *operand)
{
	return Visit(static_cast<const DispatchBase *>(operand));
}

// Address

bool ConstOperandVisitor::Visit(const _DereferencedAddress *address)
{
	return Visit(static_cast<const DispatchBase *>(address));
}

bool ConstOperandVisitor::Visit(const _MemoryAddress *address)
{
	return Visit(static_cast<const DispatchBase *>(address));
}

bool ConstOperandVisitor::Visit(const _RegisterAddress *address)
{
	return Visit(static_cast<const DispatchBase *>(address));
}

// Variables

bool ConstOperandVisitor::Visit(const _ConstVariable *variable)
{
	return Visit(static_cast<const DispatchBase *>(variable));
}

bool ConstOperandVisitor::Visit(const _GlobalVariable *variable)
{
	return Visit(static_cast<const DispatchBase *>(variable));
}

bool ConstOperandVisitor::Visit(const _LocalVariable *variable)
{
	return Visit(static_cast<const DispatchBase *>(variable));
}

bool ConstOperandVisitor::Visit(const _ParameterVariable *variable)
{
	return Visit(static_cast<const DispatchBase *>(variable));
}

bool ConstOperandVisitor::Visit(const _SharedVariable *variable)
{
	return Visit(static_cast<const DispatchBase *>(variable));
}

// Registers

bool ConstOperandVisitor::Visit(const _Register *reg)
{
	return Visit(static_cast<const DispatchBase *>(reg));
}

bool ConstOperandVisitor::Visit(const _SpecialRegister *reg)
{
	return Visit(static_cast<const DispatchBase *>(reg));
}

bool ConstOperandVisitor::Visit(const _SinkRegister *reg)
{
	return Visit(static_cast<const DispatchBase *>(reg));
}

bool ConstOperandVisitor::Visit(const _IndexedRegister *reg)
{
	return Visit(static_cast<const DispatchBase *>(reg));
}

bool ConstOperandVisitor::Visit(const _BracedRegister *reg)
{
	return Visit(static_cast<const DispatchBase *>(reg));
}

}
