#include "PTX/Traversal/OperandVisitor.h"

#include "PTX/Tree/Tree.h"

namespace PTX {

bool OperandVisitor::Visit(DispatchBase *operand)
{
	return true;
}

bool OperandVisitor::Visit(Label *label)
{
	return Visit(static_cast<DispatchBase *>(label));
}

// Constants

bool OperandVisitor::Visit(_Constant *constant)
{
	return Visit(static_cast<DispatchBase *>(constant));
}

bool OperandVisitor::Visit(_ParameterConstant *constant)
{
	return Visit(static_cast<DispatchBase *>(constant));
}

bool OperandVisitor::Visit(_Value *value)
{
	return Visit(static_cast<DispatchBase *>(value));
}

// Extended

bool OperandVisitor::Visit(_BracedOperand *operand)
{
	return Visit(static_cast<DispatchBase *>(operand));
}

bool OperandVisitor::Visit(_DualOperand *operand)
{
	return Visit(static_cast<DispatchBase *>(operand));
}

bool OperandVisitor::Visit(_InvertedOperand *operand)
{
	return Visit(static_cast<DispatchBase *>(operand));
}

bool OperandVisitor::Visit(ListOperand *operand)
{
	return Visit(static_cast<DispatchBase *>(operand));
}

bool OperandVisitor::Visit(StringOperand *operand)
{
	return Visit(static_cast<DispatchBase *>(operand));
}

// Address

bool OperandVisitor::Visit(_DereferencedAddress *address)
{
	return Visit(static_cast<DispatchBase *>(address));
}

bool OperandVisitor::Visit(_MemoryAddress *address)
{
	return Visit(static_cast<DispatchBase *>(address));
}

bool OperandVisitor::Visit(_RegisterAddress *address)
{
	return Visit(static_cast<DispatchBase *>(address));
}

// Variables

bool OperandVisitor::Visit(_ConstVariable *variable)
{
	return Visit(static_cast<DispatchBase *>(variable));
}

bool OperandVisitor::Visit(_GlobalVariable *variable)
{
	return Visit(static_cast<DispatchBase *>(variable));
}

bool OperandVisitor::Visit(_LocalVariable *variable)
{
	return Visit(static_cast<DispatchBase *>(variable));
}

bool OperandVisitor::Visit(_ParameterVariable *variable)
{
	return Visit(static_cast<DispatchBase *>(variable));
}

bool OperandVisitor::Visit(_SharedVariable *variable)
{
	return Visit(static_cast<DispatchBase *>(variable));
}

// Registers

bool OperandVisitor::Visit(_Register *reg)
{
	return Visit(static_cast<DispatchBase *>(reg));
}

bool OperandVisitor::Visit(_SpecialRegister *reg)
{
	return Visit(static_cast<DispatchBase *>(reg));
}

bool OperandVisitor::Visit(_SinkRegister *reg)
{
	return Visit(static_cast<DispatchBase *>(reg));
}

bool OperandVisitor::Visit(_IndexedRegister *reg)
{
	return Visit(static_cast<DispatchBase *>(reg));
}

bool OperandVisitor::Visit(_BracedRegister *reg)
{
	return Visit(static_cast<DispatchBase *>(reg));
}

}
