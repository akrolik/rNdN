#pragma once

namespace PTX {

class Label;
class _BracedOperand;
class _Constant;
class _Value;

class DualOperand;
class HexOperand;
class InvertedOperand;
class ListOperand;
class StringOperand;

class _DereferencedAddress;
class _MemoryAddress;
class _RegisterAddress;

class _ConstVariable;
class _GlobalVariable;
class _LocalVariable;
class _ParameterVariable;
class _SharedVariable;

class _Register;
class _SinkRegister;
class _IndexedRegister;
class _BracedRegister;

class OperandVisitor
{
public:
	// Constants

	virtual void Visit(Label *label) {}
	virtual void Visit(_BracedOperand *operand) {}
	virtual void Visit(_Constant *constant) {}
	virtual void Visit(_Value *value) {}

	// Extended

	virtual void Visit(DualOperand *operand) {}
	virtual void Visit(HexOperand *operand) {}
	virtual void Visit(InvertedOperand *operand) {}
	virtual void Visit(ListOperand *operand) {}
	virtual void Visit(StringOperand *operand) {}

	// Address

	virtual void Visit(_DereferencedAddress *address) {}
	virtual void Visit(_MemoryAddress *address) {}
	virtual void Visit(_RegisterAddress *address) {}

	// Variables

	virtual void Visit(_ConstVariable *variable) {}
	virtual void Visit(_GlobalVariable *variable) {}
	virtual void Visit(_LocalVariable *variable) {}
	virtual void Visit(_ParameterVariable *variable) {}
	virtual void Visit(_SharedVariable *variable) {}

	// Registers

	virtual void Visit(_Register *reg) {}
	virtual void Visit(_SinkRegister *reg) {}
	virtual void Visit(_IndexedRegister *reg) {}
	virtual void Visit(_BracedRegister *reg) {}
};

}
