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

class _AddressableVariable;
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
	virtual void Visit(const _Value *value) {}

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

	virtual void Visit(_AddressableVariable *variable) {}
	virtual void Visit(_Register *reg) {}
	virtual void Visit(_SinkRegister *reg) {}
	virtual void Visit(_IndexedRegister *reg) {}
	virtual void Visit(_BracedRegister *reg) {}
};

}
