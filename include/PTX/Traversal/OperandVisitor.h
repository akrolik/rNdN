#pragma once

namespace PTX {

class Label;

class _Constant;
class _ParameterConstant;
class _Value;

class _BracedOperand;
class _DualOperand;
class _InvertedOperand;
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
class _SpecialRegister;
class _SinkRegister;
class _IndexedRegister;
class _BracedRegister;

class OperandVisitor
{
public:
	virtual void Visit(Label *label) = 0;

	// Constants

	virtual void Visit(_Constant *constant) = 0;
	virtual void Visit(_ParameterConstant *constant) = 0;
	virtual void Visit(_Value *value) = 0;

	// Extended

	virtual void Visit(_BracedOperand *operand) = 0;
	virtual void Visit(_DualOperand *operand) = 0;
	virtual void Visit(_InvertedOperand *operand) = 0;
	virtual void Visit(ListOperand *operand) = 0;
	virtual void Visit(StringOperand *operand) = 0;

	// Address

	virtual void Visit(_DereferencedAddress *address) = 0;
	virtual void Visit(_MemoryAddress *address) = 0;
	virtual void Visit(_RegisterAddress *address) = 0;

	// Variables

	virtual void Visit(_ConstVariable *variable) = 0;
	virtual void Visit(_GlobalVariable *variable) = 0;
	virtual void Visit(_LocalVariable *variable) = 0;
	virtual void Visit(_ParameterVariable *variable) = 0;
	virtual void Visit(_SharedVariable *variable) = 0;

	// Registers

	virtual void Visit(_Register *reg) = 0;
	virtual void Visit(_SpecialRegister *reg) = 0;
	virtual void Visit(_SinkRegister *reg) = 0;
	virtual void Visit(_IndexedRegister *reg) = 0;
	virtual void Visit(_BracedRegister *reg) = 0;
};

}
