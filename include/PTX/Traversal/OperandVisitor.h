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
	virtual bool Visit(Label *label) { return true; }

	// Constants

	virtual bool Visit(_Constant *constant) { return true; }
	virtual bool Visit(_ParameterConstant *constant) { return true; }
	virtual bool Visit(_Value *value) { return true; }

	// Extended

	virtual bool Visit(_BracedOperand *operand) { return true; }
	virtual bool Visit(_DualOperand *operand) { return true; }
	virtual bool Visit(_InvertedOperand *operand) { return true; }
	virtual bool Visit(ListOperand *operand) { return true; }
	virtual bool Visit(StringOperand *operand) { return true; }

	// Address

	virtual bool Visit(_DereferencedAddress *address) { return true; }
	virtual bool Visit(_MemoryAddress *address) { return true; }
	virtual bool Visit(_RegisterAddress *address) { return true; }

	// Variables

	virtual bool Visit(_ConstVariable *variable) { return true; }
	virtual bool Visit(_GlobalVariable *variable) { return true; }
	virtual bool Visit(_LocalVariable *variable) { return true; }
	virtual bool Visit(_ParameterVariable *variable) { return true; }
	virtual bool Visit(_SharedVariable *variable) { return true; }

	// Registers

	virtual bool Visit(_Register *reg) { return true; }
	virtual bool Visit(_SpecialRegister *reg) { return true; }
	virtual bool Visit(_SinkRegister *reg) { return true; }
	virtual bool Visit(_IndexedRegister *reg) { return true; }
	virtual bool Visit(_BracedRegister *reg) { return true; }
};

}
