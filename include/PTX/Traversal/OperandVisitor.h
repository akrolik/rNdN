#pragma once

namespace PTX {

class DispatchBase;
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
	virtual bool Visit(DispatchBase *operand);
	virtual bool Visit(Label *label);

	// Constants

	virtual bool Visit(_Constant *constant);
	virtual bool Visit(_ParameterConstant *constant);
	virtual bool Visit(_Value *value);

	// Extended

	virtual bool Visit(_BracedOperand *operand);
	virtual bool Visit(_DualOperand *operand);
	virtual bool Visit(_InvertedOperand *operand);
	virtual bool Visit(ListOperand *operand);
	virtual bool Visit(StringOperand *operand);

	// Address

	virtual bool Visit(_DereferencedAddress *address);
	virtual bool Visit(_MemoryAddress *address);
	virtual bool Visit(_RegisterAddress *address);

	// Variables

	virtual bool Visit(_ConstVariable *variable);
	virtual bool Visit(_GlobalVariable *variable);
	virtual bool Visit(_LocalVariable *variable);
	virtual bool Visit(_ParameterVariable *variable);
	virtual bool Visit(_SharedVariable *variable);

	// Registers

	virtual bool Visit(_Register *reg);
	virtual bool Visit(_SpecialRegister *reg);
	virtual bool Visit(_SinkRegister *reg);
	virtual bool Visit(_IndexedRegister *reg);
	virtual bool Visit(_BracedRegister *reg);
};

}
