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

class ConstOperandVisitor
{
public:
	virtual bool Visit(const DispatchBase *operand);
	virtual bool Visit(const Label *label);

	// Constants

	virtual bool Visit(const _Constant *constant);
	virtual bool Visit(const _ParameterConstant *constant);
	virtual bool Visit(const _Value *value);

	// Extended

	virtual bool Visit(const _BracedOperand *operand);
	virtual bool Visit(const _DualOperand *operand);
	virtual bool Visit(const _InvertedOperand *operand);
	virtual bool Visit(const ListOperand *operand);
	virtual bool Visit(const StringOperand *operand);

	// Address

	virtual bool Visit(const _DereferencedAddress *address);
	virtual bool Visit(const _MemoryAddress *address);
	virtual bool Visit(const _RegisterAddress *address);

	// Variables

	virtual bool Visit(const _ConstVariable *variable);
	virtual bool Visit(const _GlobalVariable *variable);
	virtual bool Visit(const _LocalVariable *variable);
	virtual bool Visit(const _ParameterVariable *variable);
	virtual bool Visit(const _SharedVariable *variable);

	// Registers

	virtual bool Visit(const _Register *reg);
	virtual bool Visit(const _SpecialRegister *reg);
	virtual bool Visit(const _SinkRegister *reg);
	virtual bool Visit(const _IndexedRegister *reg);
	virtual bool Visit(const _BracedRegister *reg);
};

}
