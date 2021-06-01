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

class ConstOperandVisitor
{
public:
	virtual bool Visit(const Label *label) { return true; }

	// Constants

	virtual bool Visit(const _Constant *constant) { return true; }
	virtual bool Visit(const _ParameterConstant *constant) { return true; }
	virtual bool Visit(const _Value *value) { return true; }

	// Extended

	virtual bool Visit(const _BracedOperand *operand) { return true; }
	virtual bool Visit(const _DualOperand *operand) { return true; }
	virtual bool Visit(const _InvertedOperand *operand) { return true; }
	virtual bool Visit(const ListOperand *operand) { return true; }
	virtual bool Visit(const StringOperand *operand) { return true; }

	// Address

	virtual bool Visit(const _DereferencedAddress *address) { return true; }
	virtual bool Visit(const _MemoryAddress *address) { return true; }
	virtual bool Visit(const _RegisterAddress *address) { return true; }

	// Variables

	virtual bool Visit(const _ConstVariable *variable) { return true; }
	virtual bool Visit(const _GlobalVariable *variable) { return true; }
	virtual bool Visit(const _LocalVariable *variable) { return true; }
	virtual bool Visit(const _ParameterVariable *variable) { return true; }
	virtual bool Visit(const _SharedVariable *variable) { return true; }

	// Registers

	virtual bool Visit(const _Register *reg) { return true; }
	virtual bool Visit(const _SpecialRegister *reg) { return true; }
	virtual bool Visit(const _SinkRegister *reg) { return true; }
	virtual bool Visit(const _IndexedRegister *reg) { return true; }
	virtual bool Visit(const _BracedRegister *reg) { return true; }
};

}
