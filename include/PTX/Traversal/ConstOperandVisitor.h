#pragma once

namespace PTX {

class Label;

class _Constant;
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
	virtual void Visit(const Label *label) = 0;

	// Constants

	virtual void Visit(const _Constant *constant) = 0;
	virtual void Visit(const _Value *value) = 0;

	// Extended

	virtual void Visit(const _BracedOperand *operand) = 0;
	virtual void Visit(const _DualOperand *operand) = 0;
	virtual void Visit(const _InvertedOperand *operand) = 0;
	virtual void Visit(const ListOperand *operand) = 0;
	virtual void Visit(const StringOperand *operand) = 0;

	// Address

	virtual void Visit(const _DereferencedAddress *address) = 0;
	virtual void Visit(const _MemoryAddress *address) = 0;
	virtual void Visit(const _RegisterAddress *address) = 0;

	// Variables

	virtual void Visit(const _ConstVariable *variable) = 0;
	virtual void Visit(const _GlobalVariable *variable) = 0;
	virtual void Visit(const _LocalVariable *variable) = 0;
	virtual void Visit(const _ParameterVariable *variable) = 0;
	virtual void Visit(const _SharedVariable *variable) = 0;

	// Registers

	virtual void Visit(const _Register *reg) = 0;
	virtual void Visit(const _SpecialRegister *reg) = 0;
	virtual void Visit(const _SinkRegister *reg) = 0;
	virtual void Visit(const _IndexedRegister *reg) = 0;
	virtual void Visit(const _BracedRegister *reg) = 0;
};

}
