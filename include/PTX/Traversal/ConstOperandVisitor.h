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

class ConstOperandVisitor
{
public:
	// Constants

	virtual void Visit(const Label *label) {}
	virtual void Visit(const _BracedOperand *operand) {}
	virtual void Visit(const _Constant *constant) {}
	virtual void Visit(const _Value *value) {}

	// Extended

	virtual void Visit(const DualOperand *operand) {}
	virtual void Visit(const HexOperand *operand) {}
	virtual void Visit(const InvertedOperand *operand) {}
	virtual void Visit(const ListOperand *operand) {}
	virtual void Visit(const StringOperand *operand) {}

	// Address

	virtual void Visit(const _DereferencedAddress *address) {}
	virtual void Visit(const _MemoryAddress *address) {}
	virtual void Visit(const _RegisterAddress *address) {}

	// Variables

	virtual void Visit(const _ConstVariable *variable) {}
	virtual void Visit(const _GlobalVariable *variable) {}
	virtual void Visit(const _LocalVariable *variable) {}
	virtual void Visit(const _ParameterVariable *variable) {}
	virtual void Visit(const _SharedVariable *variable) {}

	// Registers

	virtual void Visit(const _Register *reg) {}
	virtual void Visit(const _SinkRegister *reg) {}
	virtual void Visit(const _IndexedRegister *reg) {}
	virtual void Visit(const _BracedRegister *reg) {}
};

}