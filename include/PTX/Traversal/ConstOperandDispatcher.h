#pragma once

#include "PTX/Traversal/ConstOperandVisitor.h"

#include "PTX/Tree/Tree.h"

namespace PTX {

template<class D>
class ConstOperandDispatcher : public ConstOperandVisitor
{
public:
	void Visit(const Label *label) override {}

	// Constants

	void Visit(const _Constant *constant) override
	{
		constant->Dispatch<D>(static_cast<D&>(*this));
	}

	void Visit(const _Value *value) override
	{
		value->Dispatch<D>(static_cast<D&>(*this));
	}

	template<class T> void Visit(const Constant<T> *constant) {}
	template<class T> void Visit(const Value<T> *value) {}

	// Extended

	void Visit(const _BracedOperand *operand) override
	{
		operand->Dispatch<D>(static_cast<D&>(*this));
	}

	void Visit(const _DualOperand *operand) override
	{
		operand->Dispatch<D>(static_cast<D&>(*this));
	}

	void Visit(const _InvertedOperand *operand) override
	{
		operand->Dispatch<D>(static_cast<D&>(*this));
	}

	void Visit(const ListOperand *operand) override
	{
		for (const auto& _operand : operand->GetOperands())
		{
			_operand->Accept(*this);
		}
	}

	void Visit(const StringOperand *operand) override {}
	
	template<class T, VectorSize V> void Visit(const BracedOperand<T, V> *operand)
	{
		for (const auto& _operand : operand->GetOperands())
		{
			_operand->Accept(*this);
		}
	}

	template<class T1, class T2> void Visit(const DualOperand<T1, T2> *operand)
	{
		operand->GetOperandP()->Accept(*this);
		operand->GetOperandQ()->Accept(*this);
	}

	template<class T> void Visit(const InvertedOperand<T> *operand)
	{
		operand->GetOperand()->Accept(*this);
	}

	// Address

	void Visit(const _DereferencedAddress *address) override
	{
		address->Dispatch<D>(static_cast<D&>(*this));
	}

	void Visit(const _MemoryAddress *address) override
	{
		address->Dispatch<D>(static_cast<D&>(*this));
	}

	void Visit(const _RegisterAddress *address) override
	{
		address->Dispatch<D>(static_cast<D&>(*this));
	}

	template<Bits B, class T, class S>
	void Visit(const DereferencedAddress<B, T, S> *address)
	{
		address->GetAddress()->Accept(*this);
	}

	template<Bits B, class T, class S>
	void Visit(const MemoryAddress<B, T, S> *address)
	{
		address->GetVariable()->Accept(*this);
	}

	template<Bits B, class T, class S>
	void Visit(const RegisterAddress<B, T, S> *address)
	{
		address->GetRegister()->Accept(*this);
	}

	// Variables

	void Visit(const _ConstVariable *variable) override
	{
		variable->Dispatch<D>(static_cast<D&>(*this));
	}

	void Visit(const _GlobalVariable *variable) override
	{
		variable->Dispatch<D>(static_cast<D&>(*this));
	}

	void Visit(const _LocalVariable *variable) override
	{
		variable->Dispatch<D>(static_cast<D&>(*this));
	}

	void Visit(const _ParameterVariable *variable) override
	{
		variable->Dispatch<D>(static_cast<D&>(*this));
	}

	void Visit(const _SharedVariable *variable) override
	{
		variable->Dispatch<D>(static_cast<D&>(*this));
	}

	template<class T> void Visit(const ConstVariable<T> *variable) {}
	template<class T> void Visit(const GlobalVariable<T> *variable) {}
	template<class T> void Visit(const LocalVariable<T> *variable) {}
	template<class T> void Visit(const ParameterVariable<T> *variable) {}
	template<class T> void Visit(const SharedVariable<T> *variable) {}

	// Registers

	void Visit(const _Register *reg) override
	{
		reg->Dispatch<D>(static_cast<D&>(*this));
	}

	void Visit(const _SpecialRegister *reg) override
	{
		reg->Dispatch<D>(static_cast<D&>(*this));
	}

	void Visit(const _SinkRegister *reg) override
	{
		reg->Dispatch<D>(static_cast<D&>(*this));
	}

	void Visit(const _IndexedRegister *reg) override
	{
		reg->Dispatch<D>(static_cast<D&>(*this));
	}

	void Visit(const _BracedRegister *reg) override
	{
		reg->Dispatch<D>(static_cast<D&>(*this));
	}

	template<class T> void Visit(const Register<T> *reg) {}
	template<class T> void Visit(const SpecialRegister<T> *reg) {}
	template<class T> void Visit(const SinkRegister<T> *reg) {}

	template<class T, class S, VectorSize V>
	void Visit(const IndexedRegister<T, S, V> *reg)
	{
		reg->GetVariable()->Accept(*this);
	}

	template<class T, VectorSize V>
	void Visit(const BracedRegister<T, V> *reg)
	{
		for (const auto& _reg : reg->GetRegisters())
		{
			_reg->Accept(*this);
		}
	}
};

}
