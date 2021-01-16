#pragma once

#include "PTX/Traversal/ConstOperandVisitor.h"

#include "PTX/Tree/Tree.h"

namespace PTX {

template<class D>
class ConstOperandDispatcher : public ConstOperandVisitor
{
public:
	// Constants
	// Label: Provided by base class

	void Visit(const _BracedOperand *operand) override
	{
		operand->Dispatch<D>(static_cast<D&>(*this));
	}

	void Visit(const _Constant *constant) override
	{
		constant->Dispatch<D>(static_cast<D&>(*this));
	}

	void Visit(const _Value *value) override
	{
		value->Dispatch<D>(static_cast<D&>(*this));
	}

	template<class T, VectorSize V> void Visit(const BracedOperand<T, V> *operand) {}
	template<class T> void Visit(const Constant<T> *constant) {}
	template<class T> void Visit(const Value<T> *value) {}

	// Extended: Provided by base class

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

	template<Bits B, class T, class S> void Visit(const DereferencedAddress<B, T, S> *address) {}
	template<Bits B, class T, class S> void Visit(const MemoryAddress<B, T, S> *address) {}
	template<Bits B, class T, class S> void Visit(const RegisterAddress<B, T, S> *address) {}

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
	template<class T> void Visit(const SinkRegister<T> *reg) {}
	template<class T, VectorSize V> void Visit(const IndexedRegister<T, V> *reg) {}
	template<class T, VectorSize V> void Visit(const BracedRegister<T, V> *reg) {}
};

}
