#pragma once

#include "Backend/Codegen/Generators/Generator.h"
#include "PTX/Traversal/ConstOperandVisitor.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

#include <utility>

namespace Backend {
namespace Codegen {

class CompositeGenerator : public Generator, public PTX::ConstOperandVisitor
{
public:
	using Generator::Generator;

	std::string Name() const override { return "CompositeGenerator"; }

	// Options

	bool GetZeroRegister() const { return m_zeroRegister; }
	void SetZeroRegister(bool zeroRegister) { m_zeroRegister = zeroRegister; }

	bool GetImmediateValue() const { return m_immediateValue; }
	void SetImmediateValue(bool immediateValue) { m_immediateValue = immediateValue; }

	// Generators

	std::pair<SASS::Composite *, SASS::Composite *> Generate(const PTX::Operand *operand);

	// Registers

	void Visit(const PTX::_Register *reg) override;
	void Visit(const PTX::_IndexedRegister *reg) override;

	template<class T>
	void Visit(const PTX::Register<T> *reg);

	template<class T, class S, PTX::VectorSize V>
	void Visit(const PTX::IndexedRegister<T, S, V> *reg);

	// Values

	void Visit(const PTX::_Constant *constant) override;
	void Visit(const PTX::_Value *value) override;

	template<class T>
	void Visit(const PTX::Constant<T> *constant);

	template<class T>
	void Visit(const PTX::Value<T> *value);

	// Addresses

	void Visit(const PTX::_MemoryAddress *address) override;

	template<PTX::Bits B, class T, class S>
	void Visit(const PTX::MemoryAddress<B, T, S> *address);

private:
	SASS::Composite *m_composite = nullptr;
	SASS::Composite *m_compositeHi = nullptr;

	bool m_zeroRegister = true;
	bool m_immediateValue = true;
};

}
}
