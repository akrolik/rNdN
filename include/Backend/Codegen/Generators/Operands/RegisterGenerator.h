#pragma once

#include "Backend/Codegen/Generators/Generator.h"
#include "PTX/Traversal/ConstOperandVisitor.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

#include <utility>

namespace Backend {
namespace Codegen {

class RegisterGenerator : public Generator, public PTX::ConstOperandVisitor
{
public:
	using Generator::Generator;

	std::string Name() const override { return "RegisterGenerator"; }

	// Generators

	std::pair<SASS::Register *, SASS::Register *> Generate(const PTX::Operand *operand);

	// Registers

	void Visit(const PTX::_Register *reg) override;
	void Visit(const PTX::_IndexedRegister *reg) override;
	void Visit(const PTX::_SinkRegister *reg) override;

	template<class T> void Visit(const PTX::Register<T> *reg);

	template<class T, class S, PTX::VectorSize V>
	void Visit(const PTX::IndexedRegister<T, S, V> *reg);

	template<class T>
	void Visit(const PTX::SinkRegister<T> *reg);

	// Values

	void Visit(const PTX::_Constant *constant);
	void Visit(const PTX::_Value *value);

	template<class T>
	void Visit(const PTX::Constant<T> *value);

	template<class T>
	void Visit(const PTX::Value<T> *value);

private:
	SASS::Register *m_register = nullptr;
	SASS::Register *m_registerHi = nullptr;
};

}
}
