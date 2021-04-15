#pragma once

#include "Backend/Codegen/Generators/Generator.h"
#include "PTX/Traversal/ConstOperandVisitor.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

#include <utility>

namespace Backend {
namespace Codegen {

class PredicateGenerator : public Generator, public PTX::ConstOperandVisitor
{
public:
	using Generator::Generator;

	std::string Name() const override { return "PredicateGenerator"; }

	// Generators

	std::pair<SASS::Predicate *, bool> Generate(const PTX::Operand *operand);

	// Registers

	void Visit(const PTX::_Register *reg) override;

	template<class T>
	void Visit(const PTX::Register<T> *reg);

	// Values

	void Visit(const PTX::_Value *value);

	template<class T>
	void Visit(const PTX::Value<T> *value);

private:
	SASS::Predicate *m_predicate = nullptr;
	bool m_negatePredicate = false;
};

}
}
