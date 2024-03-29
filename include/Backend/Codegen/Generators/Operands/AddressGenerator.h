#pragma once

#include "Backend/Codegen/Generators/Generator.h"
#include "PTX/Traversal/ConstOperandVisitor.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class AddressGenerator : public Generator, public PTX::ConstOperandVisitor
{
public:
	using Generator::Generator;

	std::string Name() const override { return "AddressGenerator"; }

	// Generators

	SASS::Address *Generate(const PTX::Operand *operand);

	// Options

	bool GetUseOffset() const { return m_useOffset; }
	void SetUseOffset(bool useOffset) { m_useOffset = useOffset; }

	// Addresss

	bool Visit(const PTX::_MemoryAddress *address) override;
	bool Visit(const PTX::_RegisterAddress *address) override;

	template<PTX::Bits B, class T, class S>
	void Visit(const PTX::MemoryAddress<B, T, S> *address);

	template<PTX::Bits B, class T, class S>
	void Visit(const PTX::RegisterAddress<B, T, S> *address);

private:
	SASS::Address *m_address = nullptr;
	bool m_useOffset = true;
};

}
}
