#pragma once

#include "Backend/Codegen/Generators/Generator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

namespace Backend {
namespace Codegen {

class SpecialRegisterGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "SpecialRegisterGenerator"; }

	// Generators

	SASS::SpecialRegister *Generate(const PTX::_SpecialRegister *reg);

	template<class T>
	void Visit(const PTX::SpecialRegister<T> *reg);

	//TODO: Other register kinds

private:
	SASS::SpecialRegister *m_register = nullptr;
};

}
}
