#pragma once

#include "Backend/Codegen/Generators/Generator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

namespace Backend {
namespace Codegen {

class RegisterGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "RegisterGenerator"; }

	// Generators

	SASS::Register *Generate(const PTX::_Register *reg);

	template<class T>
	void Visit(const PTX::Register<T> *reg);

	//TODO: Other register kinds

private:
	SASS::Register *m_register = nullptr;
};

}
}
