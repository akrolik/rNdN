#pragma once

#include <string>
#include <sstream>

#include "PTX/InstructionStatement.h"
#include "PTX/Register.h"
#include "PTX/StateSpace.h"

namespace PTX {

template<typename T>
class LoadInstruction : public InstructionStatement
{
public:
	LoadInstruction(Register<T> *reg, StateSpace<T> *mem) : m_register(reg), m_mem(mem) {}

	inline std::string ToString()
	{
		std::ostringstream code;

		code << "ld" << m_mem->SpaceName() << PTX::TypeName<T>() << " ";
		code << m_register->ToString() << ", [" << m_mem->GetName() << "]";
		code << ";" << std::endl;

		return code.str();
	}
private:
	Register<T> *m_register = nullptr;
	StateSpace<T> *m_mem = nullptr;
};

}
