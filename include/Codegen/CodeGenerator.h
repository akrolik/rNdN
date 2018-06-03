#pragma once

#include "PTX/Program.h"
#include "PTX/Module.h"

#include "HorseIR/Tree/Program.h"

class CodeGenerator
{
public:
	CodeGenerator(std::string target, PTX::Bits bits) : m_target(target), m_bits(bits) {}
	
	PTX::Program *GenerateProgram(HorseIR::Program *program)
	{
		PTX::Program *ptxProgram = new PTX::Program();

		const std::vector<HorseIR::Module *>& modules = program->GetModules();
		for (auto it = modules.cbegin(); it != modules.cend(); ++it)
		{
			ptxProgram->AddModule(GenerateModule(*it));
		}

		return ptxProgram;
	}

	PTX::Module *GenerateModule(HorseIR::Module *module)
	{
		PTX::Module *ptxModule = new PTX::Module();
		ptxModule->SetVersion(6, 1);
		ptxModule->SetDeviceTarget(m_target);
		ptxModule->SetAddressSize(m_bits);

		//TODO: generate statements

		return ptxModule;
	}

private:
	std::string m_target;
	PTX::Bits m_bits;
};
