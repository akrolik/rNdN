#include "Runtime/GPU/Program.h"

namespace Runtime {
namespace GPU {

CUDA::Kernel Program::GetKernel(const std::string& name) const
{
	return CUDA::Kernel(name, *m_binary);
}

const PTX::FunctionDefinition<PTX::VoidType> *Program::GetKernelCode(const std::string& name) const
{
	return m_program->GetEntryFunction(name);
}

std::string Program::ToString() const
{
	std::string string;
	string += "Generated CUDA program:";
	for (const auto& module : m_program->GetModules())
	{
		for (const auto& [name, _] : module->GetEntryFunctions())
		{
			CUDA::Kernel kernel(name, *m_binary);
			if (string.length() > 0)
			{
				string += "\n";
			}
			string += " - Kernel '" + name + "': ";
			string += "registers = " + std::to_string(kernel.GetNumberRegisters()) + "; ";
			string += "smem = " + std::to_string(kernel.GetSharedBytes()) + " bytes; ";
			string += "cmem = " + std::to_string(kernel.GetConstBytes()) + " bytes; ";
			string += "lmem = " + std::to_string(kernel.GetLocalBytes()) + " bytes; ";
			string += "threads = " + std::to_string(kernel.GetMaxThreads());
		}
	}
	return string;
}

}
}
