#include "Runtime/GPU/Compiler.h"

#include "Frontend/Compiler.h"

namespace Runtime {
namespace GPU {

const PTX::Program *Compiler::Compile(const HorseIR::Program *program) const
{
	Frontend::Compiler compiler(m_gpuManager.GetCurrentDevice());
	return compiler.Compile(program);
}

}
}
