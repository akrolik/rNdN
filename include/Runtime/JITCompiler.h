#pragma once

#include "HorseIR/Tree/Program.h"

#include "PTX/Program.h"

namespace Runtime {

class JITCompiler
{
public:
	struct Options
	{
		std::string ComputeCapability = "sm_61";
		unsigned int MaxBlockSize;
		unsigned long InputSize;

		std::string ToString() const
		{
			std::string output;
			output += "Compute capability: " + ComputeCapability + "\n";
			output += "Max block size: " + std::to_string(MaxBlockSize) + "\n";
			output += "Input size: " + std::to_string(InputSize);
			return output;
		}
	};

	JITCompiler(const Options& options) : m_options(options) {}

	PTX::Program *Compile(HorseIR::Program *program);
	void Optimize(PTX::Program *program);

private:
	Options m_options;
};

}
