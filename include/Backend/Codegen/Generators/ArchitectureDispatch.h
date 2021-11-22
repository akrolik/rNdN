#pragma once

#include "Backend/Codegen/Builder.h"

#include "Utils/Logger.h"

namespace Backend {
namespace Codegen {

class ArchitectureDispatch
{
public:
	template<class G, class I>
	static void Dispatch(G& generator, const I& instruction)
	{
		// Dispatch for each compute version

		auto computeCapability = generator.m_builder.GetComputeCapability();
		if (SASS::Maxwell::IsSupported(computeCapability))
		{
			generator.GenerateMaxwell(instruction);
		}
		else if (SASS::Volta::IsSupported(computeCapability))
		{
			generator.GenerateVolta(instruction);
		}
		else
		{
			Utils::Logger::LogError("Unsupported CUDA compute capability for dispatch 'sm_" + std::to_string(computeCapability) + "'");
		}
	}

	template<class M, class V>
	static void DispatchInline(const Builder& builder, M maxwellFunc, V voltaFunc)
	{
		// Dispatch for each compute version

		auto computeCapability = builder.GetComputeCapability();
		if (SASS::Maxwell::IsSupported(computeCapability))
		{
			maxwellFunc();
		}
		else if (SASS::Volta::IsSupported(computeCapability))
		{
			voltaFunc();
		}
		else
		{
			Utils::Logger::LogError("Unsupported CUDA compute capability for dispatch 'sm_" + std::to_string(computeCapability) + "'");
		}
	}
};

}
}
