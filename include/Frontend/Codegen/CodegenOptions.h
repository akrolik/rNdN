#pragma once

#include <string>

#include "Utils/Options.h"

namespace Frontend {
namespace Codegen {

struct CodegenOptions
{
	CodegenOptions()
	{
		auto reductionMode = Utils::Options::Get<std::string>(Utils::Options::Opt_Algo_reduction);
		if (reductionMode == "shflwarp")
		{
			Reduction = ReductionKind::ShuffleWarp;
		}
		else if (reductionMode  == "shflblock")
		{
			Reduction = ReductionKind::ShuffleBlock;
		}
		else if (reductionMode == "shared")
		{
			Reduction = ReductionKind::Shared;
		}
		else
		{
			Utils::Logger::LogError("Unknown reduction mode '" + reductionMode + "'");
		}
	}

	enum class ReductionKind {
		ShuffleBlock,
		ShuffleWarp,
		Shared
	};

	static std::string ReductionKindString(ReductionKind reductionKind)
	{
		switch (reductionKind)
		{
			case ReductionKind::ShuffleBlock:
				return "Shuffle block";
			case ReductionKind::ShuffleWarp:
				return "Shuffle warp";
			case ReductionKind::Shared:
				return "Shared";
		}
		return "<unknown>";
	}

	ReductionKind Reduction = ReductionKind::ShuffleWarp;

	std::string ToString() const
	{
		std::string output;
		output += "Reduction: " + ReductionKindString(Reduction);
		return output;
	}
};

}
}
