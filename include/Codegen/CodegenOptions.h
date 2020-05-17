#pragma once

#include <string>

namespace Codegen {

struct CodegenOptions
{
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
