#include "Transformation/Outliner/Outliner.h"

#include "Transformation/Outliner/OutlinePartitioner.h"
#include "Transformation/Outliner/OutlineBuilder.h"


#include "Analysis/Compatibility/Overlay/CompatibilityOverlayPrinter.h"
#include "Utils/Logger.h"

namespace Transformation {

std::vector<HorseIR::Function *> Outliner::Outline(const Analysis::CompatibilityOverlay *overlay)
{
	// 1. Partition the overlay into an outlined overlay with nested functions
	// 2. Build the partitioned graph into functions collected in a vector

	OutlinePartitioner partitioner;
	auto partitioned = partitioner.Partition(overlay);

	//TODO: Printing debug?
	auto partitionedString = Analysis::CompatibilityOverlayPrinter::PrettyString(partitioned);
	Utils::Logger::LogInfo(partitionedString, 0, true, Utils::Logger::NoPrefix);

	std::vector<HorseIR::Function *> functions;

	// OutlineBuilder builder(functions);
	// builder.Build(partitioned);

	return functions;
}

}
