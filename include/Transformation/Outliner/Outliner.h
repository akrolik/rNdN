#pragma once

#include <vector>

#include "Analysis/Shape/ShapeAnalysis.h"

#include "Analysis/Compatibility/Overlay/CompatibilityOverlay.h"

#include "HorseIR/Tree/Tree.h"

namespace Transformation {

class Outliner
{
public:
	std::vector<HorseIR::Function *> Outline(const Analysis::CompatibilityOverlay *overlay);
};

}
