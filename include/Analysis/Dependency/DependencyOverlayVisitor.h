#pragma once

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

class DependencyOverlay;

template<typename T>
class CompoundDependencyOverlay;

class DependencyOverlayVisitor
{
public:
	virtual void Visit(DependencyOverlay *overlay);

	virtual void Visit(CompoundDependencyOverlay<HorseIR::Function> *overlay);
	virtual void Visit(CompoundDependencyOverlay<HorseIR::IfStatement> *overlay);
	virtual void Visit(CompoundDependencyOverlay<HorseIR::WhileStatement> *overlay);
	virtual void Visit(CompoundDependencyOverlay<HorseIR::RepeatStatement> *overlay);
};

}
