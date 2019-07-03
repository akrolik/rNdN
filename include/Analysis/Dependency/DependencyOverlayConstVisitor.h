#pragma once

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

class DependencyOverlay;

template<typename T>
class CompoundDependencyOverlay;

class DependencyOverlayConstVisitor
{
public:
	virtual void Visit(const DependencyOverlay *overlay);

	virtual void Visit(const CompoundDependencyOverlay<HorseIR::Function> *overlay);
	virtual void Visit(const CompoundDependencyOverlay<HorseIR::IfStatement> *overlay);
	virtual void Visit(const CompoundDependencyOverlay<HorseIR::WhileStatement> *overlay);
	virtual void Visit(const CompoundDependencyOverlay<HorseIR::RepeatStatement> *overlay);
};

}
