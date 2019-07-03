#include "Analysis/Dependency/DependencyOverlayVisitor.h"

#include "Analysis/Dependency/DependencyOverlay.h"

namespace Analysis {

void DependencyOverlayVisitor::Visit(DependencyOverlay *overlay)
{

}

void DependencyOverlayVisitor::Visit(CompoundDependencyOverlay<HorseIR::Function> *overlay)
{
	Visit(static_cast<DependencyOverlay *>(overlay));
}

void DependencyOverlayVisitor::Visit(CompoundDependencyOverlay<HorseIR::IfStatement> *overlay)
{
	Visit(static_cast<DependencyOverlay *>(overlay));
}

void DependencyOverlayVisitor::Visit(CompoundDependencyOverlay<HorseIR::WhileStatement> *overlay)
{
	Visit(static_cast<DependencyOverlay *>(overlay));
}

void DependencyOverlayVisitor::Visit(CompoundDependencyOverlay<HorseIR::RepeatStatement> *overlay)
{
	Visit(static_cast<DependencyOverlay *>(overlay));
}

}
