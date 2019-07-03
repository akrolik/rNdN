#include "Analysis/Dependency/DependencyOverlayConstVisitor.h"

#include "Analysis/Dependency/DependencyOverlay.h"

namespace Analysis {

void DependencyOverlayConstVisitor::Visit(const DependencyOverlay *overlay)
{

}

void DependencyOverlayConstVisitor::Visit(const CompoundDependencyOverlay<HorseIR::Function> *overlay)
{
	Visit(static_cast<const DependencyOverlay *>(overlay));
}

void DependencyOverlayConstVisitor::Visit(const CompoundDependencyOverlay<HorseIR::IfStatement> *overlay)
{
	Visit(static_cast<const DependencyOverlay *>(overlay));
}

void DependencyOverlayConstVisitor::Visit(const CompoundDependencyOverlay<HorseIR::WhileStatement> *overlay)
{
	Visit(static_cast<const DependencyOverlay *>(overlay));
}

void DependencyOverlayConstVisitor::Visit(const CompoundDependencyOverlay<HorseIR::RepeatStatement> *overlay)
{
	Visit(static_cast<const DependencyOverlay *>(overlay));
}

}
