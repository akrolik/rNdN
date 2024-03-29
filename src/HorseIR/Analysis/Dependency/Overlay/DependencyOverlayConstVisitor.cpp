#include "HorseIR/Analysis/Dependency/Overlay/DependencyOverlayConstVisitor.h"

#include "HorseIR/Analysis/Dependency/Overlay/DependencyOverlay.h"

namespace HorseIR {
namespace Analysis {

void DependencyOverlayConstVisitor::Visit(const DependencyOverlay *overlay)
{

}

void DependencyOverlayConstVisitor::Visit(const FunctionDependencyOverlay *overlay)
{
	Visit(static_cast<const DependencyOverlay *>(overlay));
}

void DependencyOverlayConstVisitor::Visit(const IfDependencyOverlay *overlay)
{
	Visit(static_cast<const DependencyOverlay *>(overlay));
}

void DependencyOverlayConstVisitor::Visit(const WhileDependencyOverlay *overlay)
{
	Visit(static_cast<const DependencyOverlay *>(overlay));
}

void DependencyOverlayConstVisitor::Visit(const RepeatDependencyOverlay *overlay)
{
	Visit(static_cast<const DependencyOverlay *>(overlay));
}

}
}
