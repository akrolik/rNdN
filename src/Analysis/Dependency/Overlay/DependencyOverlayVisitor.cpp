#include "Analysis/Dependency/Overlay/DependencyOverlayVisitor.h"

#include "Analysis/Dependency/Overlay/DependencyOverlay.h"

namespace Analysis {

void DependencyOverlayVisitor::Visit(DependencyOverlay *overlay)
{

}

void DependencyOverlayVisitor::Visit(FunctionDependencyOverlay *overlay)
{
	Visit(static_cast<DependencyOverlay *>(overlay));
}

void DependencyOverlayVisitor::Visit(IfDependencyOverlay *overlay)
{
	Visit(static_cast<DependencyOverlay *>(overlay));
}

void DependencyOverlayVisitor::Visit(WhileDependencyOverlay *overlay)
{
	Visit(static_cast<DependencyOverlay *>(overlay));
}

void DependencyOverlayVisitor::Visit(RepeatDependencyOverlay *overlay)
{
	Visit(static_cast<DependencyOverlay *>(overlay));
}

}
