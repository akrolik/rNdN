#include "Analysis/Compatibility/Overlay/CompatibilityOverlayVisitor.h"

#include "Analysis/Compatibility/Overlay/CompatibilityOverlay.h"

namespace Analysis {

void CompatibilityOverlayVisitor::Visit(CompatibilityOverlay *overlay)
{

}

void CompatibilityOverlayVisitor::Visit(FunctionCompatibilityOverlay *overlay)
{
	Visit(static_cast<CompatibilityOverlay *>(overlay));
}

void CompatibilityOverlayVisitor::Visit(IfCompatibilityOverlay *overlay)
{
	Visit(static_cast<CompatibilityOverlay *>(overlay));
}

void CompatibilityOverlayVisitor::Visit(WhileCompatibilityOverlay *overlay)
{
	Visit(static_cast<CompatibilityOverlay *>(overlay));
}

void CompatibilityOverlayVisitor::Visit(RepeatCompatibilityOverlay *overlay)
{
	Visit(static_cast<CompatibilityOverlay *>(overlay));
}

}
