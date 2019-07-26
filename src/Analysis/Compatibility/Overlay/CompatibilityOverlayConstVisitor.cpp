#include "Analysis/Compatibility/Overlay/CompatibilityOverlayConstVisitor.h"

#include "Analysis/Compatibility/Overlay/CompatibilityOverlay.h"

namespace Analysis {

void CompatibilityOverlayConstVisitor::Visit(const CompatibilityOverlay *overlay)
{

}

void CompatibilityOverlayConstVisitor::Visit(const FunctionCompatibilityOverlay *overlay)
{
	Visit(static_cast<const CompatibilityOverlay *>(overlay));
}

void CompatibilityOverlayConstVisitor::Visit(const IfCompatibilityOverlay *overlay)
{
	Visit(static_cast<const CompatibilityOverlay *>(overlay));
}

void CompatibilityOverlayConstVisitor::Visit(const WhileCompatibilityOverlay *overlay)
{
	Visit(static_cast<const CompatibilityOverlay *>(overlay));
}

void CompatibilityOverlayConstVisitor::Visit(const RepeatCompatibilityOverlay *overlay)
{
	Visit(static_cast<const CompatibilityOverlay *>(overlay));
}

}
