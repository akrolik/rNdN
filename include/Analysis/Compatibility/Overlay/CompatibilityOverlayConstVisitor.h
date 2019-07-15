#pragma once

namespace Analysis {

class CompatibilityOverlay;

class FunctionCompatibilityOverlay;
class KernelCompatibilityOverlay;
class IfCompatibilityOverlay;
class WhileCompatibilityOverlay;
class RepeatCompatibilityOverlay;

class CompatibilityOverlayConstVisitor
{
public:
	virtual void Visit(const CompatibilityOverlay *node);

	virtual void Visit(const FunctionCompatibilityOverlay *node);
	virtual void Visit(const KernelCompatibilityOverlay *node);
	virtual void Visit(const IfCompatibilityOverlay *node);
	virtual void Visit(const WhileCompatibilityOverlay *node);
	virtual void Visit(const RepeatCompatibilityOverlay *node);
};

}
