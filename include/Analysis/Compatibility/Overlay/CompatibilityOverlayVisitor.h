#pragma once

namespace Analysis {

class CompatibilityOverlay;

class FunctionCompatibilityOverlay;
class KernelCompatibilityOverlay;
class IfCompatibilityOverlay;
class WhileCompatibilityOverlay;
class RepeatCompatibilityOverlay;

class CompatibilityOverlayVisitor
{
public:
	virtual void Visit(CompatibilityOverlay *node);

	virtual void Visit(FunctionCompatibilityOverlay *node);
	virtual void Visit(KernelCompatibilityOverlay *node);
	virtual void Visit(IfCompatibilityOverlay *node);
	virtual void Visit(WhileCompatibilityOverlay *node);
	virtual void Visit(RepeatCompatibilityOverlay *node);
};

}
